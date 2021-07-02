/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_SERVING_GRPC_ASYNC_SERVER_H
#define MINDSPORE_SERVING_GRPC_ASYNC_SERVER_H

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <memory>
#include <utility>
#include <string>
#include <future>
#include "common/serving_common.h"
#include "common/ssl_config.h"
#include "common/utils.h"

namespace mindspore::serving {

class GrpcAsyncServiceContextBase {
 public:
  GrpcAsyncServiceContextBase() = default;
  virtual ~GrpcAsyncServiceContextBase() = default;

  virtual void NewAndHandleRequest() = 0;

  bool HasFinish() const { return finished_; }
  void SetFinish() { finished_ = true; }

 private:
  bool finished_ = false;
};

template <class ServiceImpl, class AsyncService, class Derived>
class GrpcAsyncServiceContext : public GrpcAsyncServiceContextBase {
 public:
  GrpcAsyncServiceContext(ServiceImpl *service_impl, AsyncService *async_service, grpc::ServerCompletionQueue *cq)
      : service_impl_(service_impl), async_service_(async_service), cq_(cq) {}
  ~GrpcAsyncServiceContext() = default;
  GrpcAsyncServiceContext() = delete;

  virtual void StartEnqueueRequest() = 0;
  virtual void HandleRequest() = 0;

  static void EnqueueRequest(ServiceImpl *service_impl, AsyncService *async_service, grpc::ServerCompletionQueue *cq) {
    auto call = new Derived(service_impl, async_service, cq);
    call->StartEnqueueRequest();
  }

  void NewAndHandleRequest() final {
    EnqueueRequest(service_impl_, async_service_, cq_);
    HandleRequest();
  }

 protected:
  grpc::ServerContext ctx_;

  ServiceImpl *service_impl_;
  AsyncService *async_service_;
  grpc::ServerCompletionQueue *cq_;
};

template <class AsyncService>
class GrpcAsyncServer {
 public:
  GrpcAsyncServer() {}
  virtual ~GrpcAsyncServer() { Stop(); }

  virtual void EnqueueRequests() = 0;

  Status Start(const std::string &socket_address, const SSLConfig &ssl_config, int max_msg_mb_size,
               const std::string &server_tag) {
    if (in_running_) {
      return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Serving Error: " << server_tag << " server is already running";
    }

    grpc::ServerBuilder builder;
    if (max_msg_mb_size > 0) {
      builder.SetMaxSendMessageSize(static_cast<int>(max_msg_mb_size * (1u << 20)));
      builder.SetMaxReceiveMessageSize(static_cast<int>(max_msg_mb_size * (1u << 20)));
    }
    builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
    int port_tcpip = 0;
    auto creds = BuildServerCredentialsFromSSLConfigFile(ssl_config);

    Status status;
    status = CheckServerAddress(socket_address, server_tag);
    if (status != SUCCESS) {
      return status;
    }
    builder.AddListeningPort(socket_address, creds, &port_tcpip);
    status = RegisterService(&builder);
    if (status != SUCCESS) return status;
    cq_ = builder.AddCompletionQueue();
    server_ = builder.BuildAndStart();
    if (!server_) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Serving Error: " << server_tag
                                            << " server start failed, create server failed, address " << socket_address;
    }
    auto grpc_server_run = [this]() { HandleRequests(); };
    grpc_thread_ = std::thread(grpc_server_run);
    in_running_ = true;
    MSI_LOG(INFO) << server_tag << " server start success, listening on " << socket_address;
    std::cout << "Serving: " << server_tag << " server start success, listening on " << socket_address << std::endl;
    return SUCCESS;
  }

  Status CheckServerAddress(const std::string &address, const std::string &server_tag) {
    Status status;
    std::string prefix = "unix:";
    if (address.substr(0, prefix.size()) == prefix) {
      if (address.size() > prefix.size()) {
        return SUCCESS;
      } else {
        status = INFER_STATUS_LOG_ERROR(FAILED) << "Serving Error: Empty grpc server unix domain socket address";
        return status;
      }
    }
    status = common::CheckAddress(address, server_tag, nullptr, nullptr);
    if (status != SUCCESS) {
      return status;
    }
    return SUCCESS;
  }

  std::shared_ptr<grpc::ServerCredentials> BuildServerCredentialsFromSSLConfigFile(const SSLConfig &ssl_config) {
    if (!ssl_config.use_ssl) {
      return grpc::InsecureServerCredentials();
    }
    grpc::SslServerCredentialsOptions ssl_ops(ssl_config.verify_client
                                                ? GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY
                                                : GRPC_SSL_DONT_REQUEST_CLIENT_CERTIFICATE);

    if (!ssl_config.custom_ca.empty()) {
      ssl_ops.pem_root_certs = ssl_config.custom_ca;
    }
    grpc::SslServerCredentialsOptions::PemKeyCertPair keycert = {ssl_config.private_key, ssl_config.certificate};
    ssl_ops.pem_key_cert_pairs.push_back(keycert);

    return grpc::SslServerCredentials(ssl_ops);
  }

  Status HandleRequests() {
    void *tag;
    bool ok = false;
    EnqueueRequests();
    while (cq_->Next(&tag, &ok)) {
      ProcessRequest(tag, ok);
    }
    return SUCCESS;
  }

  void Stop() {
    if (in_running_) {
      if (server_) {
        server_->Shutdown();
      }
      // Always shutdown the completion queue after the server.
      if (cq_) {
        cq_->Shutdown();
      }
      grpc_thread_.join();
    }
    in_running_ = false;
  }

  Status RegisterService(grpc::ServerBuilder *builder) {
    builder->RegisterService(&svc_);
    return SUCCESS;
  }

  void ProcessRequest(void *tag, bool rpc_ok) {
    auto rq = static_cast<GrpcAsyncServiceContextBase *>(tag);
    if (rq->HasFinish() || !rpc_ok) {  // !rpc_ok: cancel get request when shutting down.
      delete rq;
    } else {
      rq->NewAndHandleRequest();
      rq->SetFinish();  // will delete next time
    }
  }

 protected:
  std::unique_ptr<grpc::ServerCompletionQueue> cq_;
  std::unique_ptr<grpc::Server> server_;

  AsyncService svc_;

  bool in_running_ = false;
  std::thread grpc_thread_;
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_GRPC_ASYNC_SERVER_H
