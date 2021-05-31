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

namespace mindspore::serving {

class GrpcAsyncServiceContextBase {
 public:
  GrpcAsyncServiceContextBase() = default;
  virtual ~GrpcAsyncServiceContextBase() = default;

  virtual void HandleRequest() = 0;
  virtual bool JudgeFinish() const = 0;
};

template <class ServiceImpl, class AsyncService, class Derived>
class GrpcAsyncServiceContext : public GrpcAsyncServiceContextBase {
 public:
  enum class STATE : int8_t { CREATE = 1, PROCESS = 2, FINISH = 3 };

  GrpcAsyncServiceContext(ServiceImpl *service_impl, AsyncService *async_service, grpc::ServerCompletionQueue *cq)
      : service_impl_(service_impl), async_service_(async_service), cq_(cq) {
    state_ = STATE::CREATE;
  }
  ~GrpcAsyncServiceContext() = default;
  GrpcAsyncServiceContext() = delete;

  virtual void StartEnqueueRequest() = 0;

  bool JudgeFinish() const override { return state_ == STATE::FINISH; }

  static Status EnqueueRequest(ServiceImpl *service_impl, AsyncService *async_service,
                               grpc::ServerCompletionQueue *cq) {
    auto call = new Derived(service_impl, async_service, cq);
    call->StartEnqueueRequest();
    return SUCCESS;
  }

 protected:
  STATE state_;
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

  virtual Status EnqueueRequest() = 0;

  bool IsRunning() const { return in_running_; }

  /// \brief Brings up gRPC server
  /// \return none
  Status Start(const std::string &socket_address, int max_msg_mb_size, const std::string &server_tag) {
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
    builder.AddListeningPort(socket_address, grpc::InsecureServerCredentials(), &port_tcpip);
    Status status = RegisterService(&builder);
    if (status != SUCCESS) return status;
    cq_ = builder.AddCompletionQueue();
    server_ = builder.BuildAndStart();
    if (server_) {
      MSI_LOG(INFO) << server_tag << " server start success, listening on " << socket_address;
    } else {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Serving Error: " << server_tag
                                            << " server start failed, create server failed, address " << socket_address;
    }
    auto grpc_server_run = [this]() { HandleRequest(); };
    grpc_thread_ = std::thread(grpc_server_run);
    in_running_ = true;
    return SUCCESS;
  }
  /// \brief Entry function to handle async server request
  Status HandleRequest() {
    bool success = false;
    void *tag;
    // We loop through the grpc queue. Each connection if successful
    // will come back with our own tag which is an instance of GrpcAsyncServiceContextBase
    // and we simply call its functor. But first we need to create these instances
    // and inject them into the grpc queue.
    Status status = EnqueueRequest();
    if (status != SUCCESS) return status;
    while (cq_->Next(&tag, &success)) {
      if (success) {
        status = ProcessRequest(tag);
        if (status != SUCCESS) return status;
      } else {
        MSI_LOG(DEBUG) << "cq_->Next failed.";
      }
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

      if (grpc_thread_.joinable()) {
        grpc_thread_.join();
      }
    }
    in_running_ = false;
  }

  Status RegisterService(grpc::ServerBuilder *builder) {
    builder->RegisterService(&svc_);
    return SUCCESS;
  }

  Status ProcessRequest(void *tag) {
    auto rq = static_cast<GrpcAsyncServiceContextBase *>(tag);
    if (rq->JudgeFinish()) {
      delete rq;
    } else {
      rq->HandleRequest();
    }
    return SUCCESS;
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
