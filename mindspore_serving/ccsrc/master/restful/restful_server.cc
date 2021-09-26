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

#include <memory>
#include <nlohmann/json.hpp>
#include "openssl/ssl.h"
#include "openssl/err.h"
#include "event2/bufferevent.h"
#include "event2/http.h"
#include "event2/bufferevent_ssl.h"
#include "master/restful/http_handle.h"
#include "master/restful/restful_server.h"
#include "common/utils.h"
#include "master/restful/http_process.h"

namespace mindspore::serving {

void RestfulServer::Committer(const std::shared_ptr<RestfulRequest> &restful_request) {
  thread_pool_.commit([restful_request]() { RestfulService::RunRestful(restful_request); });
}

void RestfulServer::DispatchEvHttpRequest(evhttp_request *request) {
  Status status(SUCCESS);

  auto de_request = std::make_unique<DecomposeEvRequest>(request, max_msg_size_);
  Status de_status = de_request->Decompose();
  auto restful_request = std::make_shared<RestfulRequest>(std::move(de_request));
  status = restful_request->RestfulReplayBufferInit();
  if (status != SUCCESS) {
    restful_request->ErrorMessage(status);
    return;
  }

  if (de_status != SUCCESS) {
    restful_request->ErrorMessage(de_status);
    return;
  }
  Committer(restful_request);
}

void RestfulServer::EvCallBack(evhttp_request *request, void *arg) {
  auto *restful_server = static_cast<RestfulServer *>(arg);
  restful_server->DispatchEvHttpRequest(const_cast<evhttp_request *>(request));
}

Status RestfulServer::CreatRestfulServer(int time_out_second) {
  evthread_use_pthreads();
  auto status = InitEvHttp();
  if (status != SUCCESS) {
    return status;
  }
  evhttp_set_gencb(event_http_, &EvCallBack, this);
  evhttp_set_timeout(event_http_, time_out_second);
  return SUCCESS;
}

Status RestfulServer::CreatHttpsServer(int time_out_second, const SSLConfig &ssl_config) {
  InitOpenSSL();
  evthread_use_pthreads();

  Status status;
  status = InitEvHttp();
  if (status != SUCCESS) {
    return status;
  }

  SSL_CTX *ctx = SSL_CTX_new(SSLv23_method());
  SSL_CTX_set_options(ctx, SSL_OP_SINGLE_DH_USE | SSL_OP_SINGLE_ECDH_USE | SSL_OP_NO_SSLv3 | SSL_OP_NO_TLSv1);

  if (ssl_config.verify_client) {
    SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, nullptr);
    if (!ssl_config.custom_ca.empty() &&
        SSL_CTX_load_verify_locations(ctx, ssl_config.custom_ca.c_str(), nullptr) != 1) {
      status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
               << "Serving Error: load root certificate from " << ssl_config.custom_ca << " failed";
      return status;
    } else {
      if (SSL_CTX_set_default_verify_paths(ctx) != 1) {
        status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Serving Error: set default verify paths failed";
        return status;
      }
    }
  }
  EC_KEY *ecdh = EC_KEY_new_by_curve_name(NID_X9_62_prime256v1);
  if (ecdh == nullptr) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Serving Error: EC_KEY_new_by_curve_name failed";
    return status;
  }
  if (!SSL_CTX_set_tmp_ecdh(ctx, ecdh)) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Serving Error: SSL_CTX_set_tmp_ecdh failed";
    return status;
  }

  status = ServerSetupCerts(ctx, ssl_config);
  if (status != SUCCESS) {
    return status;
  }

  evhttp_set_bevcb(event_http_, bevcb, ctx);
  evhttp_set_gencb(event_http_, &EvCallBack, this);
  evhttp_set_timeout(event_http_, time_out_second);
  return SUCCESS;
}

Status RestfulServer::ServerSetupCerts(SSL_CTX *ctx, const SSLConfig &ssl_config) {
  Status status;
  if (SSL_CTX_use_certificate_chain_file(ctx, ssl_config.certificate.c_str()) != 1) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
             << "Serving Error: load certificate_chain from " << ssl_config.certificate << " failed";
    return status;
  }
  if (SSL_CTX_use_PrivateKey_file(ctx, ssl_config.private_key.c_str(), SSL_FILETYPE_PEM) != 1) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
             << "Serving Error: load private_key from " << ssl_config.private_key << " failed";
    return status;
  }
  if (SSL_CTX_check_private_key(ctx) != 1) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
             << "Serving Error: private_key is not consistent with certificate " << ssl_config.certificate;
    return status;
  }
  return SUCCESS;
}

struct bufferevent *RestfulServer::bevcb(struct event_base *base, void *args) {
  struct bufferevent *r;
  SSL_CTX *ctx = static_cast<SSL_CTX *>(args);
  r = bufferevent_openssl_socket_new(base, -1, SSL_new(ctx), BUFFEREVENT_SSL_ACCEPTING, BEV_OPT_CLOSE_ON_FREE);
  return r;
}

Status RestfulServer::InitEvHttp() {
  event_base_ = event_base_new();
  Status status(SUCCESS);
  if (event_base_ == nullptr) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
             << "Serving Error: RESTful server start failed, new http event failed";
    return status;
  }
  event_http_ = evhttp_new(event_base_);
  if (event_http_ == nullptr) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
             << "Serving Error: RESTful server start failed, create http server failed";
    event_base_free(event_base_);
    event_base_ = nullptr;
    return status;
  }
  return status;
}

void RestfulServer::FreeEvhttp() {
  if (event_http_ != nullptr) {
    evhttp_free(event_http_);
    event_http_ = nullptr;
  }
  if (event_base_ != nullptr) {
    event_base_free(event_base_);
    event_base_ = nullptr;
  }
}

void RestfulServer::RunEvhttp() {
  auto event_http_run = [this]() {
    MSI_LOG(INFO) << "Serving RESTful server listening on " << socket_address_;
    std::cout << "Serving: Serving RESTful server start success, listening on " << socket_address_ << std::endl;
    event_base_dispatch(event_base_);
  };
  event_thread_ = std::thread(event_http_run);
}

Status RestfulServer::StartRestfulServer() {
  Status status(SUCCESS);
  uint16_t port;
  std::string ip;
  status = GetSocketAddress(&ip, &port);
  if (status != SUCCESS) {
    return status;
  }
  auto ret = evhttp_bind_socket(event_http_, ip.c_str(), port);
  if (ret != 0) {
    FreeEvhttp();
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
             << "Serving Error: RESTful server start failed, bind to the socket address " << socket_address_
             << " failed";
    return status;
  }
  RunEvhttp();
  return SUCCESS;
}

Status RestfulServer::GetSocketAddress(std::string *ip, uint16_t *port) {
  MSI_EXCEPTION_IF_NULL(ip);
  MSI_EXCEPTION_IF_NULL(port);
  Status status;
  std::string prefix = "unix:";
  if (socket_address_.substr(0, prefix.size()) == prefix) {
    status = INFER_STATUS_LOG_ERROR(FAILED)
             << "Serving Error: RESTful server does not support binding to unix domain socket";
    return status;
  }
  status = common::CheckAddress(socket_address_, "RESTful server", ip, port);
  if (status != SUCCESS) {
    return status;
  }
  return SUCCESS;
}

Status RestfulServer::Start(const std::string &socket_address, const SSLConfig &ssl_config, int max_msg_size,
                            int time_out_second) {
  Status status(SUCCESS);
  if (in_running_) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Serving Error: RESTful server is already running";
  }
  socket_address_ = socket_address;
  max_msg_size_ = static_cast<int>(max_msg_size * (uint32_t(1) << 20));

  if (ssl_config.use_ssl) {
    status = CreatHttpsServer(time_out_second, ssl_config);
  } else {
    status = CreatRestfulServer(time_out_second);
  }

  if (status != SUCCESS) {
    return status;
  }
  status = StartRestfulServer();
  if (status != SUCCESS) {
    return status;
  }
  in_running_ = true;
  return status;
}

void RestfulServer::Stop() {
  if (in_running_) {
    event_base_loopexit(event_base_, nullptr);
    event_thread_.join();
  }
  in_running_ = false;
  FreeEvhttp();
}

void RestfulServer::InitOpenSSL() {
#if (OPENSSL_VERSION_NUMBER < 0x10100000L) || (defined(LIBRESSL_VERSION_NUMBER) && OPENSSL_VERSION_NUMBER < 0x20700000L)
  SSL_library_init();
  ERR_load_crypto_strings();
  SSL_load_error_strings();
  OpenSSL_add_all_algorithms();
#endif
}

}  // namespace mindspore::serving
