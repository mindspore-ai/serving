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

#include "master/restful/restful_server.h"
#include <sys/un.h>
#include <memory>
#include <nlohmann/json.hpp>
#include "master/restful/http_handle.h"
namespace mindspore::serving {

Status RestfulServer::Run(const std::shared_ptr<RestfulRequest> &restful_request) {
  return HandleRestfulRequest(restful_request);
}

void RestfulServer::Committer(const std::shared_ptr<RestfulRequest> &restful_request) {
  thread_pool_.commit([restful_request, this]() { Run(restful_request); });
}

void RestfulServer::DispatchEvHttpRequest(evhttp_request *request) {
  Status status(SUCCESS);

  auto de_request = std::make_unique<DecomposeEvRequest>(request, max_msg_size_);
  Status de_status = de_request->Decompose();
  auto restful_request = std::make_shared<RestfulRequest>(std::move(de_request));
  status = restful_request->RestfulReplayBufferInit();
  if (status != SUCCESS) {
    if ((status = restful_request->ErrorMessage(status)) != SUCCESS) {
      return;
    }
    return;
  }

  if (de_status != SUCCESS) {
    (void)restful_request->ErrorMessage(de_status);
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
  auto free_event_base = [this] {
    if (event_base_ != nullptr) {
      event_base_free(event_base_);
      event_base_ = nullptr;
    }
  };

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
    free_event_base();
    return status;
  }

  evhttp_set_gencb(event_http_, &EvCallBack, this);
  evhttp_set_timeout(event_http_, time_out_second);
  return SUCCESS;
}

bool RestfulServer::IsUnixAddress(const std::string &socket_address) {
  std::string prefix = "unix:";
  return socket_address.substr(0, prefix.size()) == prefix;
}

bool RestfulServer::GetSocketAddress(const std::string &socket_address, uint32_t *ip, uint16_t *port) {
  auto mid = socket_address.find(':');
  if (mid == std::string::npos) {
    return false;
  }
  auto addr_str = socket_address.substr(0, mid);
  auto port_str = socket_address.substr(mid + 1);
  if (addr_str.empty() || port_str.empty()) {
    return false;
  }
  auto str_2_uint32 = [](const std::string &str, size_t start, size_t end, uint32_t *num, uint32_t max) -> bool {
    uint32_t num_ret = 0;
    for (size_t i = start; i < end; i++) {
      auto c = str[i];
      if (c < '0' || c > '9') {
        return false;
      }
      num_ret = num_ret * 10 + c - '0';
      if (num_ret > max) {
        return false;
      }
    }
    *num = num_ret;
    return true;
  };
  uint32_t port_ret = 0;
  if (!str_2_uint32(port_str, 0, port_str.size(), &port_ret, 65535)) {
    return false;
  }
  *port = static_cast<uint16_t>(port_ret);
  if (addr_str == "localhost") {
    *ip = (127 << 24) + 1;  // 127.0.0.1
  } else {
    uint32_t ip_sum = 0;
    uint32_t dot_cnt = 0;
    size_t cur_index = 0;
    for (; dot_cnt < 3; dot_cnt++) {
      auto dot_pos = addr_str.find('.', cur_index);
      if (dot_pos == std::string::npos || dot_pos == cur_index || dot_pos == addr_str.size() - 1) {
        return false;
      }
      uint32_t num = 0;
      if (!str_2_uint32(addr_str, cur_index, dot_pos, &num, 255)) {
        return false;
      }
      ip_sum = (ip_sum << 8) + num;
      cur_index = dot_pos + 1;
    }
    uint32_t num = 0;
    if (!str_2_uint32(addr_str, cur_index, addr_str.size(), &num, 255)) {
      return false;
    }
    ip_sum = (ip_sum << 8) + num;
    *ip = ip_sum;
  }
  return true;
}

Status RestfulServer::StartRestfulServer() {
  auto free_event_base = [this] {
    if (event_base_ != nullptr) {
      event_base_free(event_base_);
      event_base_ = nullptr;
    }
  };
  auto free_evhttp = [this] {
    if (event_http_ != nullptr) {
      evhttp_free(event_http_);
      event_http_ = nullptr;
    }
  };

  Status status(SUCCESS);
  uint32_t ip = 0;
  uint16_t port = 0;
  struct evconnlistener *listener;
  if (IsUnixAddress(socket_address_)) {
    struct sockaddr_un sun = {};
    std::string address = socket_address_.substr(std::string("unix:").size());
    if (address.size() >= sizeof(sun.sun_path)) {
      status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
               << "Serving Error: RESTful server start failed, unix domain socket address length>="
               << sizeof(sun.sun_path) << ", address " << socket_address_;
      free_event_base();
      free_evhttp();
      return status;
    }
    sun.sun_family = AF_UNIX;
    memset_s(sun.sun_path, sizeof(sun.sun_path), 0, sizeof(sun.sun_path));
    auto ret = memcpy_s(sun.sun_path, sizeof(sun.sun_path), address.c_str(), address.size());
    if (ret != EOK) {
      status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
               << "Serving Error: RESTful server start failed, unix domain socket address " << socket_address_;
      free_event_base();
      free_evhttp();
      return status;
    }
    listener = evconnlistener_new_bind(event_base_, nullptr, nullptr,
                                       LEV_OPT_REUSEABLE | LEV_OPT_CLOSE_ON_EXEC | LEV_OPT_CLOSE_ON_FREE, -1,
                                       reinterpret_cast<struct sockaddr *>(&sun), sizeof(sun));
  } else {
    if (!GetSocketAddress(socket_address_, &ip, &port)) {
      status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
               << "Serving Error: RESTful server start failed, failed to parse socket address " << socket_address_;
      free_event_base();
      free_evhttp();
      return status;
    }
    if (port == 0) {
      MSI_LOG_EXCEPTION << "restful port should be in range [1,65535], given address " << socket_address_;
    }
    MSI_LOG_INFO << "Get RESTful server ip " << ip << " and port  " << port << "success, given address "
                 << socket_address_;
    struct sockaddr_in sin = {};
    sin.sin_family = AF_INET;
    // not work: sin.sin_addr.s_addr = htons(ip);
    sin.sin_port = htons(port);
    listener = evconnlistener_new_bind(event_base_, nullptr, nullptr,
                                       LEV_OPT_REUSEABLE | LEV_OPT_CLOSE_ON_EXEC | LEV_OPT_CLOSE_ON_FREE, -1,
                                       reinterpret_cast<struct sockaddr *>(&sin), sizeof(sin));
  }
  if (listener == nullptr) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
             << "Serving Error: RESTful server start failed, create http listener failed, address " << socket_address_;
    free_event_base();
    free_evhttp();
    return status;
  }
  auto bound = evhttp_bind_listener(event_http_, listener);
  if (bound == nullptr) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
             << "Serving Error: RESTful server start failed, bind http listener to server failed, address "
             << socket_address_;
    evconnlistener_free(listener);
    free_event_base();
    free_evhttp();
    return status;
  }

  auto event_http_run = [this]() {
    MSI_LOG(INFO) << "Serving RESTful server start success, listening on " << socket_address_;
    std::cout << "Serving: Serving RESTful server start success, listening on " << socket_address_ << std::endl;
    event_base_dispatch(event_base_);
  };
  event_thread_ = std::thread(event_http_run);
  return SUCCESS;
}

Status RestfulServer::Start(const std::string &socket_address, int max_msg_size, int time_out_second) {
  Status status(SUCCESS);
  if (in_running_) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Serving Error: RESTful server is already running";
  }

  socket_address_ = socket_address;
  max_msg_size_ = static_cast<int>(max_msg_size * (uint32_t(1) << 20));

  status = CreatRestfulServer(time_out_second);
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
  if (event_base_ != nullptr) {
    event_base_free(event_base_);
    event_base_ = nullptr;
  }
  if (event_http_ != nullptr) {
    evhttp_free(event_http_);
    event_http_ = nullptr;
  }
}

}  // namespace mindspore::serving
