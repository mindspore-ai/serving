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
#include <unistd.h>
#include <memory>
#include <nlohmann/json.hpp>
#include "master/restful/http_handle.h"
namespace mindspore::serving {

int id = 0;
Status RestfulServer::Run(const std::shared_ptr<RestfulRequest> &restful_request) {
  Status status(SUCCESS);
  // MSI_TIME_STAMP_START(DoSometing)
  // // ke cai start --- sync
  // ++id;
  // for (int i = 0; i < 10; ++i) {
  //   MSI_LOG_INFO << "id:" << id << "--" << i << " ";
  //   usleep(600);
  // }
  // MSI_TIME_STAMP_END(DoSometing)
  // // -----------------

  // // st error
  // if (id == 5) {
  //   ERROR_INFER_STATUS(status, INVALID_INPUTS, "id == 5");
  //   if ((status = restful_request->ErrorMessage(status)) != SUCCESS) {
  //     return status;
  //   }
  //   return status;
  // }
  nlohmann::json predict_json;
  std::string predict_str;
  status = HandleRestfulRequest(restful_request, &predict_json);
  if (status != SUCCESS) {
    predict_str = status.StatusMessage();
    nlohmann::json js;
    js["error_msg"] = predict_str;
    predict_str = js.dump();
  } else {
    predict_str = predict_json.dump();
  }
  restful_request->RestfulReplay(predict_str);
  return status;
}

void RestfulServer::Committer(const std::shared_ptr<RestfulRequest> &restful_request) {
  thread_pool_.commit([restful_request, this]() { Run(restful_request); });
}

void RestfulServer::DispatchEvHttpRequest(evhttp_request *request) {
  Status status(SUCCESS);

  auto de_request = std::make_unique<DecomposeEvRequest>(request, max_msg_size_);
  if (de_request == nullptr) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "de_request is nullptr.");
    return;
  }
  Status de_status = de_request->Decompose();
  auto restful_request = std::make_shared<RestfulRequest>(std::move(de_request));
  if (restful_request == nullptr) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "restful_request is nullptr.");
    return;
  }
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
             << "Serving Error: RESTful server start failed, create http server faild";
    free_event_base();
    return status;
  }

  evhttp_set_gencb(event_http_, &EvCallBack, this);
  evhttp_set_timeout(event_http_, time_out_second);
  return SUCCESS;
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
  struct sockaddr_in sin = {};
  sin.sin_family = AF_INET;
  sin.sin_port = htons(restful_port_);
  auto listener = evconnlistener_new_bind(event_base_, nullptr, nullptr,
                                          LEV_OPT_REUSEABLE | LEV_OPT_CLOSE_ON_EXEC | LEV_OPT_CLOSE_ON_FREE, -1,
                                          reinterpret_cast<struct sockaddr *>(&sin), sizeof(sin));

  if (listener == nullptr) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
             << "Serving Error: RESTful server start failed, create http listener faild, port " << restful_port_;
    free_event_base();
    free_evhttp();
    return status;
  }
  auto bound = evhttp_bind_listener(event_http_, listener);
  if (bound == nullptr) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
             << "Serving Error: RESTful server start failed, bind http listener to server faild, port "
             << restful_port_;
    evconnlistener_free(listener);
    free_event_base();
    free_evhttp();
    return status;
  }

  auto event_http_run = [this]() {
    MSI_LOG(INFO) << "Serving RESTful server listening on " << restful_ip_ << ":" << restful_port_;
    std::cout << "Serving: Serving RESTful server start success, listening on " << restful_ip_ << ":" << restful_port_
              << std::endl;
    event_base_dispatch(event_base_);
  };
  event_thread_ = std::thread(event_http_run);
  return SUCCESS;
}

Status RestfulServer::Start(const std::string &ip, uint32_t restful_port, int max_msg_size, int time_out_second) {
  Status status(SUCCESS);
  if (in_running_) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Serving Error: RESTful server is already running";
  }

  restful_ip_ = ip;
  restful_port_ = restful_port;
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
