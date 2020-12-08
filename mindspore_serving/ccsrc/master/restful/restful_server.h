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

#ifndef MINDSPORE_SERVING_MASTER_RESTFUL_SERVER_H
#define MINDSPORE_SERVING_MASTER_RESTFUL_SERVER_H

#include <event.h>
#include <event2/event.h>
#include <event2/http.h>
#include <event2/listener.h>
#include <event2/thread.h>
#include <evhttp.h>
#include <memory>
#include <future>
#include <string>
#include <utility>

#include "master/restful/restful_request.h"
#include "common/serving_common.h"
#include "common/thread_pool.h"

namespace mindspore::serving {

constexpr const uint32_t kDefaultRestfulThreadPoolNum = 20;

typedef void (*RestfulFunc)(struct evhttp_request *const req, void *const arg);

class MS_API RestfulServer {
 public:
  RestfulServer() : thread_pool_(kDefaultRestfulThreadPoolNum) {}
  ~RestfulServer() { Stop(); }

  Status Start(const std::string &ip, uint32_t restful_port, int max_msg_size, int time_out_second);
  void Stop();

 private:
  Status CreatRestfulServer(int time_out_second);
  static void EvCallBack(evhttp_request *request, void *arg);
  void DispatchEvHttpRequest(evhttp_request *request);
  void Committer(const std::shared_ptr<RestfulRequest> &restful_request);
  Status Run(const std::shared_ptr<RestfulRequest> &restful_request);
  Status StartRestfulServer();

  std::string restful_ip_;
  uint32_t restful_port_{};
  int max_msg_size_{};
  bool in_running_ = false;

  struct evhttp *event_http_ = nullptr;
  struct event_base *event_base_ = nullptr;
  std::thread event_thread_;
  ThreadPool thread_pool_;
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_MASTER_RESTFUL_SERVER_H
