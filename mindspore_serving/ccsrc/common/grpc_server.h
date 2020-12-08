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

#ifndef MINDSPORE_SERVING_GRPC_SERVER_H
#define MINDSPORE_SERVING_GRPC_SERVER_H

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <memory>
#include <utility>
#include <string>
#include <future>
#include "common/serving_common.h"

namespace mindspore::serving {

constexpr int gRpcDefaultMsgMBSize = 100;
constexpr int gRpcMaxMBMsgSize = 512;  // max 512 MB

class MS_API GrpcServer {
 public:
  GrpcServer() = default;
  ~GrpcServer() { Stop(); }

  Status Start(std::shared_ptr<grpc::Service> service, const std::string &ip, uint32_t grpc_port, int max_msg_size,
               const std::string &server_tag);
  void Stop();

 private:
  std::unique_ptr<grpc::Server> server_;
  std::thread grpc_thread_;
  bool in_running_ = false;
  std::shared_ptr<grpc::Service> service_;
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_GRPC_SERVER_H
