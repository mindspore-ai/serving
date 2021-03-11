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
#ifndef MINDSPORE_SERVING_MASTER_SERVER_H
#define MINDSPORE_SERVING_MASTER_SERVER_H

#include <memory>
#include <string>
#include "common/serving_common.h"
#include "common/grpc_server.h"
#include "master/restful/restful_server.h"
#include "master/dispacther.h"

namespace mindspore {
namespace serving {

class MS_API Server {
 public:
  Server();
  ~Server();
  Status StartGrpcServer(const std::string &ip, uint32_t grpc_port, int max_msg_mb_size = gRpcDefaultMsgMBSize,
                         uint32_t max_infer_num = g_max_infer_num_);
  Status StartGrpcMasterServer(const std::string &ip, uint32_t grpc_port);
  Status StartRestfulServer(const std::string &ip, uint32_t restful_port, int max_msg_mb_size = gRpcDefaultMsgMBSize,
                            uint32_t max_infer_num = g_max_infer_num_, int time_out_second = 100);
  void Clear();

  std::shared_ptr<Dispatcher> GetDispatcher() { return dispatcher_; }

  static Server &Instance();

 private:
  GrpcServer grpc_server_;
  GrpcServer grpc_manager_server_;
  RestfulServer restful_server_;
  std::shared_ptr<Dispatcher> dispatcher_ = std::make_shared<Dispatcher>();
};
}  // namespace serving
}  // namespace mindspore
#endif  // MINDSPORE_SERVING_MASTER_SERVER_H
