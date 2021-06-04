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
#include "master/grpc/grpc_server.h"
#include "common/ssl_config.h"

namespace mindspore {
namespace serving {

class MS_API Server {
 public:
  Server();
  ~Server();
  Status StartGrpcServer(const std::string &socket_address, const SSLConfig &ssl_config,
                         int max_msg_mb_size = gRpcDefaultMsgMBSize);
  Status StartRestfulServer(const std::string &socket_address, const SSLConfig &ssl_config,
                            int max_msg_mb_size = gRpcDefaultMsgMBSize, int time_out_second = 100);
  Status StartGrpcMasterServer(const std::string &master_address);
  void Clear();

  std::shared_ptr<Dispatcher> GetDispatcher() { return dispatcher_; }

  static Server &Instance();

 private:
  std::shared_ptr<Dispatcher> dispatcher_ = std::make_shared<Dispatcher>();
  std::shared_ptr<MasterGrpcServer> grpc_async_server_ = nullptr;
  GrpcServer grpc_manager_server_;
  RestfulServer restful_server_;
};
}  // namespace serving
}  // namespace mindspore
#endif  // MINDSPORE_SERVING_MASTER_SERVER_H
