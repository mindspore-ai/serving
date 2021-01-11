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

class GrpcAsyncServer {
 public:
  explicit GrpcAsyncServer(const std::string &host, uint32_t port);
  virtual ~GrpcAsyncServer();
  /// \brief Brings up gRPC server
  /// \return none
  Status Run();
  /// \brief Entry function to handle async server request
  Status HandleRequest();

  void Stop();

  virtual Status RegisterService(grpc::ServerBuilder *builder) = 0;

  virtual Status EnqueueRequest() = 0;

  virtual Status ProcessRequest(void *tag) = 0;

  virtual Status SendFinish() = 0;

 protected:
  std::string host_;
  uint32_t port_;
  std::unique_ptr<grpc::ServerCompletionQueue> cq_;
  std::unique_ptr<grpc::Server> server_;
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_GRPC_ASYNC_SERVER_H
