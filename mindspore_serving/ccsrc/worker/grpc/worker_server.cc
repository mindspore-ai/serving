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

#include "worker/grpc/worker_server.h"
#include <string>
#include <memory>
#include "common/grpc_server.h"

namespace mindspore {
namespace serving {
MSWorkerServer::~MSWorkerServer() { Stop(); }

MSWorkerServer::MSWorkerServer(const std::string &hostname, int32_t port) {
  service_impl_ = std::make_unique<MSWorkerImpl>();
  async_server_ = std::make_unique<WorkerGrpcServer>(hostname, port, service_impl_.get());
}
Status MSWorkerServer::Init() {
  Status status = async_server_->Run("Worker gRPC", gRpcMaxMBMsgSize);
  if (status != SUCCESS) return status;
  auto grpc_server_run = [this]() { StartAsyncRpcService(); };
  grpc_thread_ = std::thread(grpc_server_run);
  in_running_ = true;
  return SUCCESS;
}
Status MSWorkerServer::StartAsyncRpcService() {
  Status status = async_server_->HandleRequest();
  return status;
}
Status MSWorkerServer::Stop() {
  if (in_running_) {
    async_server_->Stop();
    grpc_thread_.join();
  }
  in_running_ = false;
  return SUCCESS;
}

}  // namespace serving
}  // namespace mindspore
