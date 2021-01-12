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

#include "master/grpc/grpc_server.h"
#include <string>
#include <memory>
#include "common/grpc_async_server.h"

namespace mindspore {
namespace serving {
std::unique_ptr<MSServiceServer> grpc_async_server_;

MSServiceServer::~MSServiceServer() { Stop(); }

MSServiceServer::MSServiceServer(std::shared_ptr<MSServiceImpl> service, const std::string &hostname, int32_t port) {
  service_impl_ = service;
  async_server_ = std::make_unique<MasterGrpcServer>(hostname, port, service_impl_.get());
}
Status MSServiceServer::Init() {
  Status status = async_server_->Run();
  if (status != SUCCESS) return status;
  auto grpc_server_run = [this]() { StartAsyncRpcService(); };
  grpc_thread_ = std::thread(grpc_server_run);
  in_running_ = true;
  return SUCCESS;
}
Status MSServiceServer::StartAsyncRpcService() {
  Status status = async_server_->HandleRequest();
  return status;
}

Status MSServiceServer::SendFinish() {
  Status status = async_server_->SendFinish();
  return status;
}

Status MSServiceServer::Stop() {
  if (in_running_) {
    async_server_->Stop();
    grpc_thread_.join();
  }
  in_running_ = false;
  return SUCCESS;
}

}  // namespace serving
}  // namespace mindspore
