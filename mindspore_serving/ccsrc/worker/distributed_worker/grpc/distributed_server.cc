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

#include "worker/distributed_worker/grpc/distributed_server.h"
#include <string>
#include <memory>
#include <utility>
#include "common/grpc_server.h"

namespace mindspore {
namespace serving {

Status MSDistributedWorkerServer::StartDistributedWorkerGrpcServer(std::shared_ptr<DistributedServable> servable,
                                                                   const std::string &hostname, int32_t port) {
  if (in_running_) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Worker grpc server is already running";
  }
  auto impl = std::make_unique<MSDistributedImpl>(servable);
  async_server_ = std::make_unique<DistributedWorkerGrpcServer>(hostname, port, impl.get());
  service_impl_ = std::move(impl);
  return Init();
}

}  // namespace serving
}  // namespace mindspore
