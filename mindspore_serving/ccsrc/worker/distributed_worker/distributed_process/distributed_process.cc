/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "worker/distributed_worker/distributed_process/distributed_process.h"

namespace mindspore {
namespace serving {

grpc::Status MSDistributedImpl::Register(grpc::ServerContext *context, const proto::RegisterRequest *request,
                                         proto::RegisterReply *reply) {
  return grpc::Status::OK;
}

grpc::Status MSDistributedImpl::Predict(grpc::ServerContext *context, const proto::PredictRequest *request,
                                        proto::PredictReply *reply) {
  return grpc::Status::OK;
}

grpc::Status MSDistributedImpl::Exit(grpc::ServerContext *context, const proto::ExitRequest *request,
                                     proto::ExitReply *reply) {
  return grpc::Status::OK;
}
}  // namespace serving
}  // namespace mindspore
