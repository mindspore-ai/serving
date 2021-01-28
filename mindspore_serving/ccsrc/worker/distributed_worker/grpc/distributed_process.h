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

#ifndef MINDSPORE_SERVING_DISTRIBUTED_WORKER_WORKER_PROCESS_H
#define MINDSPORE_SERVING_DISTRIBUTED_WORKER_WORKER_PROCESS_H

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <memory>
#include "common/serving_common.h"
#include "proto/ms_service.pb.h"
#include "proto/ms_service.grpc.pb.h"
#include "proto/ms_distributed.pb.h"
#include "proto/ms_distributed.grpc.pb.h"
#include "worker/distributed_worker/distributed_servable.h"
#include "worker/grpc/worker_process.h"

namespace mindspore {
namespace serving {

// Service Implement
class MSDistributedImpl final : public MSWorkerImpl {
 public:
  explicit MSDistributedImpl(std::shared_ptr<DistributedServable> servable) : servable_(servable) {}
  ~MSDistributedImpl() = default;
  grpc::Status AgentRegister(grpc::ServerContext *context, const proto::AgentRegisterRequest *request,
                             proto::AgentRegisterReply *reply) override;
  grpc::Status AgentExit(grpc::ServerContext *context, const proto::AgentExitRequest *request,
                         proto::AgentExitReply *reply) override;

 private:
  std::shared_ptr<DistributedServable> servable_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_DISTRIBUTED_WORKER_WORKER_PROCESS_H
