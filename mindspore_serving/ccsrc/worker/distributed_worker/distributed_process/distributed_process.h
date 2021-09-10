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
#include <string>
#include "common/serving_common.h"
#include "common/heart_beat.h"
#include "proto/ms_service.pb.h"
#include "proto/ms_service.grpc.pb.h"
#include "proto/ms_distributed.pb.h"
#include "proto/ms_distributed.grpc.pb.h"
#include "worker/distributed_worker/distributed_model_loader.h"
#include "worker/grpc/worker_process.h"

namespace mindspore {
namespace serving {

// Service Implement
class MSDistributedImpl {
 public:
  explicit MSDistributedImpl(std::shared_ptr<DistributedModelLoader> servable, const std::string server_address)
      : servable_(servable) {
    if (!watcher_) {
      watcher_ = std::make_shared<Watcher<proto::MSAgent, proto::MSAgent>>(server_address);
    }
  }
  ~MSDistributedImpl() = default;
  grpc::Status AgentRegister(grpc::ServerContext *context, const proto::AgentRegisterRequest *request,
                             proto::AgentRegisterReply *reply);
  grpc::Status AgentExit(grpc::ServerContext *context, const proto::AgentExitRequest *request,
                         proto::AgentExitReply *reply);
  grpc::Status AgentFailed(grpc::ServerContext *context, const proto::AgentFailedRequest *request,
                           proto::AgentFailedReply *reply);
  grpc::Status AgentConfigAcquire(grpc::ServerContext *context, const proto::AgentConfigAcquireRequest *request,
                                  proto::AgentConfigAcquireReply *reply);

  grpc::Status Ping(grpc::ServerContext *context, const proto::PingRequest *request, proto::PingReply *reply);
  grpc::Status Pong(grpc::ServerContext *context, const proto::PongRequest *request, proto::PongReply *reply);

 private:
  std::shared_ptr<DistributedModelLoader> servable_;

  std::shared_ptr<Watcher<proto::MSAgent, proto::MSAgent>> watcher_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_DISTRIBUTED_WORKER_WORKER_PROCESS_H
