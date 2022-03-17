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

#ifndef MINDSPORE_SERVING_WORKER_AGENT_PROCESS_H
#define MINDSPORE_SERVING_WORKER_AGENT_PROCESS_H

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <memory>
#include <string>
#include "common/serving_common.h"
#include "common/heart_beat.h"
#include "proto/ms_agent.pb.h"
#include "proto/ms_agent.grpc.pb.h"
#include "proto/ms_worker.pb.h"
#include "proto/ms_worker.grpc.pb.h"

namespace mindspore {
namespace serving {
// Service Implement
class MSAgentImpl final : public proto::MSAgent::Service {
 public:
  explicit MSAgentImpl(const std::string server_address) {
    if (!watcher_) {
      watcher_ = std::make_shared<Watcher<proto::MSDistributedWorker, proto::MSDistributedWorker>>(server_address);
    }
  }
  grpc::Status Predict(grpc::ServerContext *context, const proto::DistributedPredictRequest *request,
                       proto::DistributedPredictReply *reply) override;
  grpc::Status Exit(grpc::ServerContext *context, const proto::DistributedExitRequest *request,
                    proto::DistributedExitReply *reply) override;
  grpc::Status Ping(grpc::ServerContext *context, const proto::PingRequest *request, proto::PingReply *reply) override;
  grpc::Status Pong(grpc::ServerContext *context, const proto::PongRequest *request, proto::PongReply *reply) override;

 private:
  std::shared_ptr<Watcher<proto::MSDistributedWorker, proto::MSDistributedWorker>> watcher_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_AGENT_PROCESS_H
