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

#ifndef MINDSPORE_SERVING_WORKER_AGENT_H
#define MINDSPORE_SERVING_WORKER_AGENT_H
#include <vector>
#include <memory>
#include "worker/distributed_worker/agent_executor.h"
#include "proto/ms_agent.pb.h"
#include "proto/ms_agent.grpc.pb.h"
#include "common/grpc_server.h"
#include "worker/distributed_worker/common.h"
#include "worker/distributed_worker/notify_distributed/notify_worker.h"
#include "worker/inference/inference.h"

namespace mindspore {
namespace serving {
class MS_API WorkerAgent {
 public:
  static WorkerAgent &Instance();
  Status Clear();

  Status Run(const proto::DistributedPredictRequest &request, proto::DistributedPredictReply *reply);

  Status StartAgent(const AgentStartUpConfig &config);

  void StopAgent(bool notify_worker = true);

 private:
  AgentStartUpConfig config_;
  std::shared_ptr<InferenceBase> session_ = nullptr;
  GrpcServer grpc_server_;
  bool exit_notify_worker_ = true;
  std::shared_ptr<GrpcNotifyDistributeWorker> notify_worker_;

  Status StartGrpcServer();
  Status RegisterAgent();
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_AGENT_H
