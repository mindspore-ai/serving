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

#ifndef MINDSPORE_SERVING_WORKER_NOTIFY_AGENT_H
#define MINDSPORE_SERVING_WORKER_NOTIFY_AGENT_H
#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include "worker/distributed_worker/notify_agent/base_notify_agent.h"
#include "proto/ms_agent.pb.h"
#include "proto/ms_agent.grpc.pb.h"

namespace mindspore {
namespace serving {
class MS_API GrpcNotifyAgent : public BaseNotifyAgent {
 public:
  explicit GrpcNotifyAgent(const std::string &worker_address);
  ~GrpcNotifyAgent() override;

  Status Exit() override;

  Status DispatchAsync(const proto::DistributedPredictRequest &request, proto::DistributedPredictReply *reply,
                       AsyncPredictCallback callback) override;

 private:
  std::string agent_address_;
  std::shared_ptr<proto::MSAgent::Stub> stub_ = nullptr;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_NOTIFY_AGENT_H
