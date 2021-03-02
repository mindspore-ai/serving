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

#ifndef MINDSPORE_SERVING_WORKER_NOTIFY_WORKER_H
#define MINDSPORE_SERVING_WORKER_NOTIFY_WORKER_H
#include <vector>
#include <string>
#include <memory>
#include "common/serving_common.h"
#include "worker/distributed_worker/common.h"
#include "proto/ms_distributed.pb.h"
#include "proto/ms_distributed.grpc.pb.h"
#include "proto/ms_worker.pb.h"
#include "proto/ms_worker.grpc.pb.h"
namespace mindspore {
namespace serving {

class MS_API GrpcNotifyDistributeWorker {
 public:
  GrpcNotifyDistributeWorker(const std::string &worker_ip, uint32_t worker_port, const std::string &agent_ip,
                             uint32_t agent_port);
  ~GrpcNotifyDistributeWorker();
  Status Register(const std::vector<WorkerAgentSpec> &agent_specs);
  Status Unregister();
  // from start up, not agent
  static Status NotifyFailed(const std::string &worker_ip, uint32_t worker_port);
  static Status GetAgentsConfigsFromWorker(const std::string &worker_ip, uint32_t worker_port,
                                           DistributedServableConfig *config);
  static void StartupNotifyExit(const std::string &worker_ip, uint32_t worker_port, const std::string &agent_ip);

 private:
  static Status ParseAgentConfigAcquireReply(const proto::AgentConfigAcquireReply &reply,
                                             DistributedServableConfig *config);
  std::string distributed_worker_ip_;
  uint32_t distributed_worker_port_;
  std::string host_ip_;
  uint32_t host_port_;
  std::string agent_address_;
  std::string distributed_worker_address_;
  std::unique_ptr<proto::MSWorker::Stub> stub_;
  std::atomic<bool> is_stoped_{false};
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_NOTIFY_WORKER_H
