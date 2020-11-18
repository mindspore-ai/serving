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

#ifndef MINDSPORE_SERVING_GRPC_NOTIFY_H
#define MINDSPORE_SERVING_GRPC_NOTIFY_H
#include "worker/notfiy_master/base_notify.h"
#include "proto/ms_master.pb.h"
#include "proto/ms_master.grpc.pb.h"

namespace mindspore {
namespace serving {

class MS_API GrpcNotfiyMaster : public BaseNotifyMaster {
 public:
  GrpcNotfiyMaster(const std::string &master_ip, uint32_t master_port, const std::string &host_ip, uint32_t host_port);
  ~GrpcNotfiyMaster() override;
  Status Register(const std::vector<WorkerSpec> &worker_specs) override;
  Status Unregister() override;
  Status AddWorker(const WorkerSpec &worker_spec) override;
  Status RemoveWorker(const WorkerSpec &worker_spec) override;

 private:
  std::string master_ip_;
  uint32_t master_port_;
  std::string host_ip_;
  uint32_t host_port_;
  std::string worker_address_;
  std::string master_address_;

  std::unique_ptr<proto::MSMaster::Stub> stub_;
  std::atomic<bool> is_stoped_{false};
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_GRPC_NOTIFY_H
