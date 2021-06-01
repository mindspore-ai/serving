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

#ifndef MINDSPORE_SERVING_WORKER_GRPC_NOTIFY_H
#define MINDSPORE_SERVING_WORKER_GRPC_NOTIFY_H
#include <vector>
#include <string>
#include <memory>
#include "worker/notfiy_master/base_notify.h"
#include "proto/ms_master.pb.h"
#include "proto/ms_master.grpc.pb.h"

namespace mindspore {
namespace serving {

class MS_API GrpcNotifyMaster : public BaseNotifyMaster {
 public:
  GrpcNotifyMaster(const std::string &master_address, const std::string &worker_address);
  ~GrpcNotifyMaster() override;
  Status Register(const WorkerRegSpec &worker_spec) override;
  Status Unregister() override;
  static Status NotifyFailed(const std::string &master_address, const std::string &error_msg);

 private:
  std::string master_address_;
  std::string worker_address_;

  std::atomic<bool> is_stoped_{false};
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_GRPC_NOTIFY_H
