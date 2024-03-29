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

#ifndef MINDSPORE_SERVING_MASTER_GRPC_NOTIFY_H
#define MINDSPORE_SERVING_MASTER_GRPC_NOTIFY_H
#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include "master/notify_worker/base_notify.h"
#include "proto/ms_worker.pb.h"
#include "proto/ms_worker.grpc.pb.h"

namespace mindspore {
namespace serving {
class MS_API GrpcNotifyWorker : public BaseNotifyWorker {
 public:
  explicit GrpcNotifyWorker(const std::string &worker_address);
  ~GrpcNotifyWorker() override;

  Status DispatchAsync(const proto::PredictRequest &request, proto::PredictReply *reply,
                       const PredictOnFinish &on_finish) override;

 private:
  std::string worker_address_;
  std::shared_ptr<proto::MSWorker::Stub> stub_ = nullptr;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_MASTER_GRPC_NOTIFY_H
