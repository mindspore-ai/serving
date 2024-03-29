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

#ifndef MINDSPORE_SERVING_MASTER_GRPC_PROCESS_H
#define MINDSPORE_SERVING_MASTER_GRPC_PROCESS_H

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <memory>
#include <string>
#include "common/serving_common.h"
#include "common/heart_beat.h"
#include "proto/ms_service.pb.h"
#include "proto/ms_service.grpc.pb.h"
#include "proto/ms_master.pb.h"
#include "proto/ms_master.grpc.pb.h"
#include "proto/ms_worker.pb.h"
#include "proto/ms_worker.grpc.pb.h"
#include "master/dispacther.h"

namespace mindspore {
namespace serving {
// Service Implement
class MSServiceImpl {
 public:
  explicit MSServiceImpl(std::shared_ptr<Dispatcher> dispatcher) : dispatcher_(dispatcher) {}
  ~MSServiceImpl() = default;

  void PredictAsync(const proto::PredictRequest *request, proto::PredictReply *reply, PredictOnFinish on_finish);

 private:
  std::shared_ptr<Dispatcher> dispatcher_;
};

// Service Implement
class MSMasterImpl {
 public:
  explicit MSMasterImpl(std::shared_ptr<Dispatcher> dispatcher) : dispatcher_(dispatcher) {}
  ~MSMasterImpl() = default;

  grpc::Status Register(const proto::RegisterRequest *request, proto::RegisterReply *reply);
  grpc::Status Exit(const proto::ExitRequest *request, proto::ExitReply *reply);
  grpc::Status NotifyFailed(const proto::NotifyFailedRequest *request, proto::NotifyFailedReply *reply);
  grpc::Status GetModelInfo(const proto::GetModelInfoRequest *request, proto::GetModelInfoReply *reply);
  void PredictAsync(const proto::PredictRequest *request, proto::PredictReply *reply, const PredictOnFinish &on_finish);

 private:
  std::shared_ptr<Dispatcher> dispatcher_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_MASTER_GRPC_PROCESS_H
