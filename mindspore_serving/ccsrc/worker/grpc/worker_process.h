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

#ifndef MINDSPORE_SERVING_WORKER_WORKER_PROCESS_H
#define MINDSPORE_SERVING_WORKER_WORKER_PROCESS_H

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <memory>
#include <string>
#include "common/serving_common.h"
#include "common/heart_beat.h"
#include "common/grpc_client.h"
#include "proto/ms_worker.pb.h"
#include "proto/ms_worker.grpc.pb.h"
#include "proto/ms_master.pb.h"
#include "proto/ms_master.grpc.pb.h"
#include "proto/ms_agent.pb.h"
#include "proto/ms_agent.grpc.pb.h"
namespace mindspore {
namespace serving {
// Service Implement
class MSWorkerImpl {
 public:
  MSWorkerImpl() = default;
  ~MSWorkerImpl() = default;
  void Exit(const proto::ExitRequest *request, proto::ExitReply *reply);
  void PredictAsync(const proto::PredictRequest *request, proto::PredictReply *reply, const PredictOnFinish &on_finish);
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_WORKER_PROCESS_H
