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
  explicit MSWorkerImpl(const std::string server_address) {
    if (!watcher_) {
      watcher_ = std::make_shared<Watcher<proto::MSAgent, proto::MSMaster>>(server_address);
    }
  }

  void PredictAsync(grpc::ServerContext *context, const proto::PredictRequest *request, proto::PredictReply *reply,
                    PredictOnFinish on_finish);
  grpc::Status Exit(grpc::ServerContext *context, const proto::ExitRequest *request, proto::ExitReply *reply);
  grpc::Status Ping(grpc::ServerContext *context, const proto::PingRequest *request, proto::PingReply *reply);
  grpc::Status Pong(grpc::ServerContext *context, const proto::PongRequest *request, proto::PongReply *reply);

  std::shared_ptr<Watcher<proto::MSAgent, proto::MSMaster>> watcher_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_WORKER_PROCESS_H
