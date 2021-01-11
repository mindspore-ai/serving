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

#ifndef MINDSPORE_SERVING_MASTER_GRPC_CLIENT_H
#define MINDSPORE_SERVING_MASTER_GRPC_CLIENT_H

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <memory>
#include <thread>
#include "common/serving_common.h"
#include "proto/ms_service.pb.h"
#include "proto/ms_service.grpc.pb.h"
#include "proto/ms_master.pb.h"
#include "proto/ms_master.grpc.pb.h"
#include "proto/ms_worker.grpc.pb.h"

namespace mindspore {
namespace serving {
class MSServiceClient;
extern std::unique_ptr<MSServiceClient> client_;

class MSServiceClient {
 public:
  MSServiceClient() = default;
  ~MSServiceClient();
  void AsyncCompleteRpc();
  void Start();
  void Predict(const proto::PredictRequest &request, proto::PredictReply *reply,
               std::shared_ptr<proto::MSWorker::Stub> stub);

 private:
  struct AsyncClientCall {
    grpc::ClientContext context;
    grpc::Status status;
    proto::PredictReply *reply;
    std::shared_ptr<grpc::ClientAsyncResponseReader<proto::PredictReply>> response_reader;
  };

  grpc::CompletionQueue cq_;
  std::thread client_thread_;
  bool in_running_ = false;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_MASTER_GRPC_CLIENT_H
