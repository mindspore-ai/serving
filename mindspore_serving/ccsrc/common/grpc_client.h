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

#ifndef MINDSPORE_SERVING_MASTER_GRPC_CLIENT_H
#define MINDSPORE_SERVING_MASTER_GRPC_CLIENT_H

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <memory>
#include <functional>
#include <thread>
#include <string>
#include <utility>
#include "common/serving_common.h"
#include "proto/ms_service.pb.h"
#include "proto/ms_service.grpc.pb.h"
#include "proto/ms_master.pb.h"
#include "proto/ms_master.grpc.pb.h"
#include "proto/ms_worker.grpc.pb.h"
#include "proto/ms_agent.pb.h"
#include "proto/ms_agent.grpc.pb.h"

namespace mindspore {
namespace serving {

using PredictOnFinish = std::function<void()>;

using AsyncPredictCallback = std::function<void(Status status)>;

template <typename Request, typename Reply, typename MSStub>
class MSServiceClient {
 public:
  MSServiceClient() = default;
  ~MSServiceClient() {
    if (in_running_) {
      cq_.Shutdown();
      if (client_thread_.joinable()) {
        try {
          client_thread_.join();
        } catch (const std::system_error &) {
        } catch (...) {
        }
      }
    }
    in_running_ = false;
  }

  void Start() {
    client_thread_ = std::thread(&MSServiceClient::AsyncCompleteRpc, this);
    in_running_ = true;
  }

  void AsyncCompleteRpc() {
    void *got_tag;
    bool ok = false;

    while (cq_.Next(&got_tag, &ok)) {
      AsyncClientCall *call = static_cast<AsyncClientCall *>(got_tag);
      if (call->status.ok()) {
        call->callback(SUCCESS);
      } else {
        MSI_LOG_ERROR << "RPC failed: " << call->status.error_code() << ", " << call->status.error_message();
        call->callback(Status(WORKER_UNAVAILABLE, call->status.error_message()));
      }
      delete call;
    }
  }

  void PredictAsync(const Request &request, Reply *reply, MSStub *stub, AsyncPredictCallback callback) {
    AsyncClientCall *call = new AsyncClientCall;
    call->reply = reply;
    call->callback = std::move(callback);
    call->response_reader = stub->PrepareAsyncPredict(&call->context, request, &cq_);
    call->response_reader->StartCall();
    call->response_reader->Finish(call->reply, &call->status, call);
  }

 private:
  struct AsyncClientCall {
    grpc::ClientContext context;
    grpc::Status status;
    Reply *reply;
    AsyncPredictCallback callback;
    std::shared_ptr<grpc::ClientAsyncResponseReader<Reply>> response_reader;
  };

  grpc::CompletionQueue cq_;
  std::thread client_thread_;
  bool in_running_ = false;
};

using MSPredictClient = MSServiceClient<proto::PredictRequest, proto::PredictReply, proto::MSWorker::Stub>;
using MSDistributedClient =
  MSServiceClient<proto::DistributedPredictRequest, proto::DistributedPredictReply, proto::MSAgent::Stub>;
extern std::unique_ptr<MSPredictClient> client_;
extern std::unique_ptr<MSDistributedClient> distributed_client_;
}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_MASTER_GRPC_CLIENT_H
