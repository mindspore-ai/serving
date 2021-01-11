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

#include "master/grpc/grpc_client.h"
#include <string>
#include "master/grpc/grpc_server.h"

namespace mindspore {
namespace serving {
std::unique_ptr<MSServiceClient> client_;

MSServiceClient::~MSServiceClient() {
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
void MSServiceClient::Predict(const proto::PredictRequest &request, proto::PredictReply *reply,
                              std::shared_ptr<proto::MSWorker::Stub> stub) {
  AsyncClientCall *call = new AsyncClientCall;
  call->reply = reply;
  call->response_reader = stub->PrepareAsyncPredict(&call->context, request, &cq_);
  call->response_reader->StartCall();
  call->response_reader->Finish(call->reply, &call->status, call);
  MSI_LOG(INFO) << "Finish send Predict";
}
void MSServiceClient::AsyncCompleteRpc() {
  void *got_tag;
  bool ok = false;

  while (cq_.Next(&got_tag, &ok)) {
    AsyncClientCall *call = static_cast<AsyncClientCall *>(got_tag);
    if (call->status.ok()) {
      call->reply->set_status(true);
      if (grpc_async_server_) {
        grpc_async_server_->SendFinish();
      }
    } else {
      MSI_LOG(INFO) << "RPC failed";
    }
    delete call;
  }
}

void MSServiceClient::Start() {
  client_thread_ = std::thread(&MSServiceClient::AsyncCompleteRpc, this);
  in_running_ = true;
}

}  // namespace serving
}  // namespace mindspore
