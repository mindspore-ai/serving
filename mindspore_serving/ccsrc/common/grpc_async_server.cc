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

#include "common/grpc_async_server.h"
#include <limits>

namespace mindspore::serving {
GrpcAsyncServer::GrpcAsyncServer(const std::string &host, uint32_t port) : host_(host), port_(port) {}

GrpcAsyncServer::~GrpcAsyncServer() { Stop(); }

Status GrpcAsyncServer::Run(const std::string &server_tag, int max_msg_mb_size) {
  std::string server_address = host_ + ":" + std::to_string(port_);
  grpc::ServerBuilder builder;
  if (max_msg_mb_size > 0) {
    builder.SetMaxSendMessageSize(static_cast<int>(max_msg_mb_size * (1u << 20)));
    builder.SetMaxReceiveMessageSize(static_cast<int>(max_msg_mb_size * (1u << 20)));
  }
  builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
  int port_tcpip = 0;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(), &port_tcpip);
  Status status = RegisterService(&builder);
  if (status != SUCCESS) return status;
  cq_ = builder.AddCompletionQueue();
  server_ = builder.BuildAndStart();
  if (server_) {
    MSI_LOG(INFO) << "Serving gRPC server start success, listening on " << server_address;
  } else {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Serving Error: " << server_tag
                                          << " server start failed, create server failed, address " << server_address;
  }
  return SUCCESS;
}

Status GrpcAsyncServer::HandleRequest() {
  bool success = false;
  void *tag;
  // We loop through the grpc queue. Each connection if successful
  // will come back with our own tag which is an instance of CallData
  // and we simply call its functor. But first we need to create these instances
  // and inject them into the grpc queue.
  Status status = EnqueueRequest();
  if (status != SUCCESS) return status;
  while (cq_->Next(&tag, &success)) {
    if (success) {
      status = ProcessRequest(tag);
      if (status != SUCCESS) return status;
    } else {
      MSI_LOG(DEBUG) << "cq_->Next failed.";
    }
  }
  return SUCCESS;
}

void GrpcAsyncServer::Stop() {
  if (server_) {
    server_->Shutdown();
  }
  // Always shutdown the completion queue after the server.
  if (cq_) {
    cq_->Shutdown();
  }
}

}  // namespace mindspore::serving
