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

#include "common/grpc_server.h"

namespace mindspore::serving {
Status GrpcServer::Start(const std::shared_ptr<grpc::Service> &service, const std::string &server_address,
                         int max_msg_mb_size, const std::string &server_tag) {
  service_ = service;
  if (in_running_) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Serving Error: " << server_tag << " server is already running";
  }
  // Set the port is not reuseable
  auto option = grpc::MakeChannelArgumentOption(GRPC_ARG_ALLOW_REUSEPORT, 0);
  grpc::ServerBuilder serverBuilder;
  (void)serverBuilder.SetOption(std::move(option));
  if (max_msg_mb_size > 0) {
    constexpr int mbytes_to_bytes = static_cast<int>(1u << 20);
    (void)serverBuilder.SetMaxSendMessageSize(max_msg_mb_size * mbytes_to_bytes);
    (void)serverBuilder.SetMaxReceiveMessageSize(max_msg_mb_size * mbytes_to_bytes);
  }
  (void)serverBuilder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  (void)serverBuilder.RegisterService(service.get());
  server_ = serverBuilder.BuildAndStart();
  if (server_ == nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Serving Error: " << server_tag
                                          << " server start failed, create server failed, address " << server_address;
  }

  auto grpc_server_run = [this, server_address, server_tag]() {
    MSI_LOG(INFO) << server_tag << " server start success, listening on " << server_address;
    server_->Wait();
  };

  grpc_thread_ = std::thread(grpc_server_run);
  in_running_ = true;
  return SUCCESS;
}

void GrpcServer::Stop() {
  if (in_running_) {
    server_->Shutdown();
    grpc_thread_.join();
    server_ = nullptr;
  }
  in_running_ = false;
}

std::shared_ptr<grpc::Channel> GrpcServer::CreateChannel(const std::string &target_str) {
  grpc::ChannelArguments channel_args;
  constexpr int mbytes_to_bytes = static_cast<int>(1u << 20);
  channel_args.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, gRpcMaxMBMsgSize * mbytes_to_bytes);
  std::shared_ptr<grpc::Channel> channel =
    grpc::CreateCustomChannel(target_str, grpc::InsecureChannelCredentials(), channel_args);
  return channel;
}
}  // namespace mindspore::serving
