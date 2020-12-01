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

#include "grpc_server_async.h"

namespace mindspore::serving {

Status GrpcServerAsync::Start(std::shared_ptr<grpc::Service> service, const std::string &ip, uint32_t grpc_port,
                              int max_msg_mb_size, const std::string &server_tag) {
  Status status;
  if (!is_stoped_) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Grpc server is running";
  }

  std::string server_address = ip + ":" + std::to_string(grpc_port);

  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  // Set the port is not reuseable
  auto option = grpc::MakeChannelArgumentOption(GRPC_ARG_ALLOW_REUSEPORT, 0);
  grpc::ServerBuilder serverBuilder;
  serverBuilder.SetOption(std::move(option));
  if (max_msg_mb_size > 0) {
    serverBuilder.SetMaxReceiveMessageSize(static_cast<int>(max_msg_mb_size * (1u << 20)));
  }
  serverBuilder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  serverBuilder.RegisterService(service.get());

  cq_ = serverBuilder.AddCompletionQueue();
  if (cq_ == nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Serving Error: create grpc server failed, gRPC address " << server_address;
  }

  server_ = serverBuilder.BuildAndStart();
  if (server_ == nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Serving Error: create grpc server failed, gRPC address " << server_address;
  }

  is_stoped_ = false;
  request_map_thread_ = std::thread(RequestThreadFunc, this);
  processing_que_thread_ = std::thread(HandleThreadFunc, this);
  finish_que_thread_ = std::thread(FinishThreadFunc, this);
  return SUCCESS;
}

void GrpcServerAsync::Stop() {
  if (!is_stoped_) {
    is_stoped_ = true;
    server_->Shutdown();
    cq_->Shutdown();
    request_map_thread_.join();
    processing_que_thread_.join();
    finish_que_thread_.join();
  }
}

void GrpcServerAsync::RequestThreadFunc(GrpcServerAsync *grpc_server_async) {
  MSI_EXCEPTION_IF_NULL(grpc_server_async);
  grpc_server_async->RequestThreadFuncInner();
}

void GrpcServerAsync::RequestThreadFuncInner() {
  auto update_request = [this]() {
    auto context_new = std::make_shared<GrpcRequestContext>();
    context_new->status = GrpcRequestContext::PROCESS;
    async_service_.RequestPredict(&context_new->context, &context_new->request, &context_new->responder, cq_.get(),
                                  cq_.get(), context_new.get());

    std::unique_lock<std::mutex> lock{request_lock_};
    request_map_[context_new.get()] = context_new;
  };
  update_request();
  while (true) {
    void *tag;  // uniquely identifies a request.
    bool ok;
    if (!cq_->Next(&tag, &ok) || !ok) {
      MSI_LOG_INFO << "Get Next failed, now exit";
      break;
    }
    auto context = reinterpret_cast<GrpcRequestContext *>(tag);
    if (context->status == GrpcRequestContext::PROCESS) {
      update_request();
      context->status = GrpcRequestContext::FINISH;

    } else {
      std::unique_lock<std::mutex> lock{request_lock_};
      auto it = request_map_.find(tag);
      if (it == request_map_.end()) {
        MSI_LOG_EXCEPTION << "Cannot find request on finish";
      }
      request_map_.erase(it);  // release request
    }
  }
}

void GrpcServerAsync::FinishThreadFunc(GrpcServerAsync *grpc_server_async) {
  MSI_EXCEPTION_IF_NULL(grpc_server_async);
  grpc_server_async->FinishThreadFuncInner();
}

void GrpcServerAsync::FinishThreadFuncInner() {
  while (!is_stoped_) {
    std::shared_ptr<GrpcRequestContext> context = nullptr;
    {
      std::unique_lock<std::mutex> lock{finish_que_lock_};
      if (finish_que_.empty()) {
        finish_que_cond_var_.wait(lock, [this] { return is_stoped_ || finish_que_.empty(); });
        if (is_stoped_) {
          return;
        }
        context = finish_que_.front();
        finish_que_.pop();
      }
    }
    MSI_EXCEPTION_IF_NULL(context);
    context->responder.Finish(context->reply, grpc::Status::OK, context.get());
  }
}

}  // namespace mindspore::serving
