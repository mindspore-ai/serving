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

#ifndef MINDSPORE_GRPC_SERVER_ASYNC_H
#define MINDSPORE_GRPC_SERVER_ASYNC_H

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <memory>
#include <utility>
#include <string>
#include <future>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <queue>
#include "common/serving_common.h"
#include "proto/ms_service.pb.h"
#include "proto/ms_service.grpc.pb.h"

namespace mindspore::serving {

struct GrpcRequestContext {
  proto::PredictRequest request;
  proto::PredictReply reply;
  grpc::ServerContext context;
  grpc::ServerAsyncResponseWriter<proto::PredictReply> responder;
  grpc::Status grpc_status;

  enum CallStatus { PROCESS, FINISH };
  CallStatus status = PROCESS;

  GrpcRequestContext() : responder(&context) {}
  ~GrpcRequestContext() = default;
};

using GrpcRequestContextPtr = std::shared_ptr<GrpcRequestContext>;

class MS_API GrpcServerAsync {
 public:
  GrpcServerAsync() = default;
  ~GrpcServerAsync() { Stop(); }

  Status Start(std::shared_ptr<grpc::Service> service, const std::string &ip, uint32_t grpc_port, int max_msg_size,
               const std::string &server_tag);
  void Stop();

 private:
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<grpc::ServerCompletionQueue> cq_;
  std::atomic_bool is_stoped_ = true;

  proto::MSService::AsyncService async_service_;

  std::mutex request_lock_;
  std::unordered_map<void *, GrpcRequestContextPtr> request_map_;  // on handle
  std::vector<std::thread> request_thread_pool_;

  std::mutex finish_que_lock_;
  std::condition_variable finish_que_cond_var_;
  std::queue<GrpcRequestContextPtr> finish_que_;  // on finish
  std::thread finish_que_thread_;

  static void RequestThreadFunc(GrpcServerAsync *grpc_server_async);
  void RequestThreadFuncInner();

  static void FinishThreadFunc(GrpcServerAsync *grpc_server_async);
  void FinishThreadFuncInner();
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_GRPC_SERVER_ASYNC_H
