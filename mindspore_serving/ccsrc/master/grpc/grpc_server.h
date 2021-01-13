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

#ifndef MINDSPORE_SERVING_MASTER_GRPC_SERVER_H
#define MINDSPORE_SERVING_MASTER_GRPC_SERVER_H

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <string>
#include <vector>
#include <memory>
#include "common/serving_common.h"
#include "proto/ms_worker.pb.h"
#include "proto/ms_worker.grpc.pb.h"
#include "common/grpc_async_server.h"
#include "master/grpc/grpc_process.h"

namespace mindspore {
namespace serving {

class MSServiceServer;
extern std::unique_ptr<MSServiceServer> grpc_async_server_;

// Service Implement
class MSServiceServer {
 public:
  enum ServerState { kGdsUninit = 0, kGdsInitializing, kGdsRunning, kGdsStopped };
  MSServiceServer(std::shared_ptr<MSServiceImpl> service, const std::string &hostname, int32_t port);
  ~MSServiceServer();

  Status Init();

  Status Stop();

  Status StartAsyncRpcService();

  Status SendFinish();
  bool in_running_ = false;
  std::thread grpc_thread_;
  std::shared_ptr<MSServiceImpl> service_impl_;
  std::unique_ptr<GrpcAsyncServer> async_server_;
};

class UntypedCall {
 public:
  virtual ~UntypedCall() {}

  virtual Status process() = 0;

  virtual bool JudgeFinish() = 0;

  virtual bool JudgeStatus() = 0;

  virtual bool SendFinish() = 0;
};

template <class ServiceImpl, class AsyncService, class RequestMessage, class ResponseMessage>
class CallData : public UntypedCall {
 public:
  enum class STATE : int8_t { CREATE = 1, PROCESS = 2, FINISH = 3 };
  using EnqueueFunction = void (AsyncService::*)(grpc::ServerContext *, RequestMessage *,
                                                 grpc::ServerAsyncResponseWriter<ResponseMessage> *,
                                                 grpc::CompletionQueue *, grpc::ServerCompletionQueue *, void *);
  using HandleRequestFunction = grpc::Status (ServiceImpl::*)(grpc::ServerContext *, const RequestMessage *,
                                                              ResponseMessage *);
  CallData(ServiceImpl *service_impl, AsyncService *async_service, grpc::ServerCompletionQueue *cq,
           EnqueueFunction enqueue_function, HandleRequestFunction handle_request_function)
      : status_(STATE::CREATE),
        service_impl_(service_impl),
        async_service_(async_service),
        cq_(cq),
        enqueue_function_(enqueue_function),
        handle_request_function_(handle_request_function),
        responder_(&ctx_) {}

  ~CallData() = default;

  static Status EnqueueRequest(ServiceImpl *service_impl, AsyncService *async_service, grpc::ServerCompletionQueue *cq,
                               EnqueueFunction enqueue_function, HandleRequestFunction handle_request_function) {
    auto call = new CallData<ServiceImpl, AsyncService, RequestMessage, ResponseMessage>(
      service_impl, async_service, cq, enqueue_function, handle_request_function);
    Status status = call->process();
    if (status != SUCCESS) return status;
    return SUCCESS;
  }

  Status process() override {
    if (status_ == STATE::CREATE) {
      status_ = STATE::PROCESS;
      (async_service_->*enqueue_function_)(&ctx_, &request_, &responder_, cq_, cq_, this);
    } else if (status_ == STATE::PROCESS) {
      EnqueueRequest(service_impl_, async_service_, cq_, enqueue_function_, handle_request_function_);
      status_ = STATE::FINISH;
      response_.set_status(false);
      grpc::Status s = (service_impl_->*handle_request_function_)(&ctx_, &request_, &response_);
      if (!s.ok() || response_.status() == true) {
        responder_.Finish(response_, s, this);
        response_.set_status(true);
      }
    } else {
      MSI_LOG(INFO) << "The CallData status is finish and the pointer needs to be released.";
    }
    return SUCCESS;
  }

  bool JudgeFinish() override {
    if (status_ == STATE::FINISH) {
      return true;
    } else {
      return false;
    }
  }
  bool JudgeStatus() override {
    if (status_ == STATE::FINISH && response_.status() == false) {
      return true;
    } else {
      return false;
    }
  }

  bool SendFinish() override {
    if (status_ == STATE::FINISH && response_.status() == true) {
      responder_.Finish(response_, grpc::Status::OK, this);
      return true;
    } else {
      return false;
    }
  }

 private:
  STATE status_;
  ServiceImpl *service_impl_;
  AsyncService *async_service_;
  grpc::ServerCompletionQueue *cq_;
  EnqueueFunction enqueue_function_;
  HandleRequestFunction handle_request_function_;
  grpc::ServerContext ctx_;
  grpc::ServerAsyncResponseWriter<ResponseMessage> responder_;
  RequestMessage request_;
  ResponseMessage response_;
};

#define ENQUEUE_REQUEST(service_impl, async_service, cq, method, request_msg, response_msg)                        \
  do {                                                                                                             \
    Status s = CallData<MSServiceImpl, proto::MSService::AsyncService, request_msg, response_msg>::EnqueueRequest( \
      service_impl, async_service, cq, &proto::MSService::AsyncService::Request##method, &MSServiceImpl::method);  \
    if (s != SUCCESS) return s;                                                                                    \
  } while (0)

class MasterGrpcServer : public GrpcAsyncServer {
 public:
  MasterGrpcServer(const std::string &host, int32_t port, MSServiceImpl *service_impl)
      : GrpcAsyncServer(host, port), service_impl_(service_impl) {}

  ~MasterGrpcServer() = default;

  Status RegisterService(grpc::ServerBuilder *builder) {
    builder->RegisterService(&svc_);
    return SUCCESS;
  }

  Status EnqueueRequest() {
    ENQUEUE_REQUEST(service_impl_, &svc_, cq_.get(), Predict, proto::PredictRequest, proto::PredictReply);
    return SUCCESS;
  }

  Status ProcessRequest(void *tag) {
    std::unique_lock<std::shared_mutex> lock(lock_);
    auto rq = static_cast<UntypedCall *>(tag);
    if (rq->JudgeFinish()) {
      DelRequest();
    } else {
      Status status = rq->process();
      if (status != SUCCESS) {
        return status;
      }
      if (rq->JudgeStatus()) {
        req_.push_back(rq);
      }
    }
    return SUCCESS;
  }

  Status DelRequest() {
    for (auto rq = del_.begin(); rq != del_.end();) {
      delete (*rq);
      rq = del_.erase(rq);
    }
    return SUCCESS;
  }

  Status SendFinish() {
    std::unique_lock<std::shared_mutex> lock(lock_);
    for (auto rq = req_.begin(); rq != req_.end();) {
      if ((*rq)->SendFinish()) {
        del_.push_back(*rq);
        rq = req_.erase(rq);
      } else {
        ++rq;
      }
    }
    return SUCCESS;
  }

 private:
  MSServiceImpl *service_impl_;
  proto::MSService::AsyncService svc_;
  std::vector<UntypedCall *> req_;
  std::vector<UntypedCall *> del_;
  std::shared_mutex lock_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_MASTER_GRPC_SERVER_H
