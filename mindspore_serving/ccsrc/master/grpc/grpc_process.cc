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

#include "master/grpc/grpc_process.h"
#include <string>
#include "master/dispacther.h"

namespace mindspore {
namespace serving {

namespace {
std::string GetProtorWorkerSpecRepr(const proto::WorkerSpec &worker_spec) {
  std::stringstream str;
  str << "{name:" << worker_spec.name() << ", version:" << worker_spec.version_number() << ", method:[";
  for (int k = 0; k < worker_spec.methods_size(); k++) {
    str << worker_spec.methods(k).name();
    if (k + 1 < worker_spec.methods_size()) {
      str << ",";
    }
  }
  str << "]}";
  return str.str();
}
}  // namespace

grpc::Status MSServiceImpl::Predict(grpc::ServerContext *context, const proto::PredictRequest *request,
                                    proto::PredictReply *reply) {
  MSI_EXCEPTION_IF_NULL(request);
  MSI_EXCEPTION_IF_NULL(reply);
  Status status(FAILED);
  try {
    MSI_TIME_STAMP_START(Predict)
    status = dispatcher_->Dispatch(*request, reply);
    MSI_TIME_STAMP_END(Predict)
  } catch (const std::bad_alloc &ex) {
    MSI_LOG(ERROR) << "Serving Error: malloc memory failed";
    std::cout << "Serving Error: malloc memory failed" << std::endl;
  } catch (const std::runtime_error &ex) {
    MSI_LOG(ERROR) << "Serving Error: runtime error occurred: " << ex.what();
    std::cout << "Serving Error: runtime error occurred: " << ex.what() << std::endl;
  } catch (const std::exception &ex) {
    MSI_LOG(ERROR) << "Serving Error: exception occurred: " << ex.what();
    std::cout << "Serving Error: exception occurred: " << ex.what() << std::endl;
  } catch (...) {
    MSI_LOG(ERROR) << "Serving Error: exception occurred";
    std::cout << "Serving Error: exception occurred";
  }
  MSI_LOG(INFO) << "Finish call service Eval";

  if (status != SUCCESS) {
    auto proto_error_msg = reply->add_error_msg();
    proto_error_msg->set_error_code(FAILED);
    std::string error_msg = status.StatusMessage();
    if (error_msg.empty()) {
      proto_error_msg->set_error_msg("Predict failed");
    } else {
      proto_error_msg->set_error_msg(error_msg);
    }
    return grpc::Status::OK;
  }
  return grpc::Status::OK;
}

grpc::Status MSMasterImpl::Register(grpc::ServerContext *context, const proto::RegisterRequest *request,
                                    proto::RegisterReply *reply) {
  MSI_EXCEPTION_IF_NULL(request);
  MSI_EXCEPTION_IF_NULL(reply);
  auto worker_sig = [request]() {
    std::stringstream str;
    str << "worker address: " << request->address() << ", servables: [";
    for (int i = 0; i < request->worker_spec_size(); i++) {
      str << GetProtorWorkerSpecRepr(request->worker_spec(i));
      if (i + 1 < request->worker_spec_size()) {
        str << ", ";
      }
    }
    str << "]";
    return str.str();
  };
  Status status(FAILED);
  status = dispatcher_->RegisterServable(*request, reply);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Register servable failed, " << worker_sig();
    return grpc::Status::OK;
  }
  MSI_LOG(INFO) << "Register success: " << worker_sig();
  return grpc::Status::OK;
}

grpc::Status MSMasterImpl::AddWorker(grpc::ServerContext *context, const proto::AddWorkerRequest *request,
                                     proto::AddWorkerReply *reply) {
  MSI_EXCEPTION_IF_NULL(request);
  MSI_EXCEPTION_IF_NULL(reply);
  auto worker_sig = [request]() {
    std::stringstream str;
    str << "worker address: " << request->address()
        << ", servables: " << GetProtorWorkerSpecRepr(request->worker_spec());
    return str.str();
  };
  Status status(FAILED);
  status = dispatcher_->AddServable(*request, reply);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Add servable failed, " << worker_sig();
    return grpc::Status::OK;
  }
  MSI_LOG(INFO) << "Add success, " << worker_sig();
  return grpc::Status::OK;
}

grpc::Status MSMasterImpl::RemoveWorker(grpc::ServerContext *context, const proto::RemoveWorkerRequest *request,
                                        proto::RemoveWorkerReply *reply) {
  MSI_EXCEPTION_IF_NULL(request);
  MSI_EXCEPTION_IF_NULL(reply);
  auto worker_sig = [request]() {
    std::stringstream str;
    str << "worker address: " << request->address()
        << ", servables: " << GetProtorWorkerSpecRepr(request->worker_spec());
    return str.str();
  };
  Status status(FAILED);
  status = dispatcher_->RemoveServable(*request, reply);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Add servable failed, " << worker_sig();
    return grpc::Status::OK;
  }
  MSI_LOG(INFO) << "Add success, " << worker_sig();
  return grpc::Status::OK;
}

grpc::Status MSMasterImpl::Exit(grpc::ServerContext *context, const proto::ExitRequest *request,
                                proto::ExitReply *reply) {
  MSI_EXCEPTION_IF_NULL(request);
  MSI_EXCEPTION_IF_NULL(reply);
  auto worker_sig = [request]() {
    std::stringstream str;
    str << "worker address: " << request->address();
    return str.str();
  };

  MSI_LOG(INFO) << "Worker Exit, " << worker_sig();
  Status status(FAILED);
  status = dispatcher_->UnregisterServable(*request, reply);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "UnRegister servable failed, " << worker_sig();
    return grpc::Status::OK;
  }
  return grpc::Status::OK;
}

}  // namespace serving
}  // namespace mindspore
