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
std::string GetProtorWorkerSpecRepr(const proto::WorkerRegSpec &worker_spec) {
  std::stringstream str;
  auto &servable_spec = worker_spec.servable_spec();
  str << "{name:" << servable_spec.name() << ", version:" << servable_spec.version_number() << ", method:[";
  for (int k = 0; k < servable_spec.methods_size(); k++) {
    str << servable_spec.methods(k).name();
    if (k + 1 < servable_spec.methods_size()) {
      str << ",";
    }
  }
  str << "]}";
  return str.str();
}
}  // namespace

void MSServiceImpl::PredictAsync(const proto::PredictRequest *request, proto::PredictReply *reply,
                                 PredictOnFinish on_finish) {
  dispatcher_->DispatchAsync(*request, reply, on_finish);
}

grpc::Status MSMasterImpl::Register(const proto::RegisterRequest *request, proto::RegisterReply *reply) {
  MSI_EXCEPTION_IF_NULL(request);
  MSI_EXCEPTION_IF_NULL(reply);
  auto worker_sig = [request]() {
    std::stringstream str;
    str << "worker address: " << request->worker_spec().address() << ", servable: ";
    str << GetProtorWorkerSpecRepr(request->worker_spec());
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

grpc::Status MSMasterImpl::Exit(const proto::ExitRequest *request, proto::ExitReply *reply) {
  MSI_EXCEPTION_IF_NULL(request);
  MSI_EXCEPTION_IF_NULL(reply);
  auto worker_sig = [request]() {
    std::stringstream str;
    str << "worker address: " << request->address();
    return str.str();
  };

  MSI_LOG(INFO) << "Worker Exit, " << worker_sig();
  Status status = dispatcher_->NotifyWorkerExit(*request, reply);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "UnRegister servable failed, " << worker_sig();
    return grpc::Status::OK;
  }
  return grpc::Status::OK;
}

grpc::Status MSMasterImpl::NotifyFailed(const proto::NotifyFailedRequest *request, proto::NotifyFailedReply *reply) {
  dispatcher_->NotifyWorkerFailed(request, reply);
  return grpc::Status::OK;
}

grpc::Status MSMasterImpl::GetModelInfo(const proto::GetModelInfoRequest *request, proto::GetModelInfoReply *reply) {
  dispatcher_->GetModelInfo(request, reply);
  return grpc::Status::OK;
}

void MSMasterImpl::PredictAsync(const proto::PredictRequest *request, proto::PredictReply *reply,
                                const PredictOnFinish &on_finish) {
  dispatcher_->DispatchAsync(*request, reply, on_finish);
}
}  // namespace serving
}  // namespace mindspore
