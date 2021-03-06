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

#include "worker/grpc/worker_process.h"
#include "worker/worker.h"
#include "common/proto_tensor.h"

namespace mindspore {
namespace serving {
void MSWorkerImpl::Exit(grpc::ServerContext *context, const proto::ExitRequest *request, proto::ExitReply *reply) {
  MSI_LOG(INFO) << "Master Exit";
  Worker::GetInstance().StopServable(false);
}

void MSWorkerImpl::PredictAsync(grpc::ServerContext *context, const proto::PredictRequest *request,
                                proto::PredictReply *reply, PredictOnFinish on_finish) {
  Status status(WORKER_UNAVAILABLE);
  MSI_LOG(INFO) << "Begin call service Eval";
  try {
    status = Worker::GetInstance().RunAsync(*request, reply, on_finish);
  } catch (const std::bad_alloc &ex) {
    MSI_LOG(ERROR) << "Serving Error: malloc memory failed";
  } catch (const std::runtime_error &ex) {
    MSI_LOG(ERROR) << "Serving Error: runtime error occurred: " << ex.what();
  } catch (const std::exception &ex) {
    MSI_LOG(ERROR) << "Serving Error: exception occurred: " << ex.what();
  } catch (...) {
    MSI_LOG(ERROR) << "Serving Error: exception occurred";
  }
  MSI_LOG(INFO) << "Finish call service Eval";

  if (status != SUCCESS) {
    GrpcTensorHelper::CreateReplyFromErrorMsg(status, reply);
    on_finish();
  }
}

}  // namespace serving
}  // namespace mindspore
