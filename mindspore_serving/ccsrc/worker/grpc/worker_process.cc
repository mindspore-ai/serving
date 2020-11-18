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
#include "master/dispacther.h"
#include "worker/worker.h"

namespace mindspore {
namespace serving {
grpc::Status MSWorkerImpl::Exit(grpc::ServerContext *context, const proto::ExitRequest *request,
                                proto::ExitReply *reply) {
  MSI_LOG(INFO) << "Master Exit";
  Worker::GetInstance().StopServable(false);
  return grpc::Status::OK;
}

grpc::Status MSWorkerImpl::Predict(grpc::ServerContext *context, const proto::PredictRequest *request,
                                   proto::PredictReply *reply) {
  Status status(FAILED);
  MSI_LOG(INFO) << "Begin call service Eval";
  try {
    MSI_TIME_STAMP_START(Predict)
    status = Worker::GetInstance().Run(*request, *reply);
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

  if (status == INVALID_INPUTS) {
    auto proto_error_msg = reply->add_error_msg();
    proto_error_msg->set_error_code(status.StatusCode());
    proto_error_msg->set_error_msg(status.StatusMessage());
    return grpc::Status::OK;
  } else if (status != SUCCESS) {
    auto proto_error_msg = reply->add_error_msg();
    proto_error_msg->set_error_code(FAILED);
    proto_error_msg->set_error_msg("Predict failed");
    return grpc::Status::OK;
  }
  return grpc::Status::OK;
}

}  // namespace serving
}  // namespace mindspore
