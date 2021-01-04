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

#include "master/notify_worker/local_notify.h"
#include "worker/worker.h"

namespace mindspore {
namespace serving {

Status LocalNotifyWorker::Dispatch(const proto::PredictRequest &request, proto::PredictReply *reply) {
  return Worker::GetInstance().Run(request, reply);
}

Status LocalNotifyWorker::Exit() {
  Worker::GetInstance().Clear();
  return SUCCESS;
}

}  // namespace serving
}  // namespace mindspore
