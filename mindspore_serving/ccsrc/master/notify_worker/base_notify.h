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

#ifndef MINDSPORE_SERVING_MASTER_BASE_NOTIFY_H
#define MINDSPORE_SERVING_MASTER_BASE_NOTIFY_H
#include <vector>
#include <functional>
#include <future>
#include "common/serving_common.h"
#include "common/servable.h"
#include "proto/ms_service.pb.h"

namespace mindspore {
namespace serving {

using DispatchCallback = std::function<void(Status status)>;

class MS_API BaseNotifyWorker {
 public:
  BaseNotifyWorker() = default;
  virtual ~BaseNotifyWorker() = default;
  virtual Status Exit() = 0;
  virtual Status DispatchAsync(const proto::PredictRequest &request, proto::PredictReply *reply,
                               DispatchCallback callback) = 0;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_MASTER_BASE_NOTIFY_H
