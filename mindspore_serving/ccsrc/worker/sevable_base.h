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

#ifndef MINDSPORE_SERVING_WORKER_SERVABLE_BASE_H
#define MINDSPORE_SERVING_WORKER_SERVABLE_BASE_H

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>

#include "common/serving_common.h"
#include "common/instance.h"
#include "common/servable.h"
#include "worker/inference/inference.h"

namespace mindspore::serving {

class ServableBase {
 public:
  ServableBase() = default;
  virtual ~ServableBase() = default;

  virtual Status Predict(const std::vector<TensorBasePtr> &input, std::vector<TensorBasePtr> *output) = 0;

  virtual std::vector<TensorInfo> GetInputInfos() const = 0;
  virtual std::vector<TensorInfo> GetOutputInfos() const = 0;
  virtual uint64_t GetBatchSize() const = 0;
  virtual std::string GetServableName() const = 0;
  virtual uint64_t GetServableVersion() const = 0;
  virtual uint64_t GetConfigVersion() const = 0;
  virtual void Clear() = 0;
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_WORKER_SERVABLE_BASE_H
