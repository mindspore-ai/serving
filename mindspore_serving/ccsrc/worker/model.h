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

#ifndef MINDSPORE_SERVING_WORKER_MODEL_H
#define MINDSPORE_SERVING_WORKER_MODEL_H

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
  virtual TensorBasePtr MakeInferenceTensor(DataType data_type, const std::vector<int64_t> &shape) const {
    return nullptr;
  }
};

class AscendModelServable : public ServableBase {
 public:
  AscendModelServable(const std::shared_ptr<serving::InferSession> &session, uint32_t model_id)
      : session_(session), model_id_(model_id) {}
  ~AscendModelServable() = default;

  Status Predict(const std::vector<TensorBasePtr> &input, std::vector<TensorBasePtr> *output) override;

  std::vector<TensorInfo> GetInputInfos() const override;
  std::vector<TensorInfo> GetOutputInfos() const override;
  uint64_t GetBatchSize() const override;
  TensorBasePtr MakeInferenceTensor(DataType data_type, const std::vector<int64_t> &shape) const override;

 private:
  std::shared_ptr<serving::InferSession> session_{nullptr};
  uint32_t model_id_ = 0;
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_WORKER_MODEL_H
