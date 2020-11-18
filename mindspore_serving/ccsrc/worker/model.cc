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

#include "worker/model.h"
#include <algorithm>
#include "mindspore_serving/ccsrc/common/tensor.h"

namespace mindspore::serving {

Status AscendModelServable::Predict(const std::vector<TensorBasePtr> &input, std::vector<TensorBasePtr> &output) {
  return session_->ExecuteModel(model_id_, input, output);
}

std::vector<TensorInfo> AscendModelServable::GetInputInfos() const { return session_->GetInputInfos(model_id_); }

std::vector<TensorInfo> AscendModelServable::GetOutputInfos() const { return session_->GetOutputInfos(model_id_); }

uint64_t AscendModelServable::GetBatchSize() const { return session_->GetBatchSize(model_id_); }

TensorBasePtr AscendModelServable::MakeInferenceTensor(DataType data_type, const std::vector<int64_t> &shape) const {
  return session_->MakeInferenceTensor(data_type, shape);
}

}  // namespace mindspore::serving
