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
#include "stub/graph_impl_stub.h"

namespace mindspore {

GraphImplStubAdd::GraphImplStubAdd() { Init({2, 2}); }
GraphImplStubAdd::GraphImplStubAdd(const std::vector<int64_t> &add_shape) { Init(add_shape); }
GraphImplStubAdd::~GraphImplStubAdd() {}

void GraphImplStubAdd::Init(const std::vector<int64_t> &add_shape) {
  auto element_cnt = [add_shape]() -> size_t {
    size_t element_num = 1;
    for (auto dim : add_shape) {
      if (dim <= 0) {
        return 0;
      }
      element_num *= dim;
    }
    return element_num;
  };
  auto ele_size = element_cnt() * sizeof(float);
  MSTensor tensor_x1 = MSTensor("x1", mindspore::DataType::kNumberTypeFloat32, add_shape, nullptr, ele_size);
  MSTensor tensor_x2 = MSTensor("x2", mindspore::DataType::kNumberTypeFloat32, add_shape, nullptr, ele_size);

  MSTensor tensor_y = MSTensor("y", mindspore::DataType::kNumberTypeFloat32, add_shape, nullptr, ele_size);

  inputs_.push_back(tensor_x1);
  inputs_.push_back(tensor_x2);
  outputs_.push_back(tensor_y);
}

Status GraphImplStubAdd::Run(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  if (inputs.size() != inputs_.size()) {
    return mindspore::kCoreFailed;
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs[i].DataSize() != inputs_[i].DataSize()) {
      return mindspore::kCoreFailed;
    }
    if (inputs_[i].DataSize() != 0 && inputs[i].Data() == nullptr) {
      return mindspore::kCoreFailed;
    }
  }
  auto x1 = reinterpret_cast<const float *>(inputs[0].Data().get());
  auto x2 = reinterpret_cast<const float *>(inputs[1].Data().get());
  MSTensor output = outputs_[0].Clone();
  auto y = reinterpret_cast<float *>(output.MutableData());
  for (size_t i = 0; i < outputs_[0].DataSize() / sizeof(float); i++) {
    y[i] = x1[i] + x2[i];
  }
  outputs->push_back(output);
  return mindspore::kSuccess;
}

Status GraphImplStubAdd::Load() { return kSuccess; }

std::vector<MSTensor> GraphImplStubAdd::GetInputs() { return inputs_; }

std::vector<MSTensor> GraphImplStubAdd::GetOutputs() { return outputs_; }

}  // namespace mindspore
