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

namespace mindspore::api {

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

  input_shapes_.push_back(add_shape);
  input_shapes_.push_back(add_shape);
  input_data_types_.push_back(api::kMsFloat32);
  input_data_types_.push_back(api::kMsFloat32);
  input_names_.emplace_back("x1");
  input_names_.emplace_back("x2");
  input_mem_sizes_.push_back(ele_size);
  input_mem_sizes_.push_back(ele_size);

  output_shapes_.push_back(add_shape);
  output_data_types_.push_back(api::kMsFloat32);
  output_names_.emplace_back("y");
  output_mem_sizes_.push_back(ele_size);
}

Status GraphImplStubAdd::Run(const std::vector<Buffer> &inputs, std::vector<Buffer> *outputs) {
  if (inputs.size() != input_names_.size()) {
    return FAILED;
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs[i].DataSize() != input_mem_sizes_[i]) {
      return FAILED;
    }
    if (input_mem_sizes_[i] != 0 && inputs[i].Data() == nullptr) {
      return FAILED;
    }
  }
  auto x1 = reinterpret_cast<const float *>(inputs[0].Data());
  auto x2 = reinterpret_cast<const float *>(inputs[1].Data());
  Buffer output;
  output.ResizeData(output_mem_sizes_[0]);
  auto y = reinterpret_cast<float *>(output.MutableData());
  for (size_t i = 0; i < output_mem_sizes_[0] / sizeof(float); i++) {
    y[i] = x1[i] + x2[i];
  }
  outputs->push_back(output);
  return SUCCESS;
}

Status GraphImplStubAdd::Load() { return SUCCESS; }
Status GraphImplStubAdd::GetInputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                                       std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) {
  *names = input_names_;
  *shapes = input_shapes_;
  *data_types = input_data_types_;
  *mem_sizes = input_mem_sizes_;
  return SUCCESS;
}
Status GraphImplStubAdd::GetOutputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                                        std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) {
  *names = output_names_;
  *shapes = output_shapes_;
  *data_types = output_data_types_;
  *mem_sizes = output_mem_sizes_;
  return SUCCESS;
}

}  // namespace mindspore::api
