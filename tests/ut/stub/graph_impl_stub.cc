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
  inputs_.clear();
  for (size_t i = 0; i < input_count; i++) {
    MSTensor tensor_x =
      MSTensor("x" + std::to_string(1), mindspore::DataType::kNumberTypeFloat32, add_shape, nullptr, ele_size);
    inputs_.push_back(tensor_x);
  }
  outputs_.clear();
  for (size_t i = 0; i < output_count; i++) {
    MSTensor tensor_y =
      MSTensor("x" + std::to_string(1), mindspore::DataType::kNumberTypeFloat32, add_shape, nullptr, ele_size);
    outputs_.push_back(tensor_y);
  }
}

// y=x1+x2+x3+x4, y=x1-x2-x3-x4
// y2=y1+1
Status GraphImplStubAdd::Run(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  auto file_name = graph_->graph_data_->GetFuncGraph()->file_name_;
  MS_LOG_INFO << "exec model file ------------------- " << file_name;
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
  auto item_count = outputs_[0].DataSize() / sizeof(float);

  auto get_output_tensor = [this](size_t index) -> MSTensor {
    MSTensor *output_ptr = outputs_[index].Clone();
    MSTensor output = *output_ptr;
    mindspore::MSTensor::DestroyTensorPtr(output_ptr);
    return output;
  };
  auto output = get_output_tensor(0);
  auto y = reinterpret_cast<float *>(output.MutableData());
  auto x0 = reinterpret_cast<const float *>(inputs[0].Data().get());
  for (size_t i = 0; i < item_count; i++) {
    y[i] = x0[i];
  }
  for (size_t k = 1; k < input_count; k++) {
    auto xk = reinterpret_cast<const float *>(inputs[k].Data().get());
    for (size_t i = 0; i < item_count; i++) {
      if (sub_) {
        y[i] = y[i] - xk[i];
      } else {
        y[i] = y[i] + xk[i];
      }
    }
  }
  outputs->push_back(output);
  for (size_t k = 1; k < output_count; k++) {
    auto output_k = get_output_tensor(k);
    auto yk = reinterpret_cast<float *>(output_k.MutableData());
    for (size_t i = 0; i < item_count; i++) {
      yk[i] = y[i] + k;
    }
    outputs->push_back(output_k);
  }
  return mindspore::kSuccess;
}

Status GraphImplStubAdd::Load(uint32_t device_id) {
  LoadInner();
  if (input_count == 0 || output_count == 0) {
    MS_LOG_ERROR << "Invalid input count or output count, input count: " << input_count
                 << ", output count: " << output_count;
    return kCoreFailed;
  }
  MS_LOG_INFO << "input count: " << input_count << ", output count: " << output_count;
  Init({2, 2});
  return kSuccess;
}

void GraphImplStubAdd::LoadInner() {
  auto file_name = graph_->graph_data_->GetFuncGraph()->file_name_;
  MS_LOG_INFO << "model file ------------------- " << file_name;
  auto beg = file_name.find("tensor_add");  // tensor_add_2_2.mindir or tensor_sub_2_2.mindir
  if (beg == std::string::npos) {
    beg = file_name.find("tensor_sub");
    if (beg == std::string::npos) {
      return;
    }
    sub_ = true;
  }
  beg += std::string("tensor_add").size();
  auto input_beg = file_name.find("_", beg);
  if (input_beg == std::string::npos) {
    return;
  }
  auto output_beg = file_name.find("_", input_beg + 1);
  if (output_beg == std::string::npos) {
    return;
  }
  auto dot_beg = file_name.find(".mindir", output_beg + 1);
  if (dot_beg == std::string::npos) {
    return;
  }
  input_count = std::stoi(file_name.substr(input_beg + 1, output_beg));
  output_count = std::stoi(file_name.substr(output_beg + 1, dot_beg));
}

std::vector<MSTensor> GraphImplStubAdd::GetInputs() { return inputs_; }

std::vector<MSTensor> GraphImplStubAdd::GetOutputs() { return outputs_; }

}  // namespace mindspore
