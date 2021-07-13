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

#include "worker/stage_function.h"
#include "mindspore_serving/ccsrc/common/tensor.h"

namespace mindspore::serving {

class StubCastInt32toFp32Preprocess : public CppStageFunctionBase {
 public:
  Status Call(const std::string &postprocess_name, const InstanceData &input, InstanceData *output) override {
    MSI_EXCEPTION_IF_NULL(output);
    auto x1 = input[0];
    auto x2 = input[1];
    if (x1->data_type() != kMSI_Int32 || x2->data_type() != kMSI_Int32) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Call failed: Input data type invalid " << x1->data_type() << ", " << x2->data_type();
    }

    auto y1 = std::make_shared<Tensor>();
    y1->set_data_type(serving::kMSI_Float32);
    y1->resize_data(x1->data_size());
    y1->set_shape(x1->shape());
    output->push_back(y1);

    auto y2 = std::make_shared<Tensor>();
    y2->set_data_type(serving::kMSI_Float32);
    y2->resize_data(x2->data_size());
    y2->set_shape(x2->shape());
    output->push_back(y2);

    auto x1_data = reinterpret_cast<const int32_t *>(x1->data());
    auto y1_data = reinterpret_cast<float *>(y1->mutable_data());
    for (size_t i = 0; i < y1->data_size() / 4; i++) {
      y1_data[i] = static_cast<float>(x1_data[i]);
    }

    auto x2_data = reinterpret_cast<const int32_t *>(x2->data());
    auto y2_data = reinterpret_cast<float *>(y2->mutable_data());
    for (size_t i = 0; i < y2->data_size() / 4; i++) {
      y2_data[i] = static_cast<float>(x2_data[i]);
    }
    return SUCCESS;
  }

  size_t GetInputsCount(const std::string &postprocess_name) const override { return 2; }

  size_t GetOutputsCount(const std::string &postprocess_name) const override { return 2; }
};

REGISTER_STAGE_FUNCTION(StubCastInt32toFp32Preprocess, "stub_preprocess_cast_int32_to_fp32_cpp")

}  // namespace mindspore::serving
