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

class ArgmaxStageFunc : public CppStageFunctionBase {
 public:
  template <typename DT>
  void ArgmaxImp(const void *input, int64_t *output, size_t data_size, size_t elemsize) {
    auto count = data_size / elemsize;
    auto data = reinterpret_cast<const DT *>(input);
    *output = 0;
    for (size_t i = 1; i < count; i++) {
      if (data[i] > data[*output]) {
        *output = i;
      }
    }
  }

  Status Call(const std::string &func_name, const InstanceData &input, InstanceData *output) override {
    MSI_EXCEPTION_IF_NULL(output);
    auto input_x = input[0];
    auto x_data = input_x->data();
    auto out_tensor = std::make_shared<Tensor>();
    out_tensor->set_data_type(serving::kMSI_Int64);
    out_tensor->resize_data(sizeof(int64_t));
    out_tensor->set_shape({});
    output->push_back(out_tensor);
    auto y_data = reinterpret_cast<int64_t *>(out_tensor->mutable_data());
    switch (input_x->data_type()) {
      case kMSI_Float32:
        ArgmaxImp<float>(x_data, y_data, input_x->data_size(), input_x->itemsize());
        break;
      case kMSI_Float64:
        ArgmaxImp<double>(x_data, y_data, input_x->data_size(), input_x->itemsize());
        break;
      case kMSI_Int8:
        ArgmaxImp<int8_t>(x_data, y_data, input_x->data_size(), input_x->itemsize());
        break;
      case kMSI_Uint8:
        ArgmaxImp<uint8_t>(x_data, y_data, input_x->data_size(), input_x->itemsize());
        break;
      case kMSI_Int16:
        ArgmaxImp<int16_t>(x_data, y_data, input_x->data_size(), input_x->itemsize());
        break;
      case kMSI_Uint16:
        ArgmaxImp<uint16_t>(x_data, y_data, input_x->data_size(), input_x->itemsize());
        break;
      case kMSI_Int32:
        ArgmaxImp<int32_t>(x_data, y_data, input_x->data_size(), input_x->itemsize());
        break;
      case kMSI_Uint32:
        ArgmaxImp<uint32_t>(x_data, y_data, input_x->data_size(), input_x->itemsize());
        break;
      case kMSI_Int64:
        ArgmaxImp<int64_t>(x_data, y_data, input_x->data_size(), input_x->itemsize());
        break;
      case kMSI_Uint64:
        ArgmaxImp<uint64_t>(x_data, y_data, input_x->data_size(), input_x->itemsize());
        break;
      default:
        return INFER_STATUS(FAILED) << "Argmax not support data type " << input_x->data_type();
    }
    return SUCCESS;
  }

  size_t GetInputsCount(const std::string &func_name) const override { return 1; }

  size_t GetOutputsCount(const std::string &func_name) const override { return 1; }
};

REGISTER_STAGE_FUNCTION(ArgmaxStageFunc, "argmax_cpp")

}  // namespace mindspore::serving
