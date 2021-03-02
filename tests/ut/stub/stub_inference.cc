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
#include <memory>
#include "worker/inference/inference.h"
#include "worker/inference/mindspore_model_wrap.h"

namespace mindspore::serving {

InferenceLoader::InferenceLoader() {}
InferenceLoader::~InferenceLoader() {}

InferenceLoader &InferenceLoader::Instance() {
  static InferenceLoader inference;
  return inference;
}

std::shared_ptr<InferenceBase> InferenceLoader::CreateMindSporeInfer() {
  return std::make_shared<MindSporeModelWrap>();
}

Status InferenceLoader::LoadMindSporeModelWrap() { return SUCCESS; }

DeviceType InferenceLoader::GetSupportDeviceType(DeviceType device_type, ModelType model_type) {
  auto mindspore_infer = CreateMindSporeInfer();
  if (mindspore_infer == nullptr) {
    MSI_LOG_ERROR << "Create MindSpore infer failed";
    return kDeviceTypeNotSpecified;
  }
  if (model_type == kUnknownType) {
    model_type = kMindIR;
  }
  if (device_type == kDeviceTypeNotSpecified) {
    auto ascend_list = {kDeviceTypeAscendCL, kDeviceTypeAscendMS, kDeviceTypeGpu};
    for (auto item : ascend_list) {
      if (mindspore_infer->CheckModelSupport(item, model_type)) {
        return item;
      }
    }
  } else if (device_type == kDeviceTypeAscend) {
    auto ascend_list = {kDeviceTypeAscendCL, kDeviceTypeAscendMS};
    for (auto item : ascend_list) {
      if (mindspore_infer->CheckModelSupport(item, model_type)) {
        return item;
      }
    }
  } else {
    if (mindspore_infer->CheckModelSupport(device_type, model_type)) {
      return device_type;
    }
  }
  return kDeviceTypeNotSpecified;
}
}  // namespace mindspore::serving
