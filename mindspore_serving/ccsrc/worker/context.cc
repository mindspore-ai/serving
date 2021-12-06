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

#include "worker/context.h"

namespace mindspore::serving {

std::shared_ptr<ServableContext> ServableContext::Instance() {
  static std::shared_ptr<ServableContext> instance;
  if (instance == nullptr) {
    instance = std::make_shared<ServableContext>();
  }
  return instance;
}

void ServableContext::SetDeviceType(DeviceType device_type) { device_type_ = device_type; }

DeviceType ServableContext::GetDeviceType() const { return device_type_; }

void ServableContext::SetDeviceId(uint32_t device_id) { device_id_ = device_id; }

uint32_t ServableContext::GetDeviceId() const { return device_id_; }

Status ServableContext::SetDeviceTypeStr(const std::string &device_type) {
  DeviceType type;
  std::string device_type_lowcase = device_type;
  for (auto &c : device_type_lowcase) {
    // cppcheck-suppress useStlAlgorithm
    if (c >= 'A' && c <= 'Z') {
      c = c - 'A' + 'a';
    }
  }
  if (device_type_lowcase == "ascend" || device_type_lowcase == "davinci") {
    type = kDeviceTypeAscend;
  } else if (device_type_lowcase == "gpu") {
    type = kDeviceTypeGpu;
  } else if (device_type_lowcase == "cpu") {
    type = kDeviceTypeCpu;
  } else if (device_type_lowcase == "none") {
    type = kDeviceTypeNotSpecified;
  } else {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Unsupported device type '" << device_type
                                          << "', only support 'Ascend', 'GPU', 'CPU' and None, case ignored";
  }
  SetDeviceType(type);
  return SUCCESS;
}

}  // namespace mindspore::serving
