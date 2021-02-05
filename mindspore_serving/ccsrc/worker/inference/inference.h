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

#ifndef MINDSPORE_SERVING_WORKER_INFERENCE_H
#define MINDSPORE_SERVING_WORKER_INFERENCE_H

#include <utility>
#include <map>
#include <unordered_map>
#include <memory>
#include <vector>
#include <string>
#include "common/tensor_base.h"
#include "common/tensor.h"
#include "common/log.h"
#include "common/status.h"
#include "include/api/types.h"
#include "include/api/data_type.h"

namespace mindspore {
namespace serving {

using mindspore::ModelType;
using mindspore::ModelType::kMindIR;
using mindspore::ModelType::kOM;

struct TensorInfo {
  size_t size = 0;  // -1: unspecified
  DataType data_type = kMSI_Unknown;
  std::vector<int64_t> shape;
};

struct TensorInfoWithBatch {
  TensorInfo tensor_info;
  size_t size_one_batch = 0;
  std::vector<int64_t> shape_one_batch;
};

enum DeviceType {
  kDeviceTypeNotSpecified,
  kDeviceTypeAscendMS,
  kDeviceTypeAscendCL,
  kDeviceTypeAscend,
  kDeviceTypeGpu,
  kDeviceTypeCpu,
};

static inline LogStream &operator<<(LogStream &stream, DeviceType device_type) {
  switch (device_type) {
    case kDeviceTypeAscend:
      stream << "Ascend";
      break;
    case kDeviceTypeAscendMS:
      stream << "kDeviceTypeAscend910";
      break;
    case kDeviceTypeAscendCL:
      stream << "kDeviceTypeAscend310";
      break;
    case kDeviceTypeGpu:
      stream << "Gpu";
      break;
    case kDeviceTypeCpu:
      stream << "Cpu";
      break;
    case kDeviceTypeNotSpecified:
      stream << "None(Default)";
      break;
    default:
      stream << "[device type: " << static_cast<int>(device_type) << "]";
      break;
  }
  return stream;
}

static inline LogStream &operator<<(LogStream &stream, mindspore::ModelType model_type) {
  switch (model_type) {
    case mindspore::kMindIR:
      stream << "MindIR";
      break;
    case mindspore::kOM:
      stream << "OM";
      break;
    default:
      stream << "[model type: " << static_cast<int>(model_type) << "]";
      break;
  }
  return stream;
}

}  // namespace serving
}  // namespace mindspore
#endif  // MINDSPORE_SERVING_WORKER_INFERENCE_H
