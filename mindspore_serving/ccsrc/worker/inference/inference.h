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
#include <atomic>
#include "common/serving_common.h"

namespace mindspore {
namespace serving {

enum DeviceType {
  kDeviceTypeNotSpecified,
  kDeviceTypeAscendMS,
  kDeviceTypeAscendCL,
  kDeviceTypeAscend,
  kDeviceTypeGpu,
  kDeviceTypeCpu,
};

enum ModelType : uint32_t {
  kMindIR = 0,
  kAIR = 1,
  kOM = 2,
  kONNX = 3,
  // insert new data type here
  kUnknownType = 0xFFFFFFFF
};

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

static inline LogStream &operator<<(LogStream &stream, DeviceType device_type) {
  switch (device_type) {
    case kDeviceTypeAscend:
      stream << "Ascend";
      break;
    case kDeviceTypeAscendMS:
      stream << "Ascend910";
      break;
    case kDeviceTypeAscendCL:
      stream << "Ascend310";
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

static inline LogStream &operator<<(LogStream &stream, ModelType model_type) {
  switch (model_type) {
    case kMindIR:
      stream << "MindIR";
      break;
    case kOM:
      stream << "OM";
      break;
    case kONNX:
      stream << "ONNX";
      break;
    case kAIR:
      stream << "AIR";
      break;
    default:
      stream << "[model type: " << static_cast<int>(model_type) << "]";
      break;
  }
  return stream;
}

class InferenceBase {
 public:
  InferenceBase() = default;
  virtual ~InferenceBase() = default;
  virtual Status LoadModelFromFile(DeviceType device_type, uint32_t device_id,
                                   const std::vector<std::string> &file_name, ModelType model_type, bool with_batch_dim,
                                   const std::vector<int> &without_batch_dim_inputs,
                                   const std::map<std::string, std::string> &other_options, const std::string &dec_key,
                                   const std::string &dec_mode) = 0;

  virtual Status UnloadModel() = 0;

  virtual Status ExecuteModel(const RequestBase &request, ReplyBase *reply, bool return_result,
                              uint64_t subgraph = 0) = 0;
  virtual Status ExecuteModel(const std::vector<TensorBasePtr> &request, std::vector<TensorBasePtr> *reply,
                              bool return_result, uint64_t subgraph = 0) = 0;

  virtual std::vector<TensorInfo> GetInputInfos(uint64_t subgraph = 0) const = 0;

  virtual std::vector<TensorInfo> GetOutputInfos(uint64_t subgraph = 0) const = 0;

  virtual ssize_t GetBatchSize(uint64_t subgraph = 0) const = 0;

  virtual bool CheckModelSupport(DeviceType device_type, ModelType model_type) const = 0;

  virtual uint64_t GetSubGraphNum() const = 0;
};

class MS_API InferenceLoader {
 public:
  InferenceLoader();
  ~InferenceLoader();
  static InferenceLoader &Instance();
  std::shared_ptr<InferenceBase> CreateMindSporeInfer();
  DeviceType GetSupportDeviceType(DeviceType device_type, ModelType model_type);

 private:
  typedef InferenceBase *(*CreateInferHandle)();
  void *ms_lib_handle_ = nullptr;
  void *ms_cxx_lib_handle_ = nullptr;
  void *gomp_handler_ = nullptr;
  CreateInferHandle ms_create_handle_ = nullptr;
  Status LoadMindSporeModelWrap();
};

}  // namespace serving
}  // namespace mindspore
#endif  // MINDSPORE_SERVING_WORKER_INFERENCE_H
