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

#ifndef MINDSPORE_SERVING_WROERK_MODEL_WRAP_H
#define MINDSPORE_SERVING_WROERK_MODEL_WRAP_H

#include <unordered_map>
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <mutex>
#include "common/serving_common.h"
#include "worker/inference/inference.h"
#include "include/api/model.h"
#include "include/api/types.h"
#include "include/api/data_type.h"
#include "include/api/serialization.h"
#include "include/api/context.h"

namespace mindspore {
namespace serving {

struct ApiModelInfo {
  std::vector<std::string> input_names;
  std::vector<serving::TensorInfo> input_tensor_infos;
  std::vector<std::string> output_names;
  std::vector<serving::TensorInfo> output_tensor_infos;
  std::shared_ptr<mindspore::Model> model = nullptr;
};

struct ApiCommonModelInfo {
  uint32_t batch_size = 0;
  serving::DeviceType device_type;
  uint32_t device_id = 0;
  bool with_batch_dim = false;
  std::vector<int> without_batch_dim_inputs;
};

class MindSporeModelWrap : public InferenceBase {
 public:
  MindSporeModelWrap() = default;

  ~MindSporeModelWrap() = default;

  Status LoadModelFromFile(serving::DeviceType device_type, uint32_t device_id,
                           const std::vector<std::string> &file_names, ModelType model_type, bool with_batch_dim,
                           const std::vector<int> &without_batch_dim_inputs, const ModelContext &model_context,
                           const std::string &dec_key, const std::string &dec_mode, const std::string &config_file,
                           bool enable_lite) override;

  Status UnloadModel() override;
  Status ExecuteModel(const RequestBase &request, ReplyBase *reply, bool return_result, uint64_t subgraph = 0) override;
  Status ExecuteModel(const std::vector<TensorBasePtr> &request, std::vector<TensorBasePtr> *reply, bool return_result,
                      uint64_t subgraph = 0) override;

  std::vector<serving::TensorInfo> GetInputInfos(uint64_t subgraph = 0) const override;

  std::vector<serving::TensorInfo> GetOutputInfos(uint64_t subgraph = 0) const override;

  ssize_t GetBatchSize(uint64_t subgraph = 0) const override;

  bool CheckModelSupport(DeviceType device_type, ModelType model_type) const override;

  uint64_t GetSubGraphNum() const override;
  bool SupportReuseDevice() const override;
  bool SupportMultiThreads() const;

 private:
  ApiCommonModelInfo common_model_info_;
  std::vector<ApiModelInfo> models_;
  static std::mutex infer_mutex_;

  using FuncMakeInBuffer = std::function<mindspore::MSTensor *(size_t index, const std::string &name)>;
  using FuncMakeOutTensor =
    std::function<void(const mindspore::MSTensor, DataType data_type, const std::vector<int64_t> &shape)>;
  Status ExecuteModelCommon(size_t request_size, const FuncMakeInBuffer &in_func, const FuncMakeOutTensor &out_func,
                            bool return_result, uint64_t subgraph = 0);
  Status GetModelInfos(ApiModelInfo *model_info);
  Status SetApiModelInfo(serving::DeviceType device_type, uint32_t device_id,
                         const std::vector<std::string> &file_names, ModelType model_type, bool with_batch_dim,
                         const std::vector<int> &without_batch_dim_inputs, const ModelContext &model_context,
                         const std::vector<std::shared_ptr<mindspore::Model>> &models);

  Status LoadModelFromFileInner(serving::DeviceType device_type, uint32_t device_id,
                                const std::vector<std::string> &file_names, ModelType model_type, bool with_batch_dim,
                                const std::vector<int> &without_batch_dim_inputs, const ModelContext &model_context,
                                const std::string &dec_key, const std::string &dec_mode, const std::string &config_file,
                                bool enable_lite = true);
  std::shared_ptr<Context> TransformModelContext(serving::DeviceType device_type, uint32_t device_id,
                                                 const ModelContext &model_context, bool enable_lite);

  std::shared_ptr<DeviceInfoContext> TransformAscendModelContext(uint32_t device_id, const DeviceInfo &device_info);
  std::shared_ptr<DeviceInfoContext> TransformNvidiaGPUModelContext(uint32_t device_id, const DeviceInfo &device_info);
  std::shared_ptr<DeviceInfoContext> TransformCPUModelContext(const DeviceInfo &device_info);
  DeviceInfo GetDeviceInfo(const std::vector<DeviceInfo> &device_list, serving::DeviceType device_type);

  Status CalculateBatchSize(ApiModelInfo *api_model_info);
  static mindspore::ModelType GetMsModelType(serving::ModelType model_type);
  static mindspore::DeviceType GetMsDeviceType(serving::DeviceType device_type);
  static std::string DeviceTypeToString(serving::DeviceType device_type);
};

class ApiBufferTensorWrap : public TensorBase {
 public:
  ApiBufferTensorWrap();
  explicit ApiBufferTensorWrap(const mindspore::MSTensor &buffer);
  ~ApiBufferTensorWrap() override;

  void set_data_type(DataType type) override { type_ = type; }
  DataType data_type() const override { return type_; }

  void set_shape(const std::vector<int64_t> &shape) override { shape_ = shape; }
  std::vector<int64_t> shape() const override { return shape_; }

  const uint8_t *data() const override { return static_cast<const uint8_t *>(tensor_.Data().get()); }
  size_t data_size() const override { return tensor_.DataSize(); }

  bool resize_data(size_t data_len) override { MSI_LOG_EXCEPTION << "ApiBufferTensorWrap not support resize data"; }
  uint8_t *mutable_data() override { return static_cast<uint8_t *>(tensor_.MutableData()); }

  // For kMSI_String and kMSI_Bytes
  void clear_bytes_data() override { MSI_LOG_EXCEPTION << "Not support for mindspore::Buffer Tensor"; }
  void add_bytes_data(const uint8_t *data, size_t bytes_len) override {
    MSI_LOG_EXCEPTION << "Not support for mindspore::MSTensor Tensor";
  }
  size_t bytes_data_size() const override { MSI_LOG_EXCEPTION << "Not support for mindspore::Buffer Tensor"; }
  void get_bytes_data(size_t index, const uint8_t **data, size_t *bytes_len) const override {
    MSI_LOG_EXCEPTION << "Not support for mindspore::MSTensor Tensor";
  }

 private:
  DataType type_ = kMSI_Unknown;
  std::vector<int64_t> shape_;
  mindspore::MSTensor tensor_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WROERK_MODEL_WRAP_H
