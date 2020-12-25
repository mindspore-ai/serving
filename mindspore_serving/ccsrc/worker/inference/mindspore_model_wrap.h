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
#include "common/serving_common.h"
#include "worker/inference/inference.h"
#include "api/model.h"

namespace mindspore {
namespace serving {

struct ApiModelInfo {
  std::vector<std::string> input_names;
  std::vector<serving::TensorInfo> input_tensor_infos;
  std::vector<std::string> output_names;
  std::vector<serving::TensorInfo> output_tensor_infos;
  std::shared_ptr<api::Model> model;
  uint32_t batch_size = 0;
  std::string device_type;
  uint32_t device_id = 0;
  std::vector<int> without_batch_dim_inputs;
};

class MindSporeModelWrap : public InferSession {
 public:
  MindSporeModelWrap() = default;

  ~MindSporeModelWrap() = default;

  Status InitEnv(serving::DeviceType device_type, uint32_t device_id,
                 const std::map<std::string, std::string> &other_options) override;

  Status FinalizeEnv() override;

  Status LoadModelFromFile(serving::DeviceType device_type, uint32_t device_id, const std::string &file_name,
                           ModelType model_type, const std::vector<int> &without_batch_dim_inputs,
                           const std::map<std::string, std::string> &other_options, uint32_t *model_id) override;

  Status UnloadModel(uint32_t model_id) override;

  // override this method to avoid request/reply data copy
  Status ExecuteModel(uint32_t model_id, const RequestBase &request, ReplyBase *reply) override;
  Status ExecuteModel(uint32_t model_id, const std::vector<TensorBasePtr> &request,
                      std::vector<TensorBasePtr> *reply) override;

  std::vector<serving::TensorInfo> GetInputInfos(uint32_t model_id) const override;

  std::vector<serving::TensorInfo> GetOutputInfos(uint32_t model_id) const override;

  ssize_t GetBatchSize(uint32_t model_id) const override;

  TensorBasePtr MakeInferenceTensor(DataType data_type, const std::vector<int64_t> &shape) const override;

  bool CheckModelSupport(DeviceType device_type, ModelType model_type) const override;

 private:
  std::unordered_map<uint32_t, ApiModelInfo> model_map_;
  uint32_t model_index_ = 0;

  using FuncMakeInBuffer = std::function<api::Buffer(size_t index)>;
  using FuncMakeOutTensor =
    std::function<void(const api::Buffer, DataType data_type, const std::vector<int64_t> &shape)>;
  Status ExecuteModelCommon(uint32_t model_id, size_t request_size, const FuncMakeInBuffer &in_func,
                            const FuncMakeOutTensor &out_func);
  Status GetModelInfos(ApiModelInfo *model_info);
};

class ApiBufferTensorWrap : public TensorBase {
 public:
  ApiBufferTensorWrap();
  ApiBufferTensorWrap(DataType type, const std::vector<int64_t> &shape);
  explicit ApiBufferTensorWrap(const api::Buffer &buffer);
  ~ApiBufferTensorWrap() override;

  void set_data_type(DataType type) override { type_ = type; }
  DataType data_type() const override { return type_; }

  void set_shape(const std::vector<int64_t> &shape) override { shape_ = shape; }
  std::vector<int64_t> shape() const override { return shape_; }

  const uint8_t *data() const override { return static_cast<const uint8_t *>(buffer_.Data()); }
  size_t data_size() const override { return buffer_.DataSize(); }

  bool resize_data(size_t data_len) override { return buffer_.ResizeData(data_len); }
  uint8_t *mutable_data() override { return static_cast<uint8_t *>(buffer_.MutableData()); }

  // For kMSI_String and kMSI_Bytes
  void clear_bytes_data() override { MSI_LOG_EXCEPTION << "Not support for api::Buffer Tensor"; }
  void add_bytes_data(const uint8_t *data, size_t bytes_len) override {
    MSI_LOG_EXCEPTION << "Not support for api::Buffer Tensor";
  }
  size_t bytes_data_size() const override { MSI_LOG_EXCEPTION << "Not support for api::Buffer Tensor"; }
  void get_bytes_data(size_t index, const uint8_t **data, size_t *bytes_len) const override {
    MSI_LOG_EXCEPTION << "Not support for api::Buffer Tensor";
  }

  api::Buffer GetBuffer() const { return buffer_; }

 private:
  DataType type_ = kMSI_Unknown;
  std::vector<int64_t> shape_;
  api::Buffer buffer_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WROERK_MODEL_WRAP_H
