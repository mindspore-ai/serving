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

#include <functional>
#include <map>
#include <vector>

#include "worker/inference/mindspore_model_wrap.h"
#include "include/api/types.h"
#include "include/api/serialization.h"
#include "include/api/context.h"

namespace mindspore {
namespace serving {

Status MindSporeModelWrap::InitEnv(serving::DeviceType device_type, uint32_t device_id,
                                   const std::map<std::string, std::string> &other_options) {
  return SUCCESS;
}

Status MindSporeModelWrap::FinalizeEnv() { return SUCCESS; }

api::DataType TransInferDataType2ApiTypeId(DataType data_type) {
  const std::map<DataType, api::DataType> type2id_map{
    {serving::kMSI_Unknown, api::kMsUnknown}, {serving::kMSI_Bool, api::kMsBool},
    {serving::kMSI_Int8, api::kMsInt8},       {serving::kMSI_Uint8, api::kMsUint8},
    {serving::kMSI_Int16, api::kMsInt16},     {serving::kMSI_Uint16, api::kMsUint16},
    {serving::kMSI_Int32, api::kMsInt32},     {serving::kMSI_Uint32, api::kMsUint32},
    {serving::kMSI_Int64, api::kMsInt64},     {serving::kMSI_Uint64, api::kMsUint64},
    {serving::kMSI_Float16, api::kMsFloat16}, {serving::kMSI_Float32, api::kMsFloat32},
    {serving::kMSI_Float64, api::kMsFloat64},
  };
  auto it = type2id_map.find(data_type);
  if (it == type2id_map.end()) {
    MSI_LOG_WARNING << "Unsupported MSI data type " << data_type;
    return api::kMsUnknown;
  } else {
    return it->second;
  }
}

DataType TransTypeId2InferDataType(api::DataType type_id) {
  const std::map<api::DataType, DataType> id2type_map{
    {api::kMsUnknown, kMSI_Unknown}, {api::kMsBool, kMSI_Bool},     {api::kMsFloat64, kMSI_Float64},
    {api::kMsInt8, kMSI_Int8},       {api::kMsUint8, kMSI_Uint8},   {api::kMsInt16, kMSI_Int16},
    {api::kMsUint16, kMSI_Uint16},   {api::kMsInt32, kMSI_Int32},   {api::kMsUint32, kMSI_Uint32},
    {api::kMsInt64, kMSI_Int64},     {api::kMsUint64, kMSI_Uint64}, {api::kMsFloat16, kMSI_Float16},
    {api::kMsFloat32, kMSI_Float32},
  };
  auto it = id2type_map.find(type_id);
  if (it == id2type_map.end()) {
    MSI_LOG_WARNING << "Unsupported data id " << type_id;
    return kMSI_Unknown;
  } else {
    return it->second;
  }
}

Status MindSporeModelWrap::LoadModelFromFile(serving::DeviceType device_type, uint32_t device_id,
                                             const std::string &file_name, ModelType model_type,
                                             const std::vector<int> &without_batch_dim_inputs,
                                             const std::map<std::string, std::string> &other_options,
                                             uint32_t *model_id) {
  MSI_EXCEPTION_IF_NULL(model_id);
  std::string device_type_str;
  if (device_type == kDeviceTypeAscendMS) {
    device_type_str = api::kDeviceTypeAscend910;
  } else if (device_type == kDeviceTypeAscendCL) {
    device_type_str = api::kDeviceTypeAscend310;
  } else {
    MSI_LOG_EXCEPTION << "Only support Ascend310 or Ascend910 in MindSporeModelWrap";
  }

  std::shared_ptr<api::Model> model = nullptr;
  try {
    api::Context::Instance().SetDeviceTarget(device_type_str).SetDeviceID(device_id);
    auto graph = api::Serialization::LoadModel(file_name, model_type);
    model = std::make_shared<api::Model>(api::GraphCell(graph));
  } catch (std::runtime_error &ex) {
    MSI_LOG_ERROR << "Load model from file failed, model file: " << file_name << ", device_type: '" << device_type_str
                  << "', device_id: " << device_id << ", model type: " << model_type << ", options: " << other_options;
    return FAILED;
  }
  api::Status status = model->Build(other_options);
  if (!status.IsSuccess()) {
    MSI_LOG_ERROR << "Load model from file failed, model file: " << file_name << ", device_type: '" << device_type_str
                  << "', device_id: " << device_id << ", model type: " << model_type << ", options: " << other_options;
    return Status(FAILED, status.StatusMessage());
  }
  model_index_++;
  *model_id = model_index_;
  ApiModelInfo api_model_info;
  api_model_info.model = model;
  api_model_info.device_type = device_type_str;
  api_model_info.device_id = device_id;
  api_model_info.without_batch_dim_inputs = without_batch_dim_inputs;
  auto st = GetModelInfos(&api_model_info);
  if (st != SUCCESS) {
    return st;
  }
  model_map_[*model_id] = api_model_info;
  MSI_LOG_INFO << "Load model from file success, model file: " << file_name << ", device_type: '" << device_type_str
               << "', device_id: " << device_id << ", model type: " << model_type << ", options: " << other_options;
  return SUCCESS;
}

Status MindSporeModelWrap::GetModelInfos(ApiModelInfo *api_model_info) {
  MSI_EXCEPTION_IF_NULL(api_model_info);
  auto model = api_model_info->model;

  bool first_dim_same = true;
  auto find_batch_size = [&first_dim_same, api_model_info](const std::vector<int64_t> &shape) {
    if (first_dim_same) {
      if (shape.empty()) {
        first_dim_same = false;
      } else if (api_model_info->batch_size != 0) {
        if (api_model_info->batch_size != shape[0]) {
          first_dim_same = false;
        }
      } else {
        api_model_info->batch_size = shape[0];
      }
    }
  };
  auto shape_element_num = [](const std::vector<int64_t> &shape) -> size_t {
    size_t elements_nums = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<size_t>());
    return elements_nums;
  };
  auto get_tensor_info_from_tensor = [find_batch_size, shape_element_num, api_model_info](
                                       const std::vector<int64_t> &shape, const api::DataType &data_type,
                                       const size_t mem_size, int input_index = -1) {
    serving::TensorInfo tensor_info;
    tensor_info.shape = shape;
    tensor_info.data_type = TransTypeId2InferDataType(data_type);
    tensor_info.size = mem_size;
    if (tensor_info.size == 0) {
      tensor_info.size = TensorBase::GetTypeSize(tensor_info.data_type) * shape_element_num(tensor_info.shape);
    }
    auto list = api_model_info->without_batch_dim_inputs;
    if (std::find(list.begin(), list.end(), input_index) == list.end()) {
      find_batch_size(tensor_info.shape);
    }
    return tensor_info;
  };
  {  // input infos
    std::vector<std::string> names;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<api::DataType> data_types;
    std::vector<size_t> mem_sizes;
    api::Status status = model->GetInputsInfo(&names, &shapes, &data_types, &mem_sizes);
    if (!status.IsSuccess()) {
      return Status(FAILED, status.StatusMessage());
    }
    if (names.size() != shapes.size() || names.size() != data_types.size() || names.size() != mem_sizes.size()) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Get input infos failed, names size: " << names.size() << ", shapes size: " << shapes.size()
             << ", data_types size: " << data_types.size() << ", mem_sizes: " << mem_sizes.size();
    }
    for (size_t i = 0; i < names.size(); i++) {
      api_model_info->input_names.push_back(names[i]);
      auto tensor_info = get_tensor_info_from_tensor(shapes[i], data_types[i], mem_sizes[i], i);
      if (tensor_info.data_type == kMSI_Unknown) {
        return INFER_STATUS_LOG_ERROR(FAILED) << "Unknown input api data type " << data_types[i];
      }
      api_model_info->input_tensor_infos.push_back(tensor_info);
    }
  }
  {  // output infos
    std::vector<std::string> names;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<api::DataType> data_types;
    std::vector<size_t> mem_sizes;
    api::Status status = model->GetOutputsInfo(&names, &shapes, &data_types, &mem_sizes);
    if (!status.IsSuccess()) {
      return Status(FAILED, status.StatusMessage());
    }
    if (names.size() != shapes.size() || names.size() != data_types.size() || names.size() != mem_sizes.size()) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Get output infos failed, names size: " << names.size() << ", shapes size: " << shapes.size()
             << ", data_types size: " << data_types.size() << ", mem_sizes: " << mem_sizes.size();
    }
    for (size_t i = 0; i < names.size(); i++) {
      api_model_info->output_names.push_back(names[i]);
      auto tensor_info = get_tensor_info_from_tensor(shapes[i], data_types[i], mem_sizes[i]);
      if (tensor_info.data_type == kMSI_Unknown) {
        return INFER_STATUS_LOG_ERROR(FAILED) << "Unknown output api data type " << data_types[i];
      }
      api_model_info->output_tensor_infos.push_back(tensor_info);
    }
  }
  if (!first_dim_same) {
    api_model_info->batch_size = 0;
  }
  return SUCCESS;
}

Status MindSporeModelWrap::UnloadModel(uint32_t model_id) {
  auto it = model_map_.find(model_id);
  if (it == model_map_.end()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Invalid model id " << model_id;
  }
  model_map_.erase(it);
  return SUCCESS;
}

Status MindSporeModelWrap::ExecuteModel(uint32_t model_id, const RequestBase &request, serving::ReplyBase *reply) {
  MSI_EXCEPTION_IF_NULL(reply);
  FuncMakeInBuffer func_in = [&request](size_t index) {
    auto input_tensor = request[index];
    return api::Buffer(input_tensor->data(), input_tensor->data_size());
  };

  FuncMakeOutTensor func_out = [&reply](const api::Buffer &result_tensor, DataType data_type,
                                        const std::vector<int64_t> &shape) {
    auto tensor = reply->add();
    MSI_EXCEPTION_IF_NULL(tensor);
    tensor->set_data(result_tensor.Data(), result_tensor.DataSize());
    tensor->set_data_type(data_type);
    tensor->set_shape(shape);
  };
  return ExecuteModelCommon(model_id, request.size(), func_in, func_out);
}

Status MindSporeModelWrap::ExecuteModel(uint32_t model_id, const std::vector<TensorBasePtr> &request,
                                        std::vector<TensorBasePtr> *reply) {
  MSI_EXCEPTION_IF_NULL(reply);
  FuncMakeInBuffer func_in = [&request](size_t index) {
    auto &input_tensor = request[index];
    auto api_buffer_wrap = std::dynamic_pointer_cast<ApiBufferTensorWrap>(input_tensor);
    if (api_buffer_wrap) {
      return api_buffer_wrap->GetBuffer();
    } else {
      return api::Buffer(input_tensor->data(), input_tensor->data_size());
    }
  };
  FuncMakeOutTensor func_out = [&reply](const api::Buffer &result_tensor, DataType data_type,
                                        const std::vector<int64_t> &shape) {
    auto tensor = std::make_shared<ApiBufferTensorWrap>(result_tensor);
    tensor->set_data_type(data_type);
    tensor->set_shape(shape);
    reply->push_back(tensor);
  };
  return ExecuteModelCommon(model_id, request.size(), func_in, func_out);
}

Status MindSporeModelWrap::ExecuteModelCommon(uint32_t model_id, size_t request_size, const FuncMakeInBuffer &in_func,
                                              const FuncMakeOutTensor &out_func) {
  auto it = model_map_.find(model_id);
  if (it == model_map_.end()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Invalid model id " << model_id;
  }
  auto &model_info = it->second;
  auto model = model_info.model;
  auto &input_names = model_info.input_names;
  auto &output_names = model_info.output_names;
  if (input_names.size() != request_size) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Inputs size not match, request inputs size " << request_size
                                          << ", model inputs size " << input_names.size();
  }
  std::vector<api::Buffer> inputs;
  for (size_t i = 0; i < input_names.size(); i++) {
    inputs.push_back(in_func(i));
  }
  std::vector<api::Buffer> outputs;
  api::Status status = model->Predict(inputs, &outputs);
  if (!status.IsSuccess()) {
    MSI_LOG_ERROR << "Predict failed: " << status.StatusMessage();
    return FAILED;
  }
  if (outputs.size() != output_names.size()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Outputs size not match, predict outputs size " << outputs.size()
                                          << ", model outputs size " << output_names.size();
  }
  auto &output_infos = model_info.output_tensor_infos;
  for (size_t i = 0; i < output_names.size(); i++) {
    auto &result_tensor = outputs[i];
    auto &output_info = output_infos[i];
    if (result_tensor.DataSize() != output_info.size) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Get output failed, predict output data size " << result_tensor.DataSize()
             << " not match model info data size " << output_info.size << ", output_name " << output_names[i];
    }
    out_func(result_tensor, output_info.data_type, output_info.shape);
  }
  return SUCCESS;
}

std::vector<serving::TensorInfo> MindSporeModelWrap::GetInputInfos(uint32_t model_id) const {
  auto it = model_map_.find(model_id);
  if (it == model_map_.end()) {
    MSI_LOG_ERROR << "Invalid model id " << model_id;
    return {};
  }
  auto &model_info = it->second;
  return model_info.input_tensor_infos;
}

std::vector<serving::TensorInfo> MindSporeModelWrap::GetOutputInfos(uint32_t model_id) const {
  auto it = model_map_.find(model_id);
  if (it == model_map_.end()) {
    MSI_LOG_ERROR << "Invalid model id " << model_id;
    return {};
  }
  auto &model_info = it->second;
  return model_info.output_tensor_infos;
}

ssize_t MindSporeModelWrap::GetBatchSize(uint32_t model_id) const {
  auto it = model_map_.find(model_id);
  if (it == model_map_.end()) {
    MSI_LOG_ERROR << "Invalid model id " << model_id;
    return {};
  }
  auto &model_info = it->second;
  return model_info.batch_size;
}

TensorBasePtr MindSporeModelWrap::MakeInferenceTensor(DataType data_type, const std::vector<int64_t> &shape) const {
  return std::make_shared<ApiBufferTensorWrap>(data_type, shape);
}

bool MindSporeModelWrap::CheckModelSupport(DeviceType device_type, ModelType model_type) const {
  std::string device_type_str;
  switch (device_type) {
    case kDeviceTypeAscendMS:
      device_type_str = api::kDeviceTypeAscend910;
      break;
    case kDeviceTypeAscendCL:
      device_type_str = api::kDeviceTypeAscend310;
      break;
    default:
      return false;
  }
  return api::Model::CheckModelSupport(device_type_str, model_type);
}

ApiBufferTensorWrap::ApiBufferTensorWrap() = default;

ApiBufferTensorWrap::ApiBufferTensorWrap(DataType type, const std::vector<int64_t> &shape)
    : type_(type), shape_(shape) {
  size_t data_len = itemsize() * TensorBase::element_cnt();
  buffer_.ResizeData(data_len);
}

ApiBufferTensorWrap::ApiBufferTensorWrap(const api::Buffer &buffer) : buffer_(buffer) {}

ApiBufferTensorWrap::~ApiBufferTensorWrap() = default;

REGISTER_INFER_SEESION(serving::kDeviceTypeAscendCL, kOM, MindSporeModelWrap, 1);
REGISTER_INFER_SEESION(serving::kDeviceTypeAscendCL, kMindIR, MindSporeModelWrap, 1);
REGISTER_INFER_SEESION(serving::kDeviceTypeAscendMS, kMindIR, MindSporeModelWrap, 1);

}  // namespace serving
}  // namespace mindspore
