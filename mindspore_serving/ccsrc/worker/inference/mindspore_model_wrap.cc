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

namespace mindspore {
namespace serving {

extern "C" {
MS_API InferenceBase *ServingCreateInfer() {
  auto obj = new MindSporeModelWrap();
  return dynamic_cast<InferenceBase *>(obj);
}
}

mindspore::DataType TransInferDataType2ApiTypeId(DataType data_type) {
  const std::map<DataType, mindspore::DataType> type2id_map{
    {serving::kMSI_Unknown, mindspore::DataType::kTypeUnknown},
    {serving::kMSI_Bool, mindspore::DataType::kNumberTypeBool},
    {serving::kMSI_Int8, mindspore::DataType::kNumberTypeInt8},
    {serving::kMSI_Uint8, mindspore::DataType::kNumberTypeUInt8},
    {serving::kMSI_Int16, mindspore::DataType::kNumberTypeInt16},
    {serving::kMSI_Uint16, mindspore::DataType::kNumberTypeUInt16},
    {serving::kMSI_Int32, mindspore::DataType::kNumberTypeInt32},
    {serving::kMSI_Uint32, mindspore::DataType::kNumberTypeUInt32},
    {serving::kMSI_Int64, mindspore::DataType::kNumberTypeInt64},
    {serving::kMSI_Uint64, mindspore::DataType::kNumberTypeUInt64},
    {serving::kMSI_Float16, mindspore::DataType::kNumberTypeFloat16},
    {serving::kMSI_Float32, mindspore::DataType::kNumberTypeFloat32},
    {serving::kMSI_Float64, mindspore::DataType::kNumberTypeFloat64},
  };
  auto it = type2id_map.find(data_type);
  if (it == type2id_map.end()) {
    MSI_LOG_WARNING << "Unsupported MSI data type " << data_type;
    return mindspore::DataType::kTypeUnknown;
  } else {
    return it->second;
  }
}

DataType TransTypeId2InferDataType(mindspore::DataType type_id) {
  const std::map<mindspore::DataType, DataType> id2type_map{
    {mindspore::DataType::kTypeUnknown, kMSI_Unknown},       {mindspore::DataType::kNumberTypeBool, kMSI_Bool},
    {mindspore::DataType::kNumberTypeFloat64, kMSI_Float64}, {mindspore::DataType::kNumberTypeInt8, kMSI_Int8},
    {mindspore::DataType::kNumberTypeUInt8, kMSI_Uint8},     {mindspore::DataType::kNumberTypeInt16, kMSI_Int16},
    {mindspore::DataType::kNumberTypeUInt16, kMSI_Uint16},   {mindspore::DataType::kNumberTypeInt32, kMSI_Int32},
    {mindspore::DataType::kNumberTypeUInt32, kMSI_Uint32},   {mindspore::DataType::kNumberTypeInt64, kMSI_Int64},
    {mindspore::DataType::kNumberTypeUInt64, kMSI_Uint64},   {mindspore::DataType::kNumberTypeFloat16, kMSI_Float16},
    {mindspore::DataType::kNumberTypeFloat32, kMSI_Float32},
  };
  auto it = id2type_map.find(type_id);
  if (it == id2type_map.end()) {
    MSI_LOG_WARNING << "Unsupported data id " << static_cast<int>(type_id);
    return kMSI_Unknown;
  } else {
    return it->second;
  }
}

Status MindSporeModelWrap::LoadModelFromFile(serving::DeviceType device_type, uint32_t device_id,
                                             const std::string &file_name, ModelType model_type, bool with_batch_dim,
                                             const std::vector<int> &without_batch_dim_inputs,
                                             const std::map<std::string, std::string> &other_options) {
  auto ms_model_type = GetMsModelType(model_type);
  if (ms_model_type == mindspore::kUnknownType) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Invalid model type " << model_type;
  }

  std::shared_ptr<mindspore::Model> model = nullptr;
  try {
    mindspore::Graph graph;
    auto ms_status = mindspore::Serialization::Load(file_name, ms_model_type, &graph);
    auto context = TransformModelContext(device_type, device_id, other_options);
    model = std::make_shared<mindspore::Model>();
    mindspore::Status status = model->Build(mindspore::GraphCell(graph), context);
    if (!status.IsOk()) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Load model from file failed, model file: " << file_name << ", device_type: '" << device_type
             << "', device_id: " << device_id << ", model type: " << model_type << ", options: " << other_options
             << ", build error detail: " << status.ToString();
    }
  } catch (std::runtime_error &ex) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Load model from file failed, model file: " << file_name << ", device_type: '" << device_type
           << "', device_id: " << device_id << ", model type: " << model_type << ", options: " << other_options
           << ", build error detail: " << ex.what();
  }

  ApiModelInfo api_model_info;
  api_model_info.model = model;
  api_model_info.device_type = device_type;
  api_model_info.device_id = device_id;
  api_model_info.with_batch_dim = with_batch_dim;
  api_model_info.without_batch_dim_inputs = without_batch_dim_inputs;
  auto st = GetModelInfos(&api_model_info);
  if (st != SUCCESS) {
    return st;
  }
  GetModelBatchSize(&api_model_info);
  model_ = api_model_info;
  MSI_LOG_INFO << "Load model from file success, model file: " << file_name << ", device_type: '" << device_type
               << "', device_id: " << device_id << ", model type: " << model_type << ", options: " << other_options;
  return SUCCESS;
}

std::shared_ptr<DeviceInfoContext> MindSporeModelWrap::TransformAscend310ModelContext(
  uint32_t device_id, const std::map<std::string, std::string> &options) {
  auto context_info = std::make_shared<Ascend310DeviceInfo>();
  context_info->SetDeviceID(device_id);

  using ContextStrFun = std::function<void(const std::string &)>;
  ContextStrFun set_output_type = [context_info](const std::string &val) {
    // "FP32", "FP16", "UINT8"
    if (val == "FP32") {
      context_info->SetOutputType(mindspore::DataType::kNumberTypeFloat32);
    } else if (val == "FP16") {
      context_info->SetOutputType(mindspore::DataType::kNumberTypeFloat16);
    } else if (val == "UINT8") {
      context_info->SetOutputType(mindspore::DataType::kNumberTypeUInt8);
    } else {
      MSI_LOG_ERROR << "Set model context output type failed, unknown data type " << val;
    }
  };
  for (auto &item : options) {
    const auto &key = item.first;
    const auto &value = item.second;
    if (key == "acl_option.insert_op_config_file_path") {
      context_info->SetInsertOpConfigPath(value);
    } else if (key == "acl_option.input_format") {
      context_info->SetInputFormat(value);
    } else if (key == "acl_option.input_shape") {
      context_info->SetInputShape(value);
    } else if (key == "acl_option.output_type") {
      set_output_type(value);
    } else if (key == "acl_option.precision_mode") {
      context_info->SetPrecisionMode(value);
    } else if (key == "acl_option.op_select_impl_mode") {
      context_info->SetOpSelectImplMode(value);
    }
  }
  return context_info;
}

std::shared_ptr<DeviceInfoContext> MindSporeModelWrap::TransformAscend910ModelContext(
  uint32_t device_id, const std::map<std::string, std::string> &options) {
  auto context_info = std::make_shared<Ascend910DeviceInfo>();
  context_info->SetDeviceID(device_id);
  return context_info;
}
std::shared_ptr<DeviceInfoContext> MindSporeModelWrap::TransformNvidiaGPUModelContext(
  uint32_t device_id, const std::map<std::string, std::string> &options) {
  auto context_info = std::make_shared<NvidiaGPUDeviceInfo>();
  context_info->SetDeviceID(device_id);

  for (auto &item : options) {
    const auto &key = item.first;
    const auto &value = item.second;
    if (key == "gpu_option.enable_trt_infer") {
      if (value == "True") {
        context_info->SetGpuTrtInferMode(true);
      } else {
        context_info->SetGpuTrtInferMode(false);
      }
    }
  }
  return context_info;
}

std::shared_ptr<Context> MindSporeModelWrap::TransformModelContext(serving::DeviceType device_type, uint32_t device_id,
                                                                   const std::map<std::string, std::string> &options) {
  auto context = std::make_shared<mindspore::Context>();
  std::shared_ptr<mindspore::DeviceInfoContext> context_info = nullptr;
  if (device_type == kDeviceTypeAscendMS) {
    context_info = TransformAscend910ModelContext(device_id, options);
  } else if (device_type == kDeviceTypeAscendCL) {
    context_info = TransformAscend310ModelContext(device_id, options);
  } else if (device_type == kDeviceTypeGpu) {
    context_info = TransformNvidiaGPUModelContext(device_id, options);
  }
  if (context_info != nullptr) {
    context->MutableDeviceInfo().push_back(context_info);
  }
  return context;
}

Status MindSporeModelWrap::GetModelInfos(ApiModelInfo *api_model_info) {
  MSI_EXCEPTION_IF_NULL(api_model_info);
  auto model = api_model_info->model;

  auto get_tensor_info_from_tensor = [](const mindspore::MSTensor &ms_tensor) {
    serving::TensorInfo tensor_info;
    tensor_info.shape = ms_tensor.Shape();
    tensor_info.data_type = TransTypeId2InferDataType(ms_tensor.DataType());
    tensor_info.size = ms_tensor.DataSize();
    if (tensor_info.size == 0) {
      auto &shape = tensor_info.shape;
      size_t elements_nums = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<size_t>());
      tensor_info.size = TensorBase::GetTypeSize(tensor_info.data_type) * elements_nums;
    }
    return tensor_info;
  };
  {  // input infos
    auto input_infos = model->GetInputs();
    for (size_t i = 0; i < input_infos.size(); i++) {
      auto &info = input_infos[i];
      auto tensor_info = get_tensor_info_from_tensor(info);
      if (tensor_info.data_type == kMSI_Unknown) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "Unknown input mindspore data type " << static_cast<int>(info.DataType());
      }
      api_model_info->input_tensor_infos.push_back(tensor_info);
      api_model_info->input_names.push_back(info.Name());
    }
  }
  {  // output infos
    auto output_infos = model->GetOutputs();
    for (auto &info : output_infos) {
      auto tensor_info = get_tensor_info_from_tensor(info);
      if (tensor_info.data_type == kMSI_Unknown) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "Unknown output mindspore data type " << static_cast<int>(info.DataType());
      }
      api_model_info->output_tensor_infos.push_back(tensor_info);
      api_model_info->output_names.push_back(info.Name());
    }
  }
  return SUCCESS;
}

void MindSporeModelWrap::GetModelBatchSize(ApiModelInfo *api_model_info) {
  MSI_EXCEPTION_IF_NULL(api_model_info);
  bool first_dim_same = true;
  auto find_batch_size = [&first_dim_same, api_model_info](const std::vector<int64_t> &shape) {
    if (!api_model_info->with_batch_dim) {
      first_dim_same = false;
      return;
    }
    if (!first_dim_same) {
      return;
    }
    if (shape.empty()) {
      first_dim_same = false;
      return;
    }
    if (api_model_info->batch_size != 0) {
      if (api_model_info->batch_size != shape[0]) {
        first_dim_same = false;
      }
    } else {
      api_model_info->batch_size = shape[0];
    }
  };

  auto list = api_model_info->without_batch_dim_inputs;
  auto size = api_model_info->input_tensor_infos.size();
  for (size_t i = 0; i < size; i++) {
    if (std::find(list.begin(), list.end(), i) == list.end()) {
      auto &info = api_model_info->input_tensor_infos[i];
      find_batch_size(info.shape);
    }
  }
  for (auto &info : api_model_info->output_tensor_infos) {
    find_batch_size(info.shape);
  }
  if (!first_dim_same) {
    api_model_info->batch_size = 0;
  }
}

Status MindSporeModelWrap::UnloadModel() {
  model_.model = nullptr;
  return SUCCESS;
}

Status MindSporeModelWrap::ExecuteModel(const RequestBase &request, serving::ReplyBase *reply) {
  MSI_EXCEPTION_IF_NULL(reply);
  FuncMakeInBuffer func_in = [&request](size_t index, const std::string &name) {
    auto input_tensor = request[index];
    if (input_tensor == nullptr || input_tensor->data() == nullptr) {
      MSI_LOG_EXCEPTION << "Input tensor data cannot be nullptr, index " << index;
    }
    return mindspore::MSTensor::CreateRefTensor(name, TransInferDataType2ApiTypeId(input_tensor->data_type()),
                                                input_tensor->shape(), const_cast<uint8_t *>(input_tensor->data()),
                                                input_tensor->data_size());
  };

  FuncMakeOutTensor func_out = [&reply](const mindspore::MSTensor &result_tensor, DataType data_type,
                                        const std::vector<int64_t> &shape) {
    if (result_tensor.IsDevice()) {
      MSI_LOG_EXCEPTION << "Can not support device type tensor";
    }
    auto tensor = reply->add();
    MSI_EXCEPTION_IF_NULL(tensor);
    tensor->set_data(result_tensor.Data().get(), result_tensor.DataSize());
    tensor->set_data_type(data_type);
    tensor->set_shape(shape);
  };
  return ExecuteModelCommon(request.size(), func_in, func_out);
}

Status MindSporeModelWrap::ExecuteModel(const std::vector<TensorBasePtr> &request, std::vector<TensorBasePtr> *reply) {
  MSI_EXCEPTION_IF_NULL(reply);
  FuncMakeInBuffer func_in = [&request](size_t index, const std::string &name) {
    auto &input_tensor = request[index];
    return mindspore::MSTensor::CreateRefTensor(name, TransInferDataType2ApiTypeId(input_tensor->data_type()),
                                                input_tensor->shape(), const_cast<uint8_t *>(input_tensor->data()),
                                                input_tensor->data_size());
  };
  FuncMakeOutTensor func_out = [&reply](const mindspore::MSTensor &result_tensor, DataType data_type,
                                        const std::vector<int64_t> &shape) {
    if (result_tensor.IsDevice()) {
      MSI_LOG_EXCEPTION << "Can not support device type tensor";
    }
    auto tensor = std::make_shared<ApiBufferTensorWrap>(result_tensor);
    tensor->set_data_type(data_type);
    tensor->set_shape(shape);
    reply->push_back(tensor);
  };
  return ExecuteModelCommon(request.size(), func_in, func_out);
}

Status MindSporeModelWrap::ExecuteModelCommon(size_t request_size, const FuncMakeInBuffer &in_func,
                                              const FuncMakeOutTensor &out_func) {
  if (model_.model == nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Model is not loaded";
  }
  auto &model_info = model_;
  auto model = model_info.model;
  auto &input_names = model_info.input_names;
  auto &output_names = model_info.output_names;
  if (input_names.size() != request_size) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Inputs size not match, request inputs size " << request_size
                                          << ", model inputs size " << input_names.size();
  }
  std::vector<mindspore::MSTensor> inputs;
  for (size_t i = 0; i < input_names.size(); i++) {
    auto tensor = in_func(i, input_names[i]);
    if (tensor == nullptr) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Failed to create input " << i << " MSTensor";
    }
    inputs.push_back(*tensor);
    mindspore::MSTensor::DestroyTensorPtr(tensor);
  }
  std::vector<mindspore::MSTensor> outputs;
  mindspore::Status status = model->Predict(inputs, &outputs);
  if (!status.IsOk()) {
    MSI_LOG_ERROR << "Predict failed: " << status.ToString();
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

std::vector<serving::TensorInfo> MindSporeModelWrap::GetInputInfos() const { return model_.input_tensor_infos; }

std::vector<serving::TensorInfo> MindSporeModelWrap::GetOutputInfos() const { return model_.output_tensor_infos; }

ssize_t MindSporeModelWrap::GetBatchSize() const { return model_.batch_size; }

bool MindSporeModelWrap::CheckModelSupport(DeviceType device_type, ModelType model_type) const {
  auto ms_device_type = GetMsDeviceType(device_type);
  if (ms_device_type == mindspore::kInvalidDeviceType) {
    return false;
  }
  auto ms_model_type = GetMsModelType(model_type);
  if (ms_model_type == mindspore::kUnknownType) {
    return false;
  }
  return mindspore::Model::CheckModelSupport(ms_device_type, ms_model_type);
}

mindspore::ModelType MindSporeModelWrap::GetMsModelType(serving::ModelType model_type) {
  mindspore::ModelType ms_model_type;
  switch (model_type) {
    case kMindIR:
      ms_model_type = mindspore::kMindIR;
      break;
    case kAIR:
      ms_model_type = mindspore::kAIR;
      break;
    case kOM:
      ms_model_type = mindspore::kOM;
      break;
    case kONNX:
      ms_model_type = mindspore::kONNX;
      break;
    default:
      ms_model_type = mindspore::kUnknownType;
  }
  return ms_model_type;
}

mindspore::DeviceType MindSporeModelWrap::GetMsDeviceType(serving::DeviceType device_type) {
  mindspore::DeviceType ms_device_type = mindspore::DeviceType::kInvalidDeviceType;
  switch (device_type) {
    case kDeviceTypeAscendMS:
      ms_device_type = mindspore::DeviceType::kAscend910;
      break;
    case kDeviceTypeAscendCL:
      ms_device_type = mindspore::DeviceType::kAscend310;
      break;
    case kDeviceTypeGpu:
      ms_device_type = mindspore::DeviceType::kNvidiaGPU;
      break;
    case kDeviceTypeCpu:
      ms_device_type = mindspore::DeviceType::kCPU;
      break;
    default:
      break;
  }
  return ms_device_type;
}

ApiBufferTensorWrap::ApiBufferTensorWrap() = default;

ApiBufferTensorWrap::ApiBufferTensorWrap(const mindspore::MSTensor &tensor) : tensor_(tensor) {}

ApiBufferTensorWrap::~ApiBufferTensorWrap() = default;
}  // namespace serving
}  // namespace mindspore
