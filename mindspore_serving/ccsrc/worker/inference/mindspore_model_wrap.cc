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
  std::string device_type_str;
  if (device_type == kDeviceTypeAscendMS) {
    device_type_str = mindspore::kDeviceTypeAscend910;
  } else if (device_type == kDeviceTypeAscendCL) {
    device_type_str = mindspore::kDeviceTypeAscend310;
  } else {
    MSI_LOG_EXCEPTION << "Only support Ascend310 or Ascend910 in MindSporeModelWrap";
  }

  std::shared_ptr<mindspore::Model> model = nullptr;
  try {
    mindspore::GlobalContext::SetGlobalDeviceTarget(device_type_str);
    mindspore::GlobalContext::SetGlobalDeviceID(device_id);
    auto graph = mindspore::Serialization::LoadModel(file_name, model_type);
    auto context = TransformModelContext(other_options);
    model = std::make_shared<mindspore::Model>(mindspore::GraphCell(graph), context);
  } catch (std::runtime_error &ex) {
    MSI_LOG_ERROR << "Load model from file failed, model file: " << file_name << ", device_type: '" << device_type_str
                  << "', device_id: " << device_id << ", model type: " << model_type << ", options: " << other_options;
    return FAILED;
  }
  mindspore::Status status = model->Build();
  if (!status.IsOk()) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Load model from file failed, model file: " << file_name << ", device_type: '" << device_type_str
           << "', device_id: " << device_id << ", model type: " << model_type << ", options: " << other_options
           << ", build error detail: " << status.ToString();
  }
  ApiModelInfo api_model_info;
  api_model_info.model = model;
  api_model_info.device_type = device_type_str;
  api_model_info.device_id = device_id;
  api_model_info.with_batch_dim = with_batch_dim;
  api_model_info.without_batch_dim_inputs = without_batch_dim_inputs;
  auto st = GetModelInfos(&api_model_info);
  if (st != SUCCESS) {
    return st;
  }
  GetModelBatchSize(&api_model_info);
  model_ = api_model_info;
  MSI_LOG_INFO << "Load model from file success, model file: " << file_name << ", device_type: '" << device_type_str
               << "', device_id: " << device_id << ", model type: " << model_type << ", options: " << other_options;
  return SUCCESS;
}

std::shared_ptr<Context> MindSporeModelWrap::TransformModelContext(const std::map<std::string, std::string> &options) {
  using ContextStrFun = std::function<void(const std::shared_ptr<Context> &, const std::string &)>;
  ContextStrFun set_output_type = [](const std::shared_ptr<Context> &context, const std::string &val) {
    // "FP32", "FP16", "UINT8"
    if (val == "FP32") {
      mindspore::ModelContext::SetOutputType(context, mindspore::DataType::kNumberTypeFloat32);
    } else if (val == "FP16") {
      mindspore::ModelContext::SetOutputType(context, mindspore::DataType::kNumberTypeFloat16);
    } else if (val == "UINT8") {
      mindspore::ModelContext::SetOutputType(context, mindspore::DataType::kNumberTypeUInt8);
    } else {
      MSI_LOG_ERROR << "Set model context output type failed, unknown data type " << val;
    }
  };
  std::map<std::string, ContextStrFun> option_map = {
    {"acl_option.insert_op_config_file_path", mindspore::ModelContext::SetInsertOpConfigPath},
    {"acl_option.input_format", mindspore::ModelContext::SetInputFormat},
    {"acl_option.input_shape", mindspore::ModelContext::SetInputShape},
    {"acl_option.output_type", set_output_type},
    {"acl_option.precision_mode", mindspore::ModelContext::SetPrecisionMode},
    {"acl_option.op_select_impl_mode", mindspore::ModelContext::SetOpSelectImplMode},
  };
  auto context = std::make_shared<mindspore::ModelContext>();
  for (auto &item : options) {
    const auto &key = item.first;
    const auto &value = item.second;
    auto it = option_map.find(key);
    if (it != option_map.end()) {
      MSI_LOG_INFO << "Set context options, key: " << key << ", value: " << value;
      it->second(context, value);
    }
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
    inputs.push_back(in_func(i, input_names[i]));
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
  std::string device_type_str;
  switch (device_type) {
    case kDeviceTypeAscendMS:
      if (model_type != kMindIR) {
        return false;
      }
      device_type_str = mindspore::kDeviceTypeAscend910;
      break;
    case kDeviceTypeAscendCL:
      if (model_type != kMindIR && model_type != kOM) {
        return false;
      }
      device_type_str = mindspore::kDeviceTypeAscend310;
      break;
    default:
      return false;
  }
  return mindspore::Model::CheckModelSupport(device_type_str, model_type);
}

ApiBufferTensorWrap::ApiBufferTensorWrap() = default;

ApiBufferTensorWrap::ApiBufferTensorWrap(const mindspore::MSTensor &tensor) : tensor_(tensor) {}

ApiBufferTensorWrap::~ApiBufferTensorWrap() = default;

}  // namespace serving
}  // namespace mindspore
