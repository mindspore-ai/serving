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

#include "worker/inference/mindspore_model_wrap.h"
#include <unistd.h>
#include <sys/stat.h>
#include <functional>
#include <map>
#include <vector>

namespace mindspore {
namespace serving {

extern "C" {
MS_API InferenceBase *ServingCreateInfer() {
  auto obj = new MindSporeModelWrap();
  return dynamic_cast<InferenceBase *>(obj);
}
}

std::mutex MindSporeModelWrap::infer_mutex_;

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
                                             const std::vector<std::string> &file_names, ModelType model_type,
                                             bool with_batch_dim, const std::vector<int> &without_batch_dim_inputs,
                                             const ModelContext &model_context, const std::string &dec_key,
                                             const std::string &dec_mode, const std::string &config_file,
                                             bool enable_lite) {
  char path[PATH_MAX];
  std::string current_path = getcwd(path, PATH_MAX);
  std::string build_dir = current_path + "/models_build_temp/";
  (void)mkdir(build_dir.c_str(), S_IRWXU | S_IRWXG);
  build_dir += "device_" + std::to_string(device_id);
  (void)mkdir(build_dir.c_str(), S_IRWXU | S_IRWXG);
  auto error_no = chdir(build_dir.c_str());
  if (error_no != 0) {
    MSI_LOG_WARNING << "Failed to call chdir, target build directory: " << build_dir << ", error no: " << error_no;
  }

  auto status =
    LoadModelFromFileInner(device_type, device_id, file_names, model_type, with_batch_dim, without_batch_dim_inputs,
                           model_context, dec_key, dec_mode, config_file, enable_lite);

  error_no = chdir(current_path.c_str());
  if (error_no != 0) {
    MSI_LOG_WARNING << "Failed to call chdir, target directory: " << current_path << ", error no: " << error_no;
  }
  return status;
}

Status MindSporeModelWrap::LoadModelFromFileInner(serving::DeviceType device_type, uint32_t device_id,
                                                  const std::vector<std::string> &file_names, ModelType model_type,
                                                  bool with_batch_dim, const std::vector<int> &without_batch_dim_inputs,
                                                  const ModelContext &model_context, const std::string &dec_key,
                                                  const std::string &dec_mode, const std::string &config_file,
                                                  bool enable_lite) {
  auto ms_model_type = GetMsModelType(model_type);
  if (ms_model_type == mindspore::kUnknownType) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Invalid model type " << model_type;
  }
  std::vector<std::shared_ptr<mindspore::Model>> models;
  try {
    std::vector<mindspore::Graph> graphs;
    mindspore::Key key;
    if (!dec_key.empty()) {
      auto rt = memcpy_s(key.key, sizeof(key.key), dec_key.data(), dec_key.size());
      if (rt != EOK) {
        return INFER_STATUS_LOG_ERROR(FAILED) << "Load model from file failed, dec key size " << dec_key.size()
                                              << " should less than " << key.max_key_len;
      }
      key.len = dec_key.size();
    } else {
      key.len = 0;
    }

    mindspore::Status ms_status;
    if (file_names.size() > 1) {
      ms_status = mindspore::Serialization::Load(file_names, ms_model_type, &graphs, key, dec_mode);
    } else {
      graphs.emplace_back(mindspore::Graph());
      ms_status = mindspore::Serialization::Load(file_names[0], ms_model_type, &graphs[0], key, dec_mode);
    }

    (void)memset_s(key.key, sizeof(key.key), 0, key.max_key_len);
    if (!ms_status.IsOk()) {
      MSI_LOG_ERROR << "Load model from file failed, model file: " << file_names << ", device_type: '" << device_type
                    << "', device_id: " << device_id << ", model type: " << model_type
                    << ", model context: " << model_context.AsString() << ", dec mode: " << dec_mode
                    << ", load error detail: " << ms_status.ToString();
      return Status(FAILED, ms_status.ToString());
    }
    if (file_names.size() > 1 && graphs.size() != file_names.size()) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Load model from file failed, generate graphs size " << graphs.size()
                                            << " should equal to " << file_names.size();
    }
    auto context = TransformModelContext(device_type, device_id, model_context, enable_lite);
    for (size_t i = 0; i < file_names.size(); i++) {
      auto model = std::make_shared<mindspore::Model>();
      if (!config_file.empty()) {
        auto load_status = model->LoadConfig(config_file);
        if (!load_status.IsOk()) {
          return INFER_STATUS_LOG_ERROR(FAILED)
                 << "Load config file: " << config_file << " failed, error details: " << load_status.ToString();
        }
      }
      mindspore::Status status;
      status = model->Build(mindspore::GraphCell(graphs[i]), context);
      if (!status.IsOk()) {
        MSI_LOG_ERROR << "Load model from file failed, model file: " << file_names[i] << ", device_type: '"
                      << device_type << "', device_id: " << device_id << ", model type: " << model_type
                      << ", model context: " << model_context.AsString()
                      << ", build error detail: " << status.ToString();
        return Status(FAILED, status.ToString());
      }
      models.push_back(model);
    }
  } catch (std::runtime_error &ex) {
    MSI_LOG_ERROR << "Load model from file failed, model file: " << file_names << ", device_type: '" << device_type
                  << "', device_id: " << device_id << ", model type: " << model_type
                  << ", model context: " << model_context.AsString() << ", build error detail: " << ex.what();
    return Status(FAILED, ex.what());
  }
  return SetApiModelInfo(device_type, device_id, file_names, model_type, with_batch_dim, without_batch_dim_inputs,
                         model_context, models);
}

Status MindSporeModelWrap::SetApiModelInfo(serving::DeviceType device_type, uint32_t device_id,
                                           const std::vector<std::string> &file_names, ModelType model_type,
                                           bool with_batch_dim, const std::vector<int> &without_batch_dim_inputs,
                                           const ModelContext &model_context,
                                           const std::vector<std::shared_ptr<mindspore::Model>> &models) {
  uint64_t last_batch_size = 0;
  common_model_info_.device_type = device_type;
  common_model_info_.device_id = device_id;
  common_model_info_.with_batch_dim = with_batch_dim;
  common_model_info_.without_batch_dim_inputs = without_batch_dim_inputs;
  for (size_t i = 0; i < file_names.size(); i++) {
    ApiModelInfo api_model_info;
    api_model_info.model = models[i];
    auto st = GetModelInfos(&api_model_info);
    if (st != SUCCESS) {
      return st;
    }

    MSI_LOG_INFO << "Print model info, model file: '" << file_names[i] << "', subgraph " << i;
    MSI_LOG_INFO << "Model input infos: count " << api_model_info.input_tensor_infos.size();
    for (auto &item : api_model_info.input_tensor_infos) {
      MSI_LOG_INFO << item.shape << ", " << item.data_type << ", " << item.size;
    }
    MSI_LOG_INFO << "Model output infos: count " << api_model_info.output_tensor_infos.size();
    for (auto &item : api_model_info.output_tensor_infos) {
      MSI_LOG_INFO << item.shape << ", " << item.data_type << ", " << item.size;
    }

    auto status = CalculateBatchSize(&api_model_info);
    if (status != SUCCESS) {
      MSI_LOG_ERROR << "Calculate batch size failed, model file: " << file_names[i] << ", subgraph: " << i;
      return status;
    }
    if (last_batch_size != 0 && last_batch_size != common_model_info_.batch_size) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Expect batch size to be same, last batch size: " << last_batch_size
                                            << ", subgraph " << i << " batch size: " << common_model_info_.batch_size;
    }
    last_batch_size = common_model_info_.batch_size;
    models_.push_back(api_model_info);
  }
  MSI_LOG_INFO << "Load model from file success, model file: " << file_names << ", device_type: '" << device_type
               << "', device_id: " << device_id << ", model type: " << model_type
               << ", model context: " << model_context.AsString();
  return SUCCESS;
}

std::shared_ptr<DeviceInfoContext> MindSporeModelWrap::TransformAscendModelContext(uint32_t device_id,
                                                                                   const DeviceInfo &device_info) {
  auto context_info = std::make_shared<AscendDeviceInfo>();
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

  for (auto &item : device_info) {
    const auto &key = item.first;
    const auto &value = item.second;
    if (key == "insert_op_cfg_path") {
      context_info->SetInsertOpConfigPath(value);
    } else if (key == "input_format") {
      context_info->SetInputFormat(value);
    } else if (key == "input_shape") {
      context_info->SetInputShape(value);
    } else if (key == "output_type") {
      set_output_type(value);
    } else if (key == "precision_mode") {
      context_info->SetPrecisionMode(value);
    } else if (key == "op_select_impl_mode") {
      context_info->SetOpSelectImplMode(value);
    } else if (key == "fusion_switch_config_path") {
      context_info->SetFusionSwitchConfigPath(value);
    } else if (key == "buffer_optimize_mode") {
      context_info->SetBufferOptimizeMode(value);
    }
  }
  return context_info;
}

std::shared_ptr<DeviceInfoContext> MindSporeModelWrap::TransformNvidiaGPUModelContext(uint32_t device_id,
                                                                                      const DeviceInfo &device_info) {
  auto context_info = std::make_shared<GPUDeviceInfo>();
  context_info->SetDeviceID(device_id);

  for (auto &item : device_info) {
    const auto &key = item.first;
    const auto &value = item.second;
    if (key == "precision_mode") {
      context_info->SetPrecisionMode(value);
      context_info->SetEnableFP16(value == "fp16");
    }
  }
  return context_info;
}

std::shared_ptr<DeviceInfoContext> MindSporeModelWrap::TransformCPUModelContext(const DeviceInfo &device_info) {
  auto context_info = std::make_shared<CPUDeviceInfo>();
  for (auto &item : device_info) {
    const auto &key = item.first;
    const auto &value = item.second;
    if (key == "precision_mode") {
      context_info->SetEnableFP16(value == "fp16");
    }
  }
  return context_info;
}

std::string MindSporeModelWrap::DeviceTypeToString(serving::DeviceType device_type) {
  switch (device_type) {
    case kDeviceTypeGpu:
      return "gpu";
    case kDeviceTypeCpu:
      return "cpu";
    case kDeviceTypeAscend:
      return "ascend";
    case kDeviceTypeNotSpecified:
      return "not_specified";
  }
  return "";
}

DeviceInfo MindSporeModelWrap::GetDeviceInfo(const std::vector<DeviceInfo> &device_list,
                                             serving::DeviceType device_type) {
  DeviceInfo device_info;
  for (auto &item : device_list) {
    if (item.at("device_type") == DeviceTypeToString(device_type)) {
      device_info = item;
      break;
    }
  }
  return device_info;
}

std::shared_ptr<Context> MindSporeModelWrap::TransformModelContext(serving::DeviceType device_type, uint32_t device_id,
                                                                   const ModelContext &model_context,
                                                                   bool enable_lite) {
  auto context = std::make_shared<mindspore::Context>();
  if (model_context.thread_num != -1) {
    context->SetThreadNum(model_context.thread_num);
  }
  if (model_context.enable_parallel != -1) {
    context->SetEnableParallel(model_context.enable_parallel);
  }
  if (!model_context.thread_affinity_core_list.empty()) {
    context->SetThreadAffinity(model_context.thread_affinity_core_list);
  }

  std::shared_ptr<mindspore::DeviceInfoContext> context_info = nullptr;

  auto device_info = GetDeviceInfo(model_context.device_list, device_type);
  if (device_type == kDeviceTypeAscend) {
    context_info = TransformAscendModelContext(device_id, device_info);
  } else if (device_type == kDeviceTypeCpu) {
    context_info = TransformCPUModelContext(device_info);
  } else if (device_type == kDeviceTypeGpu) {
    context_info = TransformNvidiaGPUModelContext(device_id, device_info);
  }
  if (context_info != nullptr) {
    context->MutableDeviceInfo().push_back(context_info);
  }

  if (enable_lite && device_type != kDeviceTypeCpu) {
    auto cpu_device_info = GetDeviceInfo(model_context.device_list, kDeviceTypeCpu);
    context->MutableDeviceInfo().push_back(TransformCPUModelContext(cpu_device_info));
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

Status MindSporeModelWrap::CalculateBatchSize(ApiModelInfo *api_model_info) {
  auto &input_infos = api_model_info->input_tensor_infos;
  auto &output_infos = api_model_info->output_tensor_infos;
  if (!common_model_info_.with_batch_dim) {
    common_model_info_.batch_size = 1;
    for (auto &input : input_infos) {
      input.is_no_batch_dim = true;
    }
    for (auto &output : output_infos) {
      output.is_no_batch_dim = true;
    }
    return SUCCESS;
  }
  const auto &list = common_model_info_.without_batch_dim_inputs;
  uint32_t cur_batch_size = 0;
  for (size_t i = 0; i < input_infos.size(); i++) {
    auto &input = input_infos[i];
    if (std::find(list.begin(), list.end(), i) != list.end()) {
      input.is_no_batch_dim = true;
      continue;
    }
    if (input.shape.empty()) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "The shape of model input " << i << " cannot be empty, "
                                            << "when with_batch_dim is true and without_batch_dim_inputs is " << list;
    }
    if (cur_batch_size == 0) {
      cur_batch_size = input.shape[0];
      continue;
    }
    if (input.shape[0] != cur_batch_size) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "The shape " << input.shape << " of model input " << i
                                            << " does not match current batch size " << cur_batch_size;
    }
  }
  for (size_t i = 0; i < output_infos.size(); i++) {
    auto &output = output_infos[i];
    if (output.shape.empty()) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "The shape of model output " << i << " cannot be empty";
    }
    if (cur_batch_size == 0) {
      cur_batch_size = output.shape[0];
      continue;
    }
    if (output.shape[0] != cur_batch_size) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "The shape " << output.shape << " of model output " << i
                                            << " does not match current batch size " << cur_batch_size;
    }
  }
  if (cur_batch_size == 0) {
    cur_batch_size = 1;
  }
  common_model_info_.batch_size = cur_batch_size;
  return SUCCESS;
}

Status MindSporeModelWrap::UnloadModel() {
  for (auto iter : models_) iter.model = nullptr;
  return SUCCESS;
}

Status MindSporeModelWrap::ExecuteModel(const RequestBase &request, serving::ReplyBase *reply, bool return_result,
                                        uint64_t subgraph) {
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
  return ExecuteModelCommon(request.size(), func_in, func_out, return_result, subgraph);
}

Status MindSporeModelWrap::ExecuteModel(const std::vector<TensorBasePtr> &request, std::vector<TensorBasePtr> *reply,
                                        bool return_result, uint64_t subgraph) {
  if (subgraph >= models_.size()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Inputs subgraph label error, subgraph label is " << subgraph
                                          << ", total graph number is " << models_.size();
  }
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
  return ExecuteModelCommon(request.size(), func_in, func_out, return_result, subgraph);
}

Status MindSporeModelWrap::ExecuteModelCommon(size_t request_size, const FuncMakeInBuffer &in_func,
                                              const FuncMakeOutTensor &out_func, bool return_result,
                                              uint64_t subgraph) {
  if (models_[subgraph].model == nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Model is not loaded";
  }
  auto &model_info = models_[subgraph];
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
  mindspore::Status status;
  if (SupportReuseDevice()) {
    status = model->Predict(inputs, &outputs);
  } else {  // vm backend
    std::unique_lock<std::mutex> lock(infer_mutex_);
    status = model->Predict(inputs, &outputs);
  }
  if (!status.IsOk()) {
    MSI_LOG_ERROR << "Predict failed: " << status.ToString();
    return Status(FAILED, "Predict Failed");
  }
  if (outputs.size() != output_names.size()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Outputs size not match, predict outputs size " << outputs.size()
                                          << ", model outputs size " << output_names.size();
  }
  if (return_result) {
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
  }
  return SUCCESS;
}

std::vector<serving::TensorInfo> MindSporeModelWrap::GetInputInfos(uint64_t subgraph) const {
  return models_[subgraph].input_tensor_infos;
}

std::vector<serving::TensorInfo> MindSporeModelWrap::GetOutputInfos(uint64_t subgraph) const {
  return models_[subgraph].output_tensor_infos;
}

ssize_t MindSporeModelWrap::GetBatchSize(uint64_t subgraph) const { return common_model_info_.batch_size; }

uint64_t MindSporeModelWrap::GetSubGraphNum() const { return models_.size(); }

bool MindSporeModelWrap::SupportReuseDevice() const {
  auto is_device_910 = mindspore::Model::CheckModelSupport(mindspore::kAscend910, mindspore::kMindIR);
  return !is_device_910;
}

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
    case kMindIR_Opt:
      ms_model_type = mindspore::kMindIR_Opt;
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
    case kDeviceTypeAscend:
      ms_device_type = mindspore::DeviceType::kAscend;
      break;
    case kDeviceTypeGpu:
      ms_device_type = mindspore::DeviceType::kGPU;
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
