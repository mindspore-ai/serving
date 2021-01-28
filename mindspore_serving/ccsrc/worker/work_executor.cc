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

#include "worker/work_executor.h"
#include <utility>
#include <cstring>
#include <thread>
#include <chrono>
#include "worker/preprocess.h"
#include "worker/postprocess.h"
#include "mindspore_serving/ccsrc/common/tensor.h"
#include "common/buffer_tensor.h"

namespace mindspore {
namespace serving {

#define NO_THREAD_POOL

WorkExecutor::WorkExecutor(std::shared_ptr<TaskQueue> py_preprocess_task_queue,
                           std::shared_ptr<TaskQueue> py_postprocess_task_queue,
                           std::shared_ptr<TaskQueue> cpp_preprocess_task_queue,
                           std::shared_ptr<TaskQueue> cpp_postprocess_task_queue)
    : py_preprocess_task_queue_(py_preprocess_task_queue),
      py_postprocess_task_queue_(py_postprocess_task_queue),
      cpp_preprocess_task_queue_(cpp_preprocess_task_queue),
      cpp_postprocess_task_queue_(cpp_postprocess_task_queue) {
  static std::atomic_uint64_t g_worker_id;
  worker_id_ = ++g_worker_id;
}

WorkExecutor::~WorkExecutor() = default;

Status WorkExecutor::CheckSevableSignature() {
  Status status;
  const auto &input_infos = input_infos_;
  if (servable_declare_.methods.empty()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "There is no method registered for servable";
  }
  const auto &common_meta = servable_declare_.servable_meta.common_meta;
  if (input_infos.size() != common_meta.inputs_count) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "The inputs count " << common_meta.inputs_count << " registered in method "
                                          << "not equal to the count " << input_infos.size() << " defined in servable";
  }
  const auto &output_infos = output_infos_;
  if (output_infos.size() != common_meta.outputs_count) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "The outputs count " << common_meta.outputs_count << " registered in method "
           << "not equal to the count " << output_infos.size() << " defined in servable";
  }
  MSI_LOG_INFO << "Model input infos: count " << input_infos.size();
  for (auto &item : input_infos) {
    MSI_LOG_INFO << item.shape << ", " << item.data_type << ", " << item.size;
  }
  MSI_LOG_INFO << "Model output infos: count " << output_infos.size();
  for (auto &item : output_infos) {
    MSI_LOG_INFO << item.shape << ", " << item.data_type << ", " << item.size;
  }
  if (common_meta.with_batch_dim) {
    if (model_batch_size_ == 0) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Servable batch size cannot be " << model_batch_size_;
    }
    for (size_t i = 0; i < input_infos.size(); i++) {
      if (IsNoBatchDimInput(i)) {
        continue;
      }
      const auto &item = input_infos[i];
      if (item.shape.empty() || static_cast<uint32_t>(item.shape[0]) != model_batch_size_) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "Servable batch size " << model_batch_size_ << " not match model input shape " << item.shape;
      }
    }
    for (const auto &item : output_infos) {
      if (item.shape.empty() || static_cast<uint32_t>(item.shape[0]) != model_batch_size_) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "Servable batch size " << model_batch_size_ << " not match model output shape " << item.shape;
      }
    }
  }
  return SUCCESS;
}

Status WorkExecutor::Init(const ServableSignature &servable_declare, const std::shared_ptr<ServableBase> &servable) {
  Status status;
  if (init_flag_) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Worker service has been initialized";
  }
  if (servable == nullptr) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "System error: Servable cannot be nullptr";
  }
  servable_declare_ = servable_declare;
  servable_ = servable;
  input_infos_ = servable_->GetInputInfos();
  output_infos_ = servable_->GetOutputInfos();
  if (servable_declare_.servable_meta.common_meta.with_batch_dim) {
    model_batch_size_ = servable_->GetBatchSize();
  } else {
    model_batch_size_ = 1;
  }

  status = servable_declare_.Check();
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Check servable declare failed";
    return status;
  }
  status = CheckSevableSignature();
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Check servable definition failed";
    return status;
  }
  InitInputTensors();
  // init python preprocess and postprocess
  InitPrePostprocess();
  // init predict fun
  auto predict_fun = [this](const std::vector<Instance> &inputs) { this->PredictHandle(inputs); };

  predict_thread_.Start(predict_fun, model_batch_size_);

  init_flag_ = true;
  return SUCCESS;
}

void WorkExecutor::InitPrePostprocess() {
  for (auto &method : servable_declare_.methods) {
    if (!method.preprocess_name.empty()) {
      auto preprocess = PreprocessStorage::Instance().GetPreprocess(method.preprocess_name);
      if (preprocess && preprocess->IsPythonPreprocess()) {
        python_preprocess_names_.emplace(method.preprocess_name);
      }
    }
    if (!method.postprocess_name.empty()) {
      auto postprocess = PostprocessStorage::Instance().GetPostprocess(method.postprocess_name);
      if (postprocess && postprocess->IsPythonPostprocess()) {
        python_postprocess_names_.emplace(method.postprocess_name);
      }
    }
  }
  // init cpp preprocess and postprocess
  auto preprocess_callback = [this](const std::vector<Instance> &inputs, const std::vector<ResultInstance> &outputs) {
    if (inputs.empty() || inputs.size() != outputs.size()) {
      MSI_LOG_ERROR << "Invalid inputs size " << inputs.size() << ", result size " << outputs.size();
      return;
    }
    std::vector<Instance> outputs_real;
    for (size_t i = 0; i < inputs.size(); i++) {
      auto &input = inputs[i];
      auto &output = outputs[i];
      if (output.error_msg != SUCCESS) {
        ReplyError(input, output.error_msg);
        continue;
      }
      auto output_result = CreateResultInstance(input, output, kPredictPhaseTag_Preproces);
      outputs_real.push_back(output_result);
    }
    OnRecievePredictInputs(outputs_real);
  };
  auto postprocess_callback = [this](const std::vector<Instance> &inputs, const std::vector<ResultInstance> &outputs) {
    if (inputs.empty() || inputs.size() != outputs.size()) {
      MSI_LOG_ERROR << "Invalid inputs size " << inputs.size() << ", result size " << outputs.size();
      return;
    }
    std::vector<Instance> outputs_real;
    for (size_t i = 0; i < inputs.size(); i++) {
      auto &input = inputs[i];
      auto &output = outputs[i];
      if (output.error_msg != SUCCESS) {
        ReplyError(input, output.error_msg);
        continue;
      }
      auto output_result = CreateResultInstance(input, output, kPredictPhaseTag_Postprocess);
      outputs_real.push_back(output_result);
    }
    ReplyRequest(outputs_real);
  };
  py_preprocess_task_queue_->SetWorkerCallback(GetWorkerId(), preprocess_callback);
  cpp_preprocess_task_queue_->SetWorkerCallback(GetWorkerId(), preprocess_callback);
  py_postprocess_task_queue_->SetWorkerCallback(GetWorkerId(), postprocess_callback);
  cpp_postprocess_task_queue_->SetWorkerCallback(GetWorkerId(), postprocess_callback);
}

void WorkExecutor::InitInputTensors() {
  inference_inputs_.clear();
  for (size_t i = 0; i < input_infos_.size(); i++) {
    auto &input_info = input_infos_[i];
    auto tensor = std::make_shared<Tensor>();
    tensor->set_data_type(input_info.data_type);
    tensor->set_shape(input_info.shape);
    tensor->resize_data(input_info.size);
    inference_inputs_.push_back(tensor);
  }
}

std::vector<std::future<void>> WorkExecutor::Work(const RequestSpec &request_spec,
                                                  const std::vector<InstanceData> &instances_data,
                                                  WorkCallBack on_process_done) {
  if (!init_flag_) {
    MSI_LOG_EXCEPTION << "Worker service has not been initialized";
  }
  std::vector<std::future<void>> future_list(instances_data.size());
  std::vector<Instance> new_inputs(instances_data.size());
  auto user_id = WorkExecutor::GetNextUserId();
  auto user_context = std::make_shared<WorkerUserContext>();
  user_context->worker_call_back = on_process_done;
  user_context->request_spec = request_spec;

  MethodSignature &method_def = user_context->method_def;
  Status status;
  if (!servable_declare_.GetMethodDeclare(request_spec.method_name, &method_def)) {
    MSI_LOG_EXCEPTION << "Not support method " << request_spec.method_name;
  }

  for (size_t i = 0; i < new_inputs.size(); i++) {
    new_inputs[i].input_data = instances_data[i];
    auto &context = new_inputs[i].context;
    context.user_id = user_id;
    context.instance_index = i;
    context.user_context = user_context;
    context.promise = std::make_shared<std::promise<void>>();
    future_list[i] = context.promise->get_future();
  }
  if (!method_def.preprocess_name.empty()) {
    OnRecievePreprocessInputs(new_inputs);
  } else {
    OnRecievePredictInputs(new_inputs);
  }
  return future_list;
}

void WorkExecutor::OnRecievePreprocessInputs(const std::vector<Instance> &inputs) {
  if (inputs.empty()) {
    MSI_LOG_EXCEPTION << "Inputs cannot be empty";
  }
  const MethodSignature &method_def = inputs[0].context.user_context->method_def;
  auto real_inputs = CreateInputInstance(inputs, kPredictPhaseTag_Preproces);
  if (python_preprocess_names_.count(method_def.preprocess_name) > 0) {
    py_preprocess_task_queue_->PushTask(method_def.preprocess_name, GetWorkerId(), real_inputs);
  } else {
    cpp_preprocess_task_queue_->PushTask(method_def.preprocess_name, GetWorkerId(), real_inputs);
  }
}

void WorkExecutor::OnRecievePostprocessInputs(const Instance &input) {
  const MethodSignature &method_def = input.context.user_context->method_def;
  auto real_input = CreateInputInstance(input, kPredictPhaseTag_Postprocess);
  if (python_postprocess_names_.count(method_def.postprocess_name) > 0) {
    py_postprocess_task_queue_->PushTask(method_def.postprocess_name, GetWorkerId(), {real_input});
  } else {
    cpp_postprocess_task_queue_->PushTask(method_def.postprocess_name, GetWorkerId(), {real_input});
  }
}

void WorkExecutor::OnRecievePredictInputs(const std::vector<Instance> &inputs) {
  // create input for predict, and check
  auto real_inputs = CreateInputInstance(inputs, kPredictPhaseTag_Predict);
  std::vector<Instance> valid_inputs;
  for (auto &item : real_inputs) {
    auto status = CheckPredictInput(item);
    if (status != SUCCESS) {
      ReplyError(item, status);
      continue;
    }
    valid_inputs.push_back(item);
  }
  predict_thread_.PushPredictTask(valid_inputs);
}

bool WorkExecutor::ReplyRequest(const std::vector<Instance> &outputs) {
  for (auto &item : outputs) {
    ReplyRequest(item);
  }
  return true;
}

bool WorkExecutor::ReplyRequest(const Instance &outputs) {
  Status status;
  Instance trans_outputs = CreateInputInstance(outputs, kPredictPhaseTag_Output);
  Instance real_outputs;
  real_outputs.data = trans_outputs.data;
  real_outputs.context = trans_outputs.context;
  outputs.context.user_context->worker_call_back(real_outputs, Status(SUCCESS));
  outputs.context.promise->set_value();
  return true;
}

bool WorkExecutor::ReplyError(const std::vector<Instance> &context, const Status &error_msg) {
  for (auto &item : context) {
    ReplyError(item, error_msg);
  }
  return true;
}

bool WorkExecutor::ReplyError(const Instance &instance, const Status &error_msg) {
  Instance error_instance;
  error_instance.context = instance.context;
  instance.context.user_context->worker_call_back(error_instance, error_msg);
  instance.context.promise->set_value();
  return true;
}

void WorkExecutor::PredictHandle(const std::vector<Instance> &inputs) {
  Status status;
  try {
    std::vector<Instance> outputs;
    status = Predict(inputs, &outputs);
    if (status != SUCCESS) {
      this->ReplyError(inputs, status);
      return;
    }
    for (auto &output : outputs) {
      MethodSignature &method_def = output.context.user_context->method_def;
      if (!method_def.postprocess_name.empty()) {
        OnRecievePostprocessInputs(output);
      } else {
        ReplyRequest(output);
      }
    }
    return;
  } catch (const std::bad_alloc &ex) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Serving Error: malloc memory failed";
  } catch (const std::runtime_error &ex) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Serving Error: runtime error occurred: " << ex.what();
  } catch (const std::exception &ex) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Serving Error: exception occurred: " << ex.what();
  } catch (...) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Serving Error: exception occurred";
  }
  ReplyError(inputs, status);
}

Status WorkExecutor::PrePredict(const std::vector<Instance> &inputs) {
  auto input_batch_size = static_cast<uint32_t>(inputs.size());
  uint32_t model_batch_size = model_batch_size_;
  if (input_batch_size == 0 || input_batch_size > model_batch_size) {
    MSI_LOG_ERROR << "Input batch size " << input_batch_size << " invalid, model batch size " << model_batch_size;
    return SYSTEM_ERROR;
  }
  for (size_t i = 0; i < inference_inputs_.size(); i++) {
    auto &tensor = inference_inputs_[i];
    auto data_size = tensor->data_size();
    auto dst_buffer = reinterpret_cast<uint8_t *>(tensor->mutable_data());
    if (IsNoBatchDimInput(i)) {
      memcpy_s(dst_buffer, data_size, inputs[0].data[i]->data(), data_size);
      continue;
    }
    auto item_size = static_cast<size_t>(data_size / model_batch_size);
    for (uint32_t k = 0; k < input_batch_size; k++) {
      if (i >= inputs[k].data.size()) {
        return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << " Batch index " << k << " does not have input " << i;
      }
      if (item_size != inputs[k].data[i]->data_size()) {
        return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
               << " Batch index " << k << " input data size " << inputs[k].data[i]->data_size()
               << "does not match size " << item_size << " defined in model";
      }
      memcpy_s(dst_buffer + k * item_size, data_size - k * item_size, inputs[k].data[i]->data(), item_size);
    }
    for (uint32_t k = input_batch_size; k < model_batch_size; k++) {
      memcpy_s(dst_buffer + k * item_size, data_size - k * item_size, inputs[0].data[i]->data(), item_size);
    }
  }
  return SUCCESS;
}

Status WorkExecutor::PostPredict(const std::vector<Instance> &inputs, const std::vector<TensorBasePtr> &predict_result,
                                 std::vector<Instance> *outputs) {
  MSI_EXCEPTION_IF_NULL(outputs);
  auto input_batch_size = static_cast<uint32_t>(inputs.size());
  uint32_t model_batch_size = model_batch_size_;
  if (input_batch_size == 0 || input_batch_size > model_batch_size) {
    MSI_LOG_ERROR << "Input batch size " << input_batch_size << " invalid, model batch size " << model_batch_size;
    return SYSTEM_ERROR;
  }
  std::vector<ResultInstance> results_data(input_batch_size);
  for (auto &item : predict_result) {
    size_t item_size = item->data_size() / model_batch_size;
    if (item_size == 0) {
      MSI_LOG_EXCEPTION << "Output result data size cannot be 0";
    }
    auto shape = item->shape();
    if (servable_declare_.servable_meta.common_meta.with_batch_dim) {
      if (shape.empty() || shape[0] != model_batch_size) {
        MSI_LOG_EXCEPTION << "Output shape " << shape << " not match batch size " << model_batch_size;
      }
      shape.erase(shape.begin());
    }
    auto src_buffer = const_cast<uint8_t *>(item->data());
    for (uint32_t k = 0; k < input_batch_size; k++) {
      auto tensor = std::make_shared<BufferTensorWithOwner>(item, item->data_type(), shape, src_buffer + item_size * k,
                                                            item_size, true);
      results_data[k].data.push_back(tensor);
    }
  }
  *outputs = CreateResultInstance(inputs, results_data, kPredictPhaseTag_Predict);
  return SUCCESS;
}

Status WorkExecutor::Predict(const std::vector<Instance> &inputs, std::vector<Instance> *outputs) {
  MSI_EXCEPTION_IF_NULL(outputs);
  Status status;
  std::vector<TensorBasePtr> predict_outputs;
  status = PrePredict(inputs);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Call Pre Predict failed, model info " << servable_declare_.servable_meta.Repr();
    return status;
  }
  status = servable_->Predict(inference_inputs_, &predict_outputs);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Predict failed, model info " << servable_declare_.servable_meta.Repr();
    return status;
  }
  status = PostPredict(inputs, predict_outputs, outputs);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Call Post Predict failed, model info " << servable_declare_.servable_meta.Repr();
    return status;
  }
  return SUCCESS;
}

bool WorkExecutor::IsNoBatchDimInput(int input_index) const {
  auto without_batch_dim_inputs = servable_declare_.servable_meta.common_meta.without_batch_dim_inputs;
  bool no_batch_dim = true;
  if (servable_declare_.servable_meta.common_meta.with_batch_dim) {
    no_batch_dim = std::find(without_batch_dim_inputs.begin(), without_batch_dim_inputs.end(), input_index) !=
                   without_batch_dim_inputs.end();
  }
  return no_batch_dim;
}

Status WorkExecutor::CheckPredictInput(const Instance &instance) {
  const auto &inputs_info = input_infos_;
  if (instance.data.size() < inputs_info.size()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Given model inputs size " << instance.data.size()
                                                  << " less than model inputs size " << inputs_info.size();
  }
  for (size_t i = 0; i < instance.data.size(); i++) {
    auto input_data = instance.data[i];
    if (IsNoBatchDimInput(i)) {
      if (static_cast<size_t>(inputs_info[i].size) != input_data->data_size()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
               << "Given model input " << i << " size " << input_data->data_size() << " not match the size "
               << inputs_info[i].size << " defined in model";
      }
    } else if (static_cast<size_t>(inputs_info[i].size / model_batch_size_) != input_data->data_size()) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
             << "Given model input " << i << " size " << input_data->data_size() << " not match the size "
             << inputs_info[i].size / model_batch_size_ << " defined in model";
    }
    if (inputs_info[i].data_type != input_data->data_type()) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
             << "Given model input " << i << " data type " << input_data->data_type() << " not match the data type "
             << inputs_info[i].data_type << " defined in model";
    }
  }
  return SUCCESS;
}

std::vector<Instance> WorkExecutor::CreateInputInstance(const std::vector<Instance> &instances, PredictPhaseTag phase) {
  std::vector<Instance> ret(instances.size());
  for (size_t i = 0; i < instances.size(); i++) {
    ret[i] = CreateInputInstance(instances[i], phase);
  }
  return ret;
}

Instance WorkExecutor::CreateInputInstance(const Instance &instance, PredictPhaseTag phase) {
  Instance result = instance;
  result.data.clear();

  MethodSignature &method_def = instance.context.user_context->method_def;
  std::vector<std::pair<PredictPhaseTag, uint64_t>> *inputs = nullptr;
  switch (phase) {
    case kPredictPhaseTag_Preproces:
      inputs = &method_def.preprocess_inputs;
      break;
    case kPredictPhaseTag_Predict:
      inputs = &method_def.servable_inputs;
      break;
    case kPredictPhaseTag_Postprocess:
      inputs = &method_def.postprocess_inputs;
      break;
    case kPredictPhaseTag_Output:
      inputs = &method_def.returns;
      break;
    default: {
      MSI_LOG_EXCEPTION << "Invalid phase tag " << phase;
    }
  }
  for (auto &item : *inputs) {
    const InstanceData *data = nullptr;
    switch (item.first) {
      case kPredictPhaseTag_Input:
        data = &instance.input_data;
        break;
      case kPredictPhaseTag_Preproces:
        data = &instance.preprocess_data;
        break;
      case kPredictPhaseTag_Predict:
        data = &instance.predict_data;
        break;
      case kPredictPhaseTag_Postprocess:
        data = &instance.postprocess_data;
        break;
      default: {
        MSI_LOG_EXCEPTION << "Invalid input phase tag " << item.first;
      }
    }
    if (data->size() <= item.second) {
      MSI_LOG_EXCEPTION << "Invalid output index " << item.second << ", output count " << data->size()
                        << ", input phase tag " << item.first << ", phase " << phase << ", method "
                        << method_def.method_name;
    }
    result.data.push_back(data->at(item.second));
  }
  return result;
}

std::vector<Instance> WorkExecutor::CreateResultInstance(const std::vector<Instance> &inputs,
                                                         const std::vector<ResultInstance> &results,
                                                         PredictPhaseTag phase) {
  std::vector<Instance> ret(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    ret[i] = CreateResultInstance(inputs[i], results[i], phase);
  }
  return ret;
}

Instance WorkExecutor::CreateResultInstance(const Instance &input, const ResultInstance &result,
                                            PredictPhaseTag phase) {
  Instance instance = input;
  instance.data.clear();
  switch (phase) {
    case kPredictPhaseTag_Input:
      instance.input_data = result.data;
      break;
    case kPredictPhaseTag_Preproces:
      instance.preprocess_data = result.data;
      break;
    case kPredictPhaseTag_Predict:
      instance.predict_data = result.data;
      break;
    case kPredictPhaseTag_Postprocess:
      instance.postprocess_data = result.data;
      break;
    default: {
      MSI_LOG_EXCEPTION << "Invalid phase tag " << phase;
    }
  }
  return instance;
}

uint64_t WorkExecutor::GetNextUserId() {
  static std::atomic<uint64_t> user_id;
  return ++user_id;
}

uint32_t WorkExecutor::GetWorkerId() const { return worker_id_; }

}  // namespace serving
}  // namespace mindspore
