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
#include <map>
#include "worker/preprocess.h"
#include "worker/postprocess.h"
#include "worker/pipeline.h"
#include "mindspore_serving/ccsrc/common/tensor.h"
#include "common/buffer_tensor.h"
#include "worker/worker.h"

namespace mindspore {
namespace serving {

WorkExecutor::WorkExecutor(std::shared_ptr<TaskQueue> py_preprocess_task_queue,
                           std::shared_ptr<TaskQueue> py_postprocess_task_queue,
                           std::shared_ptr<TaskQueue> cpp_preprocess_task_queue,
                           std::shared_ptr<TaskQueue> cpp_postprocess_task_queue,
                           std::shared_ptr<TaskQueue> py_pipeline_task_queue)
    : py_preprocess_task_queue_(std::move(py_preprocess_task_queue)),
      py_postprocess_task_queue_(std::move(py_postprocess_task_queue)),
      cpp_preprocess_task_queue_(std::move(cpp_preprocess_task_queue)),
      cpp_postprocess_task_queue_(std::move(cpp_postprocess_task_queue)),
      py_pipeline_task_queue_(std::move(py_pipeline_task_queue)) {
  static std::atomic_uint64_t g_worker_id;
  worker_id_ = ++g_worker_id;
}

WorkExecutor::~WorkExecutor() {
  predict_thread_.Stop();
  ClearInstances(Status(WORKER_UNAVAILABLE, "Servable stopped"));
}

Status WorkExecutor::CheckServableSignature(uint64_t subgraph) {
  Status status;
  const auto &input_infos = input_infos_[subgraph];
  if (servable_declare_.methods.empty()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "There is no method registered for servable";
  }
  const auto &common_meta = servable_declare_.servable_meta.common_meta;
  if (input_infos.size() != common_meta.inputs_count) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "The inputs count " << common_meta.inputs_count << " registered in method "
                                          << "not equal to the count " << input_infos.size() << " defined in servable";
  }
  if (output_infos_[subgraph].size() != common_meta.outputs_count) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "The outputs count " << common_meta.outputs_count << " registered in method "
           << "not equal to the count " << output_infos_[subgraph].size() << " defined in servable";
  }
  MSI_LOG_INFO << "Model input infos: count " << input_infos.size();
  for (auto &item : input_infos) {
    MSI_LOG_INFO << item.shape << ", " << item.data_type << ", " << item.size;
  }
  MSI_LOG_INFO << "Model output infos: count " << output_infos_[subgraph].size();
  for (auto &item : output_infos_[subgraph]) {
    MSI_LOG_INFO << item.tensor_info.shape << ", " << item.tensor_info.data_type << ", " << item.tensor_info.size;
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
    for (auto &item : output_infos_[subgraph]) {
      auto &tensor_info = item.tensor_info;
      if (tensor_info.shape.empty() || static_cast<uint32_t>(tensor_info.shape[0]) != model_batch_size_) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "Servable batch size " << model_batch_size_ << " not match model output shape " << tensor_info.shape;
      }
      item.shape_one_batch = tensor_info.shape;
      item.shape_one_batch.erase(item.shape_one_batch.begin());
      item.size_one_batch = tensor_info.size / model_batch_size_;
    }
  } else {
    for (auto &item : output_infos_[subgraph]) {
      auto &tensor_info = item.tensor_info;
      item.shape_one_batch = tensor_info.shape;
      item.size_one_batch = tensor_info.size;
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
  uint64_t graph_num = servable_->GetGraphNum();
  for (uint64_t i = 0; i < graph_num; i++) {
    input_infos_[i] = servable_->GetInputInfos(i);
    auto output_infos = servable_->GetOutputInfos(i);
    for (auto &item : output_infos) {
      TensorInfoWithBatch info;
      info.tensor_info = item;
      output_infos_[i].push_back(info);
    }
    if (servable_declare_.servable_meta.common_meta.with_batch_dim) {
      model_batch_size_ = servable_->GetBatchSize(i);
    } else {
      model_batch_size_ = 1;
    }
    status = CheckServableSignature(i);
    if (status != SUCCESS) {
      MSI_LOG_ERROR << "Check servable definition failed";
      return status;
    }
    InitInputTensors(i);
  }

  status = servable_declare_.Check(graph_num);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Check servable declare failed";
    return status;
  }
  // init python preprocess and postprocess
  InitPrePostprocess();
  InitPipeline();
  // init predict fun
  auto predict_fun = [this](const std::vector<InstancePtr> &inputs) { this->PredictHandle(inputs); };

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
  auto preprocess_callback = [this](const std::vector<InstancePtr> &instances,
                                    const std::vector<ResultInstance> &outputs) {
    if (instances.empty() || instances.size() != outputs.size()) {
      MSI_LOG_ERROR << "Invalid inputs size " << instances.size() << ", result size " << outputs.size();
      return;
    }
    std::vector<InstancePtr> outputs_real;
    for (size_t i = 0; i < instances.size(); i++) {
      auto &instance = instances[i];
      auto &output = outputs[i];
      if (output.error_msg != SUCCESS) {
        ReplyError(instance, output.error_msg);
        continue;
      }
      CreateResultInstance(instance, output, kPredictPhaseTag_Preproces);
      outputs_real.push_back(instance);
    }
    OnReceivePredictInputs(outputs_real);
  };
  auto postprocess_callback = [this](const std::vector<InstancePtr> &instances,
                                     const std::vector<ResultInstance> &outputs) {
    if (instances.empty() || instances.size() != outputs.size()) {
      MSI_LOG_ERROR << "Invalid inputs size " << instances.size() << ", result size " << outputs.size();
      return;
    }
    std::vector<InstancePtr> outputs_real;
    for (size_t i = 0; i < instances.size(); i++) {
      auto &instance = instances[i];
      auto &output = outputs[i];
      if (output.error_msg != SUCCESS) {
        ReplyError(instance, output.error_msg);
        continue;
      }
      CreateResultInstance(instance, output, kPredictPhaseTag_Postprocess);
      outputs_real.push_back(instance);
    }
    ReplyRequest(outputs_real);
  };
  py_preprocess_task_queue_->SetWorkerCallback(GetWorkerId(), preprocess_callback);
  cpp_preprocess_task_queue_->SetWorkerCallback(GetWorkerId(), preprocess_callback);
  py_postprocess_task_queue_->SetWorkerCallback(GetWorkerId(), postprocess_callback);
  cpp_postprocess_task_queue_->SetWorkerCallback(GetWorkerId(), postprocess_callback);
}
void WorkExecutor::InitPipeline() {
  std::vector<PipelineSignature> pipelines;
  PipelineStorage::Instance().GetPipelineDef(&pipelines);
  for (auto &method : pipelines) {
    auto name = servable_declare_.servable_meta.common_meta.servable_name + "." + method.pipeline_name;
    python_pipeline_names_.emplace(name);
  }
  auto pipeline_callback = [this](const std::vector<InstancePtr> &instances,
                                  const std::vector<ResultInstance> &outputs) {
    if (instances.empty() || instances.size() != outputs.size()) {
      MSI_LOG_ERROR << "Invalid inputs size " << instances.size() << ", result size " << outputs.size();
      return;
    }
    std::vector<InstancePtr> outputs_real;
    for (size_t i = 0; i < instances.size(); i++) {
      auto &instance = instances[i];
      auto &output = outputs[i];
      if (output.error_msg != SUCCESS) {
        ReplyError(instance, output.error_msg);
        continue;
      }
      CreateResultInstance(instance, output, kPredictPhaseTag_Pipeline);
      outputs_real.push_back(instance);
    }
    ReplyRequest(outputs_real);
  };
  py_pipeline_task_queue_->SetWorkerCallback(GetWorkerId(), pipeline_callback);
}
void WorkExecutor::InitInputTensors(uint64_t subgraph) {
  inference_inputs_[subgraph].clear();
  for (auto &input_info : input_infos_[subgraph]) {
    auto tensor = std::make_shared<Tensor>();
    tensor->set_data_type(input_info.data_type);
    tensor->set_shape(input_info.shape);
    tensor->resize_data(input_info.size);
    inference_inputs_[subgraph].push_back(tensor);
  }
}

Status WorkExecutor::Work(const RequestSpec &request_spec, const std::vector<InstanceData> &instances_data,
                          WorkCallBack on_process_done) {
  if (!init_flag_) {
    MSI_LOG_EXCEPTION << "Worker service has not been initialized";
  }
  auto user_id = WorkExecutor::GetNextUserId();
  InferSession infer_session;
  infer_session.call_back = std::move(on_process_done);

  auto user_context = std::make_shared<WorkerUserContext>();
  user_context->request_spec = request_spec;

  MethodSignature &method_def = user_context->method_def;
  Status status;
  if (!servable_declare_.GetMethodDeclare(request_spec.method_name, &method_def)) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Not support method " << request_spec.method_name;
  }

  std::vector<InstancePtr> instances;
  for (size_t i = 0; i < instances_data.size(); i++) {
    if (method_def.inputs.size() != instances_data[i].size()) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Input count " << instances_data[i].size() << " does not equal to input count "
             << method_def.inputs.size() << " defined in method " << request_spec.method_name;
    }

    auto instance = std::make_shared<Instance>();
    instances.push_back(instance);

    instance->input_data = instances_data[i];
    auto &context = instance->context;
    context.user_id = user_id;
    context.instance_index = i;
    context.user_context = user_context;
  }
  infer_session.instances = instances;
  {
    std::unique_lock<std::mutex> lock(infer_session_map_mutex_);
    infer_session_map_[user_id] = infer_session;
  }

  if (!method_def.preprocess_name.empty()) {
    OnReceivePreprocessInputs(instances);
  } else {
    OnReceivePredictInputs(instances);
  }
  return SUCCESS;
}
Status WorkExecutor::Pipe(const RequestSpec &request_spec, const std::vector<InstanceData> &instances_data,
                          const PipelineSignature &method_signature, WorkCallBack on_process_done) {
  if (!init_flag_) {
    MSI_LOG_EXCEPTION << "Worker service has not been initialized";
  }
  auto user_id = WorkExecutor::GetNextUserId();
  InferSession infer_session;
  infer_session.call_back = std::move(on_process_done);

  auto user_context = std::make_shared<WorkerUserContext>();
  user_context->request_spec = request_spec;

  MethodSignature &method_def = user_context->method_def;
  method_def.pipeline_name = request_spec.servable_name + "." + method_signature.pipeline_name;
  for (size_t i = 0; i < method_signature.inputs.size(); i++) {
    method_def.pipeline_inputs.push_back(std::make_pair(kPredictPhaseTag_Pipeline, i));
  }
  for (size_t i = 0; i < method_signature.outputs.size(); i++) {
    method_def.returns.push_back(std::make_pair(kPredictPhaseTag_Pipeline, i));
  }
  method_def.inputs = method_signature.inputs;
  method_def.outputs = method_signature.outputs;
  std::vector<InstancePtr> instances;
  for (size_t i = 0; i < instances_data.size(); i++) {
    auto instance = std::make_shared<Instance>();
    instances.push_back(instance);

    instance->pipeline_data = instances_data[i];
    auto &context = instance->context;
    context.user_id = user_id;
    context.instance_index = i;
    context.user_context = user_context;
  }
  infer_session.instances = instances;
  {
    std::unique_lock<std::mutex> lock(infer_session_map_mutex_);
    infer_session_map_[user_id] = infer_session;
  }

  OnReceivePipelineInputs(instances);
  return SUCCESS;
}

void WorkExecutor::OnReceivePreprocessInputs(const std::vector<InstancePtr> &instances) {
  if (instances.empty()) {
    MSI_LOG_EXCEPTION << "Inputs cannot be empty";
  }
  const MethodSignature &method_def = instances[0]->context.user_context->method_def;
  CreateInputInstance(instances, kPredictPhaseTag_Preproces);
  if (python_preprocess_names_.count(method_def.preprocess_name) > 0) {
    py_preprocess_task_queue_->PushTask(method_def.preprocess_name, GetWorkerId(), instances);
  } else {
    cpp_preprocess_task_queue_->PushTask(method_def.preprocess_name, GetWorkerId(), instances);
  }
}

void WorkExecutor::OnReceivePostprocessInputs(const std::vector<InstancePtr> &instances) {
  if (instances.empty()) {
    MSI_LOG_EXCEPTION << "Inputs cannot be empty";
  }
  const MethodSignature &method_def = instances[0]->context.user_context->method_def;
  CreateInputInstance(instances, kPredictPhaseTag_Postprocess);
  if (python_postprocess_names_.count(method_def.postprocess_name) > 0) {
    py_postprocess_task_queue_->PushTask(method_def.postprocess_name, GetWorkerId(), instances);
  } else {
    cpp_postprocess_task_queue_->PushTask(method_def.postprocess_name, GetWorkerId(), instances);
  }
}

void WorkExecutor::OnReceivePipelineInputs(const std::vector<InstancePtr> &instances) {
  if (instances.empty()) {
    MSI_LOG_EXCEPTION << "Inputs cannot be empty";
  }
  const MethodSignature &method_def = instances[0]->context.user_context->method_def;
  CreateInputInstance(instances, kPredictPhaseTag_Pipeline);
  if (python_pipeline_names_.count(method_def.pipeline_name) > 0) {
    py_pipeline_task_queue_->PushTask(method_def.pipeline_name, GetWorkerId(), instances);
  }
}
void WorkExecutor::OnReceivePredictInputs(const std::vector<InstancePtr> &instances) {
  // create input for predict, and check
  CreateInputInstance(instances, kPredictPhaseTag_Predict);
  std::vector<InstancePtr> valid_instances;
  for (auto &instance : instances) {
    auto status = CheckPredictInput(instance);
    if (status != SUCCESS) {
      ReplyError(instance, status);
      continue;
    }
    valid_instances.push_back(instance);
  }
  predict_thread_.PushPredictTask(valid_instances);
}

bool WorkExecutor::ReplyRequest(const std::vector<InstancePtr> &outputs) {
  for (auto &item : outputs) {
    ReplyRequest(item);
  }
  return true;
}

bool WorkExecutor::ReplyRequest(const InstancePtr &instance) {
  Status status;
  CreateInputInstance(instance, kPredictPhaseTag_Output);
  instance->error_msg = SUCCESS;
  instance->preprocess_data.clear();
  instance->predict_data.clear();
  instance->postprocess_data.clear();
  instance->pipeline_data.clear();
  std::unique_lock<std::mutex> lock(infer_session_map_mutex_);
  auto it = infer_session_map_.find(instance->context.user_id);
  if (it == infer_session_map_.end()) {
    MSI_LOG_WARNING << "Cannot find user in session map, user id " << instance->context.user_id;
    return false;
  }
  auto &infer_session = it->second;
  infer_session.reply_count++;
  if (infer_session.reply_count == infer_session.instances.size()) {
    infer_session.call_back(infer_session.instances);
    infer_session_map_.erase(it);
  }
  return true;
}

bool WorkExecutor::ReplyError(const std::vector<InstancePtr> &context, const Status &error_msg) {
  for (auto &item : context) {
    ReplyError(item, error_msg);
  }
  return true;
}

bool WorkExecutor::ReplyError(const InstancePtr &instance, const Status &error_msg) {
  instance->error_msg = error_msg;
  instance->data.clear();
  instance->preprocess_data.clear();
  instance->predict_data.clear();
  instance->postprocess_data.clear();
  instance->pipeline_data.clear();
  instance->phase = kInstancePhaseDone;

  std::unique_lock<std::mutex> lock(infer_session_map_mutex_);
  auto it = infer_session_map_.find(instance->context.user_id);
  if (it == infer_session_map_.end()) {
    MSI_LOG_WARNING << "Cannot find user in session map, user id " << instance->context.user_id;
    return false;
  }
  auto &infer_session = it->second;
  infer_session.reply_count++;
  if (infer_session.reply_count == infer_session.instances.size()) {
    infer_session.call_back(infer_session.instances);
    infer_session_map_.erase(it);
  }
  return true;
}

void WorkExecutor::PredictHandle(const std::vector<InstancePtr> &instances) {
  Status status;
  try {
    status = Predict(instances);
    if (status != SUCCESS) {
      this->ReplyError(instances, status);
      return;
    }
    std::map<std::string, std::vector<InstancePtr>> map_output;
    std::vector<InstancePtr> reply_result;
    for (auto &instance : instances) {
      MethodSignature &method_def = instance->context.user_context->method_def;
      if (!method_def.postprocess_name.empty()) {
        map_output[method_def.postprocess_name].push_back(instance);
      } else {
        reply_result.push_back(instance);
      }
    }
    if (!reply_result.empty()) {
      ReplyRequest(reply_result);
    }
    for (auto &item : map_output) {
      OnReceivePostprocessInputs(item.second);
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
  ReplyError(instances, status);
}

Status WorkExecutor::PrePredict(const std::vector<InstancePtr> &instances) {
  auto subgraph = instances[0]->context.user_context->method_def.subgraph;
  auto input_batch_size = static_cast<uint32_t>(instances.size());
  uint32_t model_batch_size = model_batch_size_;
  if (input_batch_size == 0 || input_batch_size > model_batch_size) {
    MSI_LOG_ERROR << "Input batch size " << input_batch_size << " invalid, model batch size " << model_batch_size;
    return SYSTEM_ERROR;
  }
  for (size_t i = 0; i < inference_inputs_[subgraph].size(); i++) {
    auto &inputs = inference_inputs_[subgraph];
    auto &tensor = inputs[i];
    auto data_size = tensor->data_size();
    auto dst_buffer = reinterpret_cast<uint8_t *>(tensor->mutable_data());
    if (IsNoBatchDimInput(i)) {
      if (data_size != instances[0]->data[i]->data_size()) {
        return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
               << "Input " << i << " data size " << instances[0]->data[i]->data_size() << "does not match size "
               << data_size << " defined in model";
      }
      memcpy_s(dst_buffer, data_size, instances[0]->data[i]->data(), data_size);
      continue;
    }
    auto item_size = static_cast<size_t>(data_size / model_batch_size);
    for (uint32_t k = 0; k < input_batch_size; k++) {
      if (i >= instances[k]->data.size()) {
        return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << " Batch index " << k << " does not have input " << i;
      }
      if (item_size != instances[k]->data[i]->data_size()) {
        return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
               << "Input " << i << " Batch index " << k << " input data size " << instances[k]->data[i]->data_size()
               << "does not match size " << item_size << " defined in model";
      }
      memcpy_s(dst_buffer + k * item_size, data_size - k * item_size, instances[k]->data[i]->data(), item_size);
    }
    for (uint32_t k = input_batch_size; k < model_batch_size; k++) {
      memcpy_s(dst_buffer + k * item_size, data_size - k * item_size, instances[0]->data[i]->data(), item_size);
    }
  }
  return SUCCESS;
}

Status WorkExecutor::PostPredict(const std::vector<InstancePtr> &instances,
                                 const std::vector<TensorBasePtr> &predict_result) {
  auto subgraph = instances[0]->context.user_context->method_def.subgraph;
  auto input_batch_size = static_cast<uint32_t>(instances.size());
  uint32_t model_batch_size = model_batch_size_;
  if (input_batch_size == 0 || input_batch_size > model_batch_size) {
    MSI_LOG_ERROR << "Input batch size " << input_batch_size << " invalid, model batch size " << model_batch_size;
    return SYSTEM_ERROR;
  }
  if (predict_result.size() != output_infos_[subgraph].size()) {
    MSI_LOG_ERROR << "Output result count " << predict_result.size() << " not equal to output_infos_ count "
                  << output_infos_[subgraph].size();
    return SYSTEM_ERROR;
  }
  std::vector<ResultInstance> results_data(input_batch_size);
  auto &output = output_infos_[subgraph];
  for (size_t i = 0; i < predict_result.size(); i++) {
    auto &item = predict_result[i];
    auto &output_info = output[i];
    if (item->data_size() != output_info.tensor_info.size) {
      MSI_LOG_ERROR << "Output result " << i << " data size " << item->data_size() << " not equal to size "
                    << output_info.tensor_info.size << " in output_infos_ ";
      return SYSTEM_ERROR;
    }
    auto item_size = output_info.size_one_batch;
    auto shape = output_info.shape_one_batch;
    auto data_type = output_info.tensor_info.data_type;
    auto src_buffer = const_cast<uint8_t *>(item->data());
    for (uint32_t k = 0; k < input_batch_size; k++) {
      auto tensor =
        std::make_shared<BufferTensorWithOwner>(item, data_type, shape, src_buffer + item_size * k, item_size, true);
      results_data[k].data.push_back(tensor);
    }
  }
  CreateResultInstance(instances, results_data, kPredictPhaseTag_Predict);
  return SUCCESS;
}

Status WorkExecutor::Predict(const std::vector<InstancePtr> &instances) {
  Status status;
  std::vector<TensorBasePtr> predict_outputs;
  status = PrePredict(instances);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Call Pre Predict failed, model info " << servable_declare_.servable_meta.Repr();
    return status;
  }
  status = servable_->Predict(inference_inputs_[instances[0]->context.user_context->method_def.subgraph],
                              &predict_outputs, instances[0]->context.user_context->method_def.subgraph);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Predict failed, model info " << servable_declare_.servable_meta.Repr();
    return status;
  }
  status = PostPredict(instances, predict_outputs);
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

Status WorkExecutor::CheckPredictInput(const InstancePtr &instance) {
  auto subgraph = instance->context.user_context->method_def.subgraph;
  const auto &inputs_info = input_infos_[subgraph];
  if (instance->data.size() < inputs_info.size()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Given model inputs size " << instance->data.size()
                                                  << " less than model inputs size " << inputs_info.size();
  }
  for (size_t i = 0; i < instance->data.size(); i++) {
    auto input_data = instance->data[i];
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

void WorkExecutor::CreateInputInstance(const std::vector<InstancePtr> &instances, PredictPhaseTag phase) {
  for (auto &instance : instances) {
    CreateInputInstance(instance, phase);
  }
}

void WorkExecutor::CreateInputInstance(const InstancePtr &instance, PredictPhaseTag phase) {
  instance->data.clear();
  MethodSignature &method_def = instance->context.user_context->method_def;
  std::vector<std::pair<PredictPhaseTag, uint64_t>> *inputs = nullptr;
  switch (phase) {
    case kPredictPhaseTag_Preproces:
      inputs = &method_def.preprocess_inputs;
      instance->phase = kInstancePhasePreprocess;
      break;
    case kPredictPhaseTag_Predict:
      inputs = &method_def.servable_inputs;
      instance->phase = kInstancePhasePredict;
      break;
    case kPredictPhaseTag_Postprocess:
      inputs = &method_def.postprocess_inputs;
      instance->phase = kInstancePhasePostprocess;
      break;
    case kPredictPhaseTag_Pipeline:
      inputs = &method_def.pipeline_inputs;
      instance->phase = kInstancePhasePipeline;
      break;
    case kPredictPhaseTag_Output:
      inputs = &method_def.returns;
      instance->phase = kInstancePhaseDone;
      break;
    default: {
      MSI_LOG_EXCEPTION << "Invalid phase tag " << phase;
    }
  }
  for (auto &item : *inputs) {
    const InstanceData *data = nullptr;
    switch (item.first) {
      case kPredictPhaseTag_Input:
        data = &instance->input_data;
        break;
      case kPredictPhaseTag_Pipeline:
        data = &instance->pipeline_data;
        break;
      case kPredictPhaseTag_Preproces:
        data = &instance->preprocess_data;
        break;
      case kPredictPhaseTag_Predict:
        data = &instance->predict_data;
        break;
      case kPredictPhaseTag_Postprocess:
        data = &instance->postprocess_data;
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
    instance->data.push_back(data->at(item.second));
  }
}

void WorkExecutor::CreateResultInstance(std::vector<InstancePtr> instances, const std::vector<ResultInstance> &results,
                                        PredictPhaseTag phase) {
  for (size_t i = 0; i < instances.size(); i++) {
    CreateResultInstance(instances[i], results[i], phase);
  }
}

void WorkExecutor::CreateResultInstance(const InstancePtr &instance, const ResultInstance &result,
                                        PredictPhaseTag phase) {
  instance->data.clear();
  switch (phase) {
    case kPredictPhaseTag_Input:
      instance->input_data = result.data;
      break;
    case kPredictPhaseTag_Pipeline:
      instance->pipeline_data = result.data;
      break;
    case kPredictPhaseTag_Preproces:
      instance->preprocess_data = result.data;
      break;
    case kPredictPhaseTag_Predict:
      instance->predict_data = result.data;
      break;
    case kPredictPhaseTag_Postprocess:
      instance->postprocess_data = result.data;
      break;
    default: {
      MSI_LOG_EXCEPTION << "Invalid phase tag " << phase;
    }
  }
}

uint64_t WorkExecutor::GetNextUserId() {
  static std::atomic<uint64_t> user_id;
  return ++user_id;
}

uint32_t WorkExecutor::GetWorkerId() const { return worker_id_; }

void WorkExecutor::ClearInstances(Status error_msg) {
  std::unique_lock<std::mutex> lock(infer_session_map_mutex_);
  for (auto &item : infer_session_map_) {
    auto &infer_session = item.second;
    for (auto &instance : infer_session.instances) {
      if (instance->phase != kInstancePhaseDone) {
        instance->error_msg = error_msg;
      }
    }
    item.second.call_back(item.second.instances);
  }
  infer_session_map_.clear();
}

}  // namespace serving
}  // namespace mindspore
