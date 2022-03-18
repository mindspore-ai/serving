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

#include "worker/predict_thread.h"
#include <vector>
#include <memory>
#include <string>
#include "worker/task_queue.h"
#include "worker/stage_function.h"
#include "common/buffer_tensor.h"

namespace mindspore::serving {
serving::PredictThread::PredictThread() {}
PredictThread::~PredictThread() noexcept { Stop(); }

void PredictThread::PushPredictTask(const MethodStage &stage, const std::vector<InstancePtr> &inputs) {
  // create input for predict, and check
  std::vector<InstancePtr> valid_instances;
  for (auto &instance : inputs) {
    auto status = CheckPredictInput(stage.subgraph, instance);
    if (status != SUCCESS) {
      task_que_.PushTaskResult({instance}, status);
      continue;
    }
    valid_instances.push_back(instance);
  }
  if (!valid_instances.empty()) {
    auto group_name = AsGroupName(stage.stage_key, stage.subgraph);
    task_que_.PushTask(group_name, 0, valid_instances);
  }
}

void PredictThread::ThreadFunc(PredictThread *queue) { queue->Predict(); }

void PredictThread::Predict() {
  while (true) {
    TaskItem task_item;
    task_que_.PopTask(&task_item);
    if (task_item.has_stopped) {
      MSI_LOG_INFO << "Predict task has stopped, exit predict thread";
      break;
    }
    MSI_TIME_STAMP_START(InvokePredict)
    PredictHandle(task_item.task_info, task_item.instance_list);
    MSI_TIME_STAMP_END_EXTRA(InvokePredict, task_item.task_info.tag)
  }
}

void PredictThread::Stop() {
  task_que_.Stop();
  if (predict_thread_.joinable()) {
    try {
      predict_thread_.join();
    } catch (const std::system_error &) {
    } catch (...) {
    }
  }
}

std::string PredictThread::AsGroupName(const std::string &model_key, uint64_t subgraph) const {
  return model_key + "_subgraph" + std::to_string(subgraph);
}

void PredictThread::Start(const std::string &que_name, const std::shared_ptr<ModelLoaderBase> &model_loader,
                          const ModelMeta &model_meta, const TaskCallBack &task_callback) {
  MSI_EXCEPTION_IF_NULL(model_loader);
  MSI_EXCEPTION_IF_NULL(task_callback);
  model_loader_ = model_loader;
  model_meta_ = model_meta;
  auto &model_key = model_meta.common_meta.model_key;
  auto graph_num = model_loader_->GetGraphNum();

  auto batch_size = model_loader->GetBatchSize();
  // init executor info
  executor_info_.sub_graph_infos.resize(graph_num);
  executor_info_.batch_size = batch_size;
  for (uint64_t i = 0; i < graph_num; i++) {
    auto input_infos = model_loader_->GetInputInfos(i);
    auto &subgraph_info = executor_info_.sub_graph_infos[i];
    subgraph_info.input_infos = input_infos;
  }
  // init task infos
  std::vector<TaskInfo> task_infos;
  for (uint64_t i = 0; i < graph_num; i++) {
    TaskInfo info;
    info.group_name = AsGroupName(model_key, i);
    info.subgraph = i;
    info.task_name = info.group_name;
    info.priority = 0;
    info.batch_size = batch_size;
    info.tag = "Model " + model_key + (graph_num > 1 ? " subgraph " + std::to_string(i) : "");
    task_infos.push_back(info);
  }
  task_que_.Start(que_name, task_infos, task_callback);  // start before predict_thread_ start
  predict_thread_ = std::thread(ThreadFunc, this);
}

void PredictThread::PredictHandle(const TaskInfo &task_info, const std::vector<InstancePtr> &instances) {
  Status status;
  try {
    std::vector<ResultInstance> instance_result;
    status = PredictInner(task_info, instances, &instance_result);
    if (status != SUCCESS) {
      task_que_.PushTaskResult(instances, status);
      return;
    }
    task_que_.PushTaskResult(instances, instance_result);
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
  task_que_.PushTaskResult(instances, status);
}

Status PredictThread::PredictInner(const TaskInfo &task_info, const std::vector<InstancePtr> &instances,
                                   std::vector<ResultInstance> *instance_result) {
  Status status;
  std::vector<InstanceData> inputs;
  for (auto &item : instances) {
    // cppcheck-suppress useStlAlgorithm
    inputs.push_back(item->data);
  }
  status = model_loader_->Predict(inputs, instance_result, task_info.subgraph);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Predict failed, model info " << model_meta_.common_meta.model_key;
    return status;
  }
  return SUCCESS;
}

Status PredictThread::CheckPredictInput(uint64_t subgraph, const InstancePtr &instance) {
  const auto &inputs_info = executor_info_.sub_graph_infos[subgraph].input_infos;
  if (instance->data.size() < inputs_info.size()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Given model inputs size " << instance->data.size()
                                                  << " less than model inputs size " << inputs_info.size();
  }
  for (size_t i = 0; i < instance->data.size(); i++) {
    auto input_data = instance->data[i];
    if (inputs_info[i].is_no_batch_dim) {
      if (static_cast<size_t>(inputs_info[i].size) != input_data->data_size()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
               << "Given model input " << i << " size " << input_data->data_size() << " not match the size "
               << inputs_info[i].size << " defined in model";
      }
    } else if (static_cast<size_t>(inputs_info[i].size / executor_info_.batch_size) != input_data->data_size()) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
             << "Given model input " << i << " size " << input_data->data_size() << " not match the size "
             << inputs_info[i].size / executor_info_.batch_size << " defined in model";
    }
    if (inputs_info[i].data_type != input_data->data_type()) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
             << "Given model input " << i << " data type " << input_data->data_type() << " not match the data type "
             << inputs_info[i].data_type << " defined in model";
    }
  }
  return SUCCESS;
}
}  // namespace mindspore::serving
