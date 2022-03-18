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

#include "worker/task_queue.h"
#include <utility>
#include <unordered_map>
#include "worker/stage_function.h"

namespace mindspore::serving {
TaskQueue::TaskQueue() {}

void TaskQueue::Start(const std::string &que_name, const std::vector<TaskInfo> &task_infos,
                      const TaskCallBack &callback) {
  std::unique_lock<std::mutex> lock{que_lock_};
  if (is_running) {
    return;
  }
  que_name_ = que_name;
  task_callback_ = callback;
  methods_queue_.group_que_map.clear();
  methods_queue_.groups_que_instances_count = 0;
  for (auto &info : task_infos) {
    if (info.batch_size == 0) {
      MSI_LOG_EXCEPTION << "Invalid batch size 0, queue name: " << que_name;
    }
    auto &method_queue = methods_queue_.group_que_map[info.group_name];
    auto &stage_queue = method_queue.priority_que_map[info.priority];
    stage_queue.task_info = info;
  }
  is_running = true;
}

void TaskQueue::Stop() {
  std::unique_lock<std::mutex> lock{que_lock_};
  if (!is_running) {
    return;
  }
  methods_queue_.group_que_map.clear();
  task_callback_ = nullptr;

  is_running = false;
  cond_var_.notify_all();
}

void TaskQueue::PushTask(const std::string &group_name, size_t priority, const std::vector<InstancePtr> &instances) {
  if (instances.empty()) {
    MSI_LOG_WARNING << "Instances cannot be empty";
    return;
  }
  MSI_LOG_DEBUG << que_name_ << " Push instances count " << instances.size()
                << ", inputs size: " << instances[0]->data.size();
  {
    std::unique_lock<std::mutex> lock{que_lock_};
    auto method_it = methods_queue_.group_que_map.find(group_name);
    if (method_it == methods_queue_.group_que_map.end()) {
      MSI_LOG_EXCEPTION << "Cannot find method " << group_name << " in task queue, queue name: " << que_name_;
    }
    auto &stage_queue = method_it->second;
    auto stage_it = stage_queue.priority_que_map.find(priority);
    if (stage_it == stage_queue.priority_que_map.end()) {
      MSI_LOG_EXCEPTION << "Cannot find stage index " << priority << " in task queue, method name: " << group_name
                        << ", queue name: " << que_name_;
    }
    auto &que = stage_it->second;
    for (auto &instance : instances) {
      que.instance_list.push_back(instance);
    }
    stage_queue.priority_que_instances_count += instances.size();
    methods_queue_.groups_que_instances_count += instances.size();
  }
  cond_var_.notify_all();
}

bool TaskQueue::FindProcessTaskQueue(std::string *method_name) {
  auto next_que = methods_queue_.next_exe_que;
  auto &que_map = methods_queue_.group_que_map;
  size_t index = 0;
  std::string name;
  for (auto &item : que_map) {
    if (item.second.priority_que_instances_count > 0 && (name.empty() || index >= next_que)) {
      name = item.first;
      if (index >= next_que) {
        break;
      }
    }
    index++;
  }
  if (name.empty()) {
    return false;
  }
  if (index + 1 >= que_map.size()) {
    methods_queue_.next_exe_que = 0;
  } else {
    methods_queue_.next_exe_que = index + 1;
  }
  *method_name = name;
  return true;
}

void TaskQueue::PopTask(TaskItem *task_item) {
  MSI_EXCEPTION_IF_NULL(task_item);
  std::unique_lock<std::mutex> lock{que_lock_};
  if (!is_running) {  // before start, or after stop
    MSI_LOG_INFO << "Detect task queue is not running, maybe the Serving server is stopped.";
    task_item->has_stopped = true;
    return;
  }
  while (true) {
    if (methods_queue_.groups_que_instances_count == 0) {
      cond_var_.wait(lock, [this] { return !is_running || methods_queue_.groups_que_instances_count > 0; });
      if (!is_running) {
        MSI_LOG_INFO << "Detect task queue '" << que_name_ << "' is not running, maybe the Serving server is stopped.";
        task_item->has_stopped = true;
        return;
      }
    }
    std::string method_name;
    if (!FindProcessTaskQueue(&method_name)) {
      MSI_LOG_EXCEPTION << "Cannot find task when the number " << methods_queue_.groups_que_instances_count
                        << " of instances in task queue is not 0";
    }
    auto &method_que = methods_queue_.group_que_map[method_name];
    auto &stage_que_map = method_que.priority_que_map;
    auto stage_it = stage_que_map.rbegin();
    for (; stage_it != stage_que_map.rend(); ++stage_it) {
      if (!stage_it->second.instance_list.empty()) {
        break;
      }
    }
    if (stage_it == stage_que_map.rend()) {
      MSI_LOG_EXCEPTION << "Cannot find task when the number " << method_que.priority_que_instances_count
                        << " of instances in method task queue is not 0";
    }
    auto &task_handle = stage_it->second;
    auto batch_size = task_handle.task_info.batch_size;
    // Pop a maximum of batch_size instances
    if (task_handle.instance_list.size() <= batch_size) {
      *task_item = task_handle;
      task_handle.instance_list.clear();
    } else {
      *task_item = task_handle;
      auto &instances_ret = task_item->instance_list;
      (void)instances_ret.erase(instances_ret.begin() + static_cast<ptrdiff_t>(batch_size), instances_ret.end());
      auto &instances_reserved = task_handle.instance_list;
      (void)instances_reserved.erase(instances_reserved.begin(),
                                     instances_reserved.begin() + static_cast<ptrdiff_t>(batch_size));
    }
    MSI_LOG_DEBUG << que_name_ << " Pop instances count " << task_item->instance_list.size()
                  << ", batch size: " << batch_size;

    method_que.priority_que_instances_count -= task_item->instance_list.size();
    methods_queue_.groups_que_instances_count -= task_item->instance_list.size();
    break;
  }
}

void TaskQueue::PushTaskResult(const InstancePtr &input, const ResultInstance &output) {
  if (!is_running) {
    MSI_LOG_INFO << "Task queue has exited";
    return;
  }
  task_callback_({input}, {output});
}

void TaskQueue::PushTaskResult(const std::vector<InstancePtr> &inputs, const std::vector<ResultInstance> &outputs) {
  if (!is_running) {
    MSI_LOG_INFO << "Task queue has exited";
    return;
  }
  task_callback_(inputs, outputs);
}

void TaskQueue::PushTaskResult(const std::vector<InstancePtr> &inputs, const Status &failed_result) {
  std::vector<ResultInstance> result;
  for (auto &item : inputs) {
    (void)item;
    ResultInstance output;
    output.error_msg = failed_result;
    result.push_back(output);
  }
  PushTaskResult(inputs, result);
}

void PyTaskQueue::Start(const std::string &que_name, const std::vector<MethodStage> &stage_infos,
                        const TaskCallBack &callback) {
  std::vector<TaskInfo> task_infos;
  for (auto &item : stage_infos) {
    TaskInfo info;
    info.batch_size = item.batch_size;
    info.priority = item.stage_index;
    info.group_name = item.method_name;
    info.task_name = item.stage_key;
    info.tag = item.tag;
    task_infos.push_back(info);
  }
  task_queue_.Start(que_name, task_infos, callback);
  py_task_item_processing_ = TaskItem();
}

void PyTaskQueue::Stop() { task_queue_.Stop(); }

void PyTaskQueue::PushTask(const std::string &method_name, size_t stage_index,
                           const std::vector<InstancePtr> &instances) {
  task_queue_.PushTask(method_name, stage_index, instances);
}

void PyTaskQueue::PyPopTask(TaskItem *task_item) {
  MSI_EXCEPTION_IF_NULL(task_item);
  task_queue_.PopTask(task_item);
  if (!task_item->has_stopped) {
    py_task_item_processing_ = *task_item;
  }
}

void PyTaskQueue::PyPushTaskResult(const std::vector<ResultInstance> &outputs) {
  if (!task_queue_.IsRunning()) {
    MSI_LOG_INFO << "Task queue has exited";
    return;
  }
  auto &instance_list = py_task_item_processing_.instance_list;
  if (outputs.empty() || instance_list.size() < outputs.size()) {
    MSI_LOG_EXCEPTION << "processing task not match result, processing size " << instance_list.size()
                      << ", result size " << outputs.size();
  }
  std::vector<InstancePtr> instances;
  std::vector<ResultInstance> results;
  for (size_t i = 0; i < outputs.size(); i++) {
    instances.push_back(instance_list[i]);
    results.push_back(outputs[i]);
  }
  task_queue_.PushTaskResult(instances, results);
  (void)instance_list.erase(instance_list.begin(), instance_list.begin() + static_cast<ptrdiff_t>(outputs.size()));
}

CppTaskQueueThreadPool::CppTaskQueueThreadPool() = default;

CppTaskQueueThreadPool::~CppTaskQueueThreadPool() = default;

void CppTaskQueueThreadPool::ThreadFunc(CppTaskQueueThreadPool *thread_pool) {
  while (true) {
    TaskItem task_item;
    thread_pool->task_queue_.PopTask(&task_item);
    if (task_item.has_stopped) {
      return;
    }
    auto status = thread_pool->HandleTask(task_item);
    if (status != SUCCESS) {
      MSI_LOG_ERROR << "System error happens, thread exit";
      return;
    }
  }
}

void CppTaskQueueThreadPool::Start(const std::string &que_name, const std::vector<MethodStage> &stage_infos,
                                   const TaskCallBack &callback, uint32_t size) {
  if (is_running_) {
    return;
  }
  is_running_ = true;  // start before ThreadFunc thread pool start
  std::vector<TaskInfo> task_infos;
  for (auto &item : stage_infos) {
    TaskInfo info;
    info.batch_size = item.batch_size;
    info.priority = item.stage_index;
    info.group_name = item.method_name;
    info.task_name = item.stage_key;
    info.tag = item.tag;
    task_infos.push_back(info);
  }
  task_queue_.Start(que_name, task_infos, callback);  // start before ThreadFunc thread pool start
  for (uint32_t i = 0; i < size; ++i) {
    (void)pool_.emplace_back(ThreadFunc, this);
  }
}

void CppTaskQueueThreadPool::Stop() {
  task_queue_.Stop();
  for (std::thread &thd : pool_) {
    if (thd.joinable()) {
      try {
        thd.join();
      } catch (const std::system_error &) {
      } catch (...) {
      }
    }
  }
  pool_.clear();
  is_running_ = false;
}

void CppTaskQueueThreadPool::PushTask(const std::string &method_name, size_t stage_index,
                                      const std::vector<InstancePtr> &instances) {
  task_queue_.PushTask(method_name, stage_index, instances);
}

Status CppTaskQueueThreadPool::HandleTask(const TaskItem &task_item) {
  Status status;
  auto &task_name = task_item.task_info.task_name;
  auto preprocess = CppStageFunctionStorage::Instance().GetFunction(task_name);
  if (!preprocess) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "System error, get preprocess " << task_name << " failed";
    return status;
  }
  for (const auto &instance : task_item.instance_list) {
    ResultInstance result;
    try {
      status = preprocess->Call(task_name, instance->data, &result.data);
    } catch (const std::bad_alloc &ex) {
      status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Serving Error: malloc memory failed";
    } catch (const std::runtime_error &ex) {
      status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Serving Error: runtime error occurred: " << ex.what();
    } catch (const std::exception &ex) {
      status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Serving Error: exception occurred: " << ex.what();
    } catch (...) {
      status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Serving Error: exception occurred";
    }
    if (status != SUCCESS) {
      result.error_msg = status;
    }
    task_queue_.PushTaskResult(instance, result);
  }
  return SUCCESS;
}
}  // namespace mindspore::serving
