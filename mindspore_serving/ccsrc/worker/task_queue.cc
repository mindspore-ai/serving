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
#include "worker/preprocess.h"
#include "worker/postprocess.h"

static const char *kTaskTypeStop = "stop";
static const char *kTaskTypeEmpty = "empty";
static const char *kTaskTypePreprocess = "preprocess";
static const char *kTaskTypePostprocess = "postprocess";

namespace mindspore::serving {

TaskQueue::TaskQueue() {}

TaskQueue::TaskQueue(std::shared_ptr<std::mutex> lock, std::shared_ptr<std::condition_variable> cond_var)
    : lock_(lock), cond_var_(cond_var) {}

TaskQueue::~TaskQueue() { Stop(); }

Status TaskQueue::SetWorkerCallback(uint64_t worker_id, TaskCallBack on_task_done) {
  callback_map_[worker_id] = on_task_done;
  return SUCCESS;
}

void TaskQueue::PushTask(const std::string &task_name, uint64_t worker_id, const std::vector<Instance> &inputs) {
  if (inputs.empty()) {
    MSI_LOG_EXCEPTION << "Inputs cannot be empty";
  }
  {
    std::unique_lock<std::mutex> lock{*lock_};
    auto &task_list = task_map_[task_name];
    task_list.name = task_name;
    for (auto &input : inputs) {
      TaskContext context;
      context.user_id = input.context.user_id;
      context.instance_index = input.context.instance_index;
      context.worker_id = worker_id;

      task_list.context_list.push_back(context);
      task_list.instance_list.push_back(input);
    }
    task_priority_list_.push(task_name);
  }
  cond_var_->notify_one();
}

void TaskQueue::PushTaskResult(uint64_t worker_id, const Instance &input, const ResultInstance &output) {
  auto it = callback_map_.find(worker_id);
  if (it == callback_map_.end()) {
    MSI_LOG_ERROR << "Worker serivce " << worker_id << " has not specificated callback";
    return;
  }
  if (it->second == nullptr) {
    MSI_LOG_ERROR << "Worker serivce " << worker_id << " has not specify callback preprocess";
    return;
  }
  it->second({input}, {output});
}

void TaskQueue::PushTaskResult(uint64_t worker_id, const std::vector<Instance> &inputs,
                               const std::vector<ResultInstance> &outputs) {
  auto it = callback_map_.find(worker_id);
  if (it == callback_map_.end()) {
    MSI_LOG_ERROR << "Worker serivce " << worker_id << " has not specificated callback";
    return;
  }
  if (it->second == nullptr) {
    MSI_LOG_ERROR << "Worker serivce " << worker_id << " has not specify callback preprocess";
    return;
  }
  it->second(inputs, outputs);
}

Status TaskQueue::PushTaskPyResult(const std::vector<ResultInstance> &outputs) {
  auto &context_list = task_item_processing_.context_list;
  auto &instance_list = task_item_processing_.instance_list;
  if (outputs.empty() || context_list.size() < outputs.size()) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "processing task not match result, processing size "
                                                << context_list.size() << ", result size " << outputs.size();
  }
  std::unordered_map<uint64_t, std::pair<std::vector<Instance>, std::vector<ResultInstance>>> worker_result_map;
  for (size_t i = 0; i < outputs.size(); i++) {
    auto &result_item = worker_result_map[context_list[i].worker_id];
    result_item.first.push_back(instance_list[i]);
    result_item.second.push_back(outputs[i]);
  }
  for (auto &item : worker_result_map) {
    PushTaskResult(item.first, item.second.first, item.second.second);
  }
  context_list.erase(context_list.begin(), context_list.begin() + outputs.size());
  instance_list.erase(instance_list.begin(), instance_list.begin() + outputs.size());
  return SUCCESS;
}

void TaskQueue::PopTask(TaskItem &task_item) {
  if (is_stoped_) {
    task_item.task_type = kTaskTypeStop;
    return;
  }
  std::unique_lock<std::mutex> lock{*lock_};
  while (true) {
    if (task_priority_list_.empty()) {
      cond_var_->wait(lock, [this] { return is_stoped_.load() || !task_priority_list_.empty(); });
      if (is_stoped_) {
        task_item.task_type = kTaskTypeStop;
        return;
      }
    }
    auto task_item_info = task_priority_list_.front();
    task_priority_list_.pop();
    auto &cur_task = task_map_[task_item_info];
    if (cur_task.instance_list.empty()) {
      continue;
    }
    task_item = cur_task;
    cur_task.context_list.clear();
    cur_task.instance_list.clear();
    break;
  }
}

void TaskQueue::TryPopTask(TaskItem &task_item) {
  if (is_stoped_) {
    task_item.task_type = kTaskTypeStop;
    return;
  }
  std::unique_lock<std::mutex> lock{*lock_};
  while (true) {
    if (task_priority_list_.empty()) {
      task_item.task_type = kTaskTypeEmpty;
      return;
    }
    auto task_item_info = task_priority_list_.front();
    task_priority_list_.pop();
    auto &cur_task = task_map_[task_item_info];
    if (cur_task.instance_list.empty()) {
      continue;
    }
    task_item = cur_task;
    cur_task.context_list.clear();
    cur_task.instance_list.clear();
    break;
  }
}

void TaskQueue::TryPopPyTask(TaskItem &task_item) {
  TryPopTask(task_item);
  if (IsValidTask(task_item)) {
    task_item_processing_ = task_item;
  }
}

void TaskQueue::Stop() {
  if (is_stoped_) {
    return;
  }
  is_stoped_.store(true);
  cond_var_->notify_all();
}

bool TaskQueue::IsValidTask(const TaskItem &task_item) {
  return task_item.task_type != kTaskTypeStop && task_item.task_type != kTaskTypeEmpty;
}

bool TaskQueue::Empty() const { return task_priority_list_.empty(); }

PyTaskQueueGroup::PyTaskQueueGroup() = default;

PyTaskQueueGroup::~PyTaskQueueGroup() = default;

std::shared_ptr<TaskQueue> PyTaskQueueGroup::GetPreprocessTaskQueue() { return preprocess_task_que_; }
std::shared_ptr<TaskQueue> PyTaskQueueGroup::GetPostprocessTaskQueue() { return postprocess_task_que_; }
void PyTaskQueueGroup::PopPyTask(TaskItem &task_item) {
  while (true) {
    {
      std::unique_lock<std::mutex> lock{*lock_};
      if (preprocess_task_que_->Empty() && postprocess_task_que_->Empty()) {
        cond_var_->wait(lock, [this] {
          return is_stoped_.load() || !(preprocess_task_que_->Empty() && postprocess_task_que_->Empty());
        });
        if (is_stoped_) {
          task_item.task_type = kTaskTypeStop;
          return;
        }
      }
    }
    preprocess_task_que_->TryPopPyTask(task_item);
    if (TaskQueue::IsValidTask(task_item)) {
      task_item.task_type = kTaskTypePreprocess;
      break;
    }
    postprocess_task_que_->TryPopPyTask(task_item);
    if (TaskQueue::IsValidTask(task_item)) {
      task_item.task_type = kTaskTypePostprocess;
      break;
    }
  }
}

void PyTaskQueueGroup::TryPopPreprocessTask(TaskItem &task_item) {
  preprocess_task_que_->TryPopPyTask(task_item);
  if (TaskQueue::IsValidTask(task_item)) {
    task_item.task_type = kTaskTypePreprocess;
  }
}

void PyTaskQueueGroup::TryPopPostprocessTask(TaskItem &task_item) {
  postprocess_task_que_->TryPopPyTask(task_item);
  if (TaskQueue::IsValidTask(task_item)) {
    task_item.task_type = kTaskTypePostprocess;
  }
}

void PyTaskQueueGroup::Stop() {
  is_stoped_ = true;
  preprocess_task_que_->Stop();
  postprocess_task_que_->Stop();
}

TaskQueueThreadPool::TaskQueueThreadPool() = default;

TaskQueueThreadPool::~TaskQueueThreadPool() = default;

void TaskQueueThreadPool::ThreadFunc(TaskQueueThreadPool *thread_pool) {
  while (true) {
    TaskItem task_item;
    thread_pool->task_queue_->PopTask(task_item);
    if (task_item.task_type == kTaskTypeStop) {
      return;
    }
    auto status = thread_pool->HandleTask(task_item);
    if (status != SUCCESS) {
      MSI_LOG_ERROR << "System error happens, thread exit";
      return;
    }
  }
}

void TaskQueueThreadPool::Start(uint32_t size) {
  if (has_started) {
    return;
  }
  has_started = true;
  for (uint32_t i = 0; i < size; ++i) {
    pool_.emplace_back(ThreadFunc, this);
  }
}

void TaskQueueThreadPool::Stop() {
  task_queue_->Stop();
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
}

Status PreprocessThreadPool::HandleTask(TaskItem &task_item) {
  Status status;
  auto preprocess = PreprocessStorage::Instance().GetPreprocess(task_item.name);
  if (!preprocess) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "System error, get preprocess " << task_item.name << " failed";
    return status;
  }
  for (size_t i = 0; i < task_item.instance_list.size(); i++) {
    auto &instance = task_item.instance_list[i];
    auto &context = task_item.context_list[i];
    ResultInstance result;
    try {
      status = preprocess->Preprocess(task_item.name, instance.data, result.data);
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
    task_queue_->PushTaskResult(context.worker_id, instance, result);
  }
  return SUCCESS;
}

Status PostprocessThreadPool::HandleTask(TaskItem &task_item) {
  Status status;
  auto postprocess = PostprocessStorage::Instance().GetPostprocess(task_item.name);
  if (!postprocess) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "System error, get postprocess " << task_item.name << " failed";
    return status;
  }
  for (size_t i = 0; i < task_item.instance_list.size(); i++) {
    auto &instance = task_item.instance_list[i];
    auto &context = task_item.context_list[i];
    ResultInstance result;
    try {
      status = postprocess->Postprocess(task_item.name, instance.data, result.data);
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
    task_queue_->PushTaskResult(context.worker_id, instance, result);
  }
  return SUCCESS;
}

}  // namespace mindspore::serving
