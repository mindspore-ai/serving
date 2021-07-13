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

#ifndef MINDSPORE_SERVING_WORKER_TASK_QUEUE_H
#define MINDSPORE_SERVING_WORKER_TASK_QUEUE_H

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <set>
#include <thread>
#include <map>
#include "common/instance.h"

namespace mindspore::serving {

struct TaskInfo {
  std::string group_name;  // method name
  std::string task_name;   // function name, model name
  uint64_t priority = 0;
  uint64_t batch_size = 0;
  uint64_t subgraph = 0;  // for model
  std::string tag;
};

struct TaskItem {
  bool has_stopped = false;  // whether system is stopped
  TaskInfo task_info;
  std::vector<InstancePtr> instance_list;
};

using TaskCallBack =
  std::function<void(const std::vector<InstancePtr> &inputs, const std::vector<ResultInstance> &output)>;

struct TaskQueuePriority {
  std::map<uint64_t, TaskItem> priority_que_map;  // priority: stage index, task list
  uint64_t priority_que_instances_count = 0;
};

struct TaskQueueGroups {
  std::map<std::string, TaskQueuePriority> group_que_map;  // group name: method name, task que
  size_t next_exe_que = 0;                                 // next method index
  uint64_t groups_que_instances_count = 0;
};

class TaskQueue {
 public:
  TaskQueue();
  void Start(const std::string &que_name, const std::vector<TaskInfo> &task_infos, TaskCallBack callback);
  void Stop();
  void PushTask(const std::string &group_name, size_t priority, const std::vector<InstancePtr> &instances);
  void PopTask(TaskItem *task_item);

  void PushTaskResult(const InstancePtr &input, const ResultInstance &output);
  void PushTaskResult(const std::vector<InstancePtr> &inputs, const std::vector<ResultInstance> &outputs);
  void PushTaskResult(const std::vector<InstancePtr> &inputs, const Status &failed_result);

  bool IsRunning() const { return is_running; }

 private:
  std::string que_name_;
  TaskQueueGroups methods_queue_;

  TaskCallBack task_callback_ = nullptr;
  std::mutex que_lock_;  // Lock only when the queue changes to avoid deadlock caused by lock in complex scenarios.
  std::condition_variable cond_var_;
  bool is_running = false;

  std::chrono::steady_clock::time_point time_last_ = std::chrono::steady_clock::now();

  bool FindProcessTaskQueue(std::string *method_name);
};

class MS_API PyTaskQueue {
 public:
  PyTaskQueue() = default;
  ~PyTaskQueue() = default;

  void Start(const std::string &que_name, const std::vector<MethodStage> &stage_infos, TaskCallBack callback);
  void Stop();
  void PushTask(const std::string &method_name, size_t stage_index, const std::vector<InstancePtr> &instances);
  // for python task
  void PyPopTask(TaskItem *task_item);
  void PyPushTaskResult(const std::vector<ResultInstance> &outputs);
  TaskInfo GetHandledTaskInfo() const { return py_task_item_processing_.task_info; }

  bool IsRunning() const { return task_queue_.IsRunning(); }

 private:
  TaskQueue task_queue_;
  TaskItem py_task_item_processing_;
};

class CppTaskQueueThreadPool {
 public:
  CppTaskQueueThreadPool();
  virtual ~CppTaskQueueThreadPool();

  void Start(const std::string &que_name, const std::vector<MethodStage> &stage_infos, TaskCallBack callback,
             uint32_t size = 4);
  void Stop();

  void PushTask(const std::string &method_name, size_t stage_index, const std::vector<InstancePtr> &instances);

 protected:
  TaskQueue task_queue_;
  std::atomic<bool> is_running_ = false;
  std::vector<std::thread> pool_;

  Status HandleTask(const TaskItem &task_item);
  static void ThreadFunc(CppTaskQueueThreadPool *thread_pool);
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_WORKER_TASK_QUEUE_H
