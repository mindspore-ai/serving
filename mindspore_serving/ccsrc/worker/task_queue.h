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
#include "common/instance.h"

namespace mindspore::serving {

// key
struct TaskContext {
  uint64_t user_id = 0;
  uint32_t instance_index = 0;
  uint64_t worker_id = 0;
  bool operator==(const TaskContext &other) const {
    return user_id == other.user_id && instance_index == other.instance_index && worker_id == other.worker_id;
  }
  bool operator!=(const TaskContext &other) const { return !operator==(other); }
};

struct TaskItem {
  std::string task_type;  // preprocess, postprocess, pipeline, stop
  std::string name;       // preprocess name, postprocess name
  std::vector<TaskContext> context_list;
  std::vector<InstancePtr> instance_list;
};

using TaskCallBack =
  std::function<void(const std::vector<InstancePtr> &inputs, const std::vector<ResultInstance> &output)>;
// task queue for preprocess and postprocess
class MS_API TaskQueue {
 public:
  TaskQueue();
  TaskQueue(std::shared_ptr<std::mutex> lock, std::shared_ptr<std::condition_variable> cond_var);
  ~TaskQueue();

  Status SetWorkerCallback(uint64_t worker_id, TaskCallBack on_task_done);

  void PushTask(const std::string &task_name, uint64_t worker_id, const std::vector<InstancePtr> &inputs);
  void PopTask(TaskItem *task_item);
  void TryPopTask(TaskItem *task_item);
  void PushTaskResult(uint64_t worker_id, const InstancePtr &input, const ResultInstance &output);
  void PushTaskResult(uint64_t worker_id, const std::vector<InstancePtr> &inputs,
                      const std::vector<ResultInstance> &outputs);
  void TryPopPyTask(TaskItem *task_item);
  Status PushTaskPyResult(const std::vector<ResultInstance> &outputs);

  void Start();
  void Stop();
  bool Empty() const;

  static bool IsValidTask(const TaskItem &task_item);

 private:
  std::unordered_map<std::string, TaskItem> task_map_;
  std::queue<std::string> task_priority_list_;
  TaskItem task_item_processing_;
  std::unordered_map<uint64_t, TaskCallBack> callback_map_;
  std::shared_ptr<std::mutex> lock_ = std::make_shared<std::mutex>();
  std::shared_ptr<std::condition_variable> cond_var_ = std::make_shared<std::condition_variable>();
  std::atomic<bool> is_running = false;
};

class MS_API PyTaskQueueGroup {
 public:
  PyTaskQueueGroup();
  ~PyTaskQueueGroup();

  std::shared_ptr<TaskQueue> GetPreprocessTaskQueue();
  std::shared_ptr<TaskQueue> GetPostprocessTaskQueue();
  std::shared_ptr<TaskQueue> GetPipelineTaskQueue();
  void PopPyTask(TaskItem *task_item);
  void PopPipelineTask(TaskItem *task_item);
  void TryPopPreprocessTask(TaskItem *task_item);
  void TryPopPostprocessTask(TaskItem *task_item);
  void TryPopPipelineTask(TaskItem *task_item);
  void Start();
  void Stop();

 private:
  std::shared_ptr<std::mutex> lock_ = std::make_shared<std::mutex>();
  std::shared_ptr<std::condition_variable> cond_var_ = std::make_shared<std::condition_variable>();
  std::shared_ptr<std::mutex> pipeline_lock_ = std::make_shared<std::mutex>();
  std::shared_ptr<std::condition_variable> cond_pipeline_ = std::make_shared<std::condition_variable>();
  std::atomic<bool> is_running = false;
  std::shared_ptr<TaskQueue> preprocess_task_que_ = std::make_shared<TaskQueue>(lock_, cond_var_);
  std::shared_ptr<TaskQueue> postprocess_task_que_ = std::make_shared<TaskQueue>(lock_, cond_var_);
  std::shared_ptr<TaskQueue> pipeline_task_que_ = std::make_shared<TaskQueue>(pipeline_lock_, cond_pipeline_);
};

class TaskQueueThreadPool {
 public:
  TaskQueueThreadPool();
  virtual ~TaskQueueThreadPool();

  void Start(uint32_t size = 4);
  void Stop();

  std::shared_ptr<TaskQueue> GetTaskQueue() { return task_queue_; }

 protected:
  std::atomic<bool> is_running_ = false;
  std::vector<std::thread> pool_;
  std::shared_ptr<TaskQueue> task_queue_ = std::make_shared<TaskQueue>();

  virtual Status HandleTask(const TaskItem &task_item) = 0;
  static void ThreadFunc(TaskQueueThreadPool *thread_pool);
};

class PreprocessThreadPool : public TaskQueueThreadPool {
 protected:
  Status HandleTask(const TaskItem &task_item) override;
};

class PostprocessThreadPool : public TaskQueueThreadPool {
 protected:
  Status HandleTask(const TaskItem &task_item) override;
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_WORKER_TASK_QUEUE_H
