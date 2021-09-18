/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_SERVING_MASTER_MODEL_THREAD_H
#define MINDSPORE_SERVING_MASTER_MODEL_THREAD_H
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <utility>
#include <mutex>
#include <map>
#include <queue>
#include "common/serving_common.h"
#include "common/instance.h"
#include "master/notify_worker/base_notify.h"
#include "proto/ms_service.pb.h"
#include "proto/ms_service.grpc.pb.h"
#include "master/worker_context.h"

namespace mindspore::serving {

struct Task {
  const proto::Instance *input = nullptr;
  const proto::Instance *output = nullptr;
  proto::ErrorMsg error;
  uint64_t pid = 0;  // 0:not execute or have executed.others: executing
};

struct PredictContext {
  proto::PredictRequest request;
  proto::PredictReply reply;
  uint64_t pid;
  std::vector<std::pair<uint64_t, uint64_t>> inputs;
};

struct Job {
  std::vector<Task> task;
  uint64_t wait_task_num = 0;
  PredictOnFinish callback;
  const proto::PredictRequest *request = nullptr;
  proto::PredictReply *reply = nullptr;
  std::vector<std::shared_ptr<PredictContext>> reply_context_list;
};

class MS_API ModelThread {
 public:
  ModelThread(const std::string &servable_name, const std::string &method_name, uint64_t version_number,
              uint64_t batch_size, ServableMethodInfo method_info);
  ~ModelThread();
  Status DelWorker(uint64_t pid);
  Status AddWorker(uint64_t pid, const std::shared_ptr<WorkerContext> &notify);
  Status DispatchAsync(const proto::PredictRequest &request, proto::PredictReply *reply,
                       const PredictOnFinish &callback);

 private:
  std::map<uint64_t, std::shared_ptr<WorkerContext>> pid_process_;
  uint64_t last_worker_pid_ = 0;
  std::map<uint64_t, int64_t> worker_wait_map_;
  std::queue<std::pair<uint64_t, uint64_t>> task_wait_queue_;
  std::map<uint64_t, Job> job_;
  uint64_t job_id_ = 0;
  uint64_t round_ = 3;
  std::mutex lock_;
  RequestSpec spec_;
  ServableMethodInfo method_info_;
  uint64_t batch_size_;
  bool single_batch_dispatch_ = false;

  void Clear();
  void InnerClear();
  Status FindProcessQueue(uint64_t *pid);
  Status PushTasks(const proto::PredictRequest &request, proto::PredictReply *reply, const PredictOnFinish &callback);
  Status Combine(const std::vector<std::pair<uint64_t, uint64_t>> &ids, uint64_t pid, proto::PredictRequest *msg);
  void OnTasksFinished(const std::shared_ptr<PredictContext> &context);
  void SendTasks();
  void Commit(const std::shared_ptr<PredictContext> &context);
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_MASTER_MODEL_THREAD_H
