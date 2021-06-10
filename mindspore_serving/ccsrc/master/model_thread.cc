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

#include "master/model_thread.h"
#include "common/proto_tensor.h"

namespace mindspore {
namespace serving {

ModelThread::ModelThread(const std::string &servable_name, const std::string &method_name, uint64_t version_number,
                         uint64_t batch_size, ServableMethodInfo method_info) {
  spec_.servable_name = servable_name;
  spec_.method_name = method_name;
  spec_.version_number = version_number;
  method_info_ = method_info;
  batch_size_ = batch_size;
}

void ModelThread::Clear() {
  std::unique_lock<std::mutex> lock(lock_);
  InnerClear();
}

void ModelThread::InnerClear() {
  for (auto &job_item : job_) {
    auto reply = job_item.second.reply;
    bool has_reply = false;
    bool has_error = false;
    proto::ErrorMsg detect_error;
    proto::ErrorMsg exit_error;
    RequestSpec request_spec;
    GrpcTensorHelper::GetRequestSpec(*job_item.second.request, &request_spec);
    auto status = INFER_STATUS(INVALID_INPUTS) << "Request " << request_spec.Repr() << ", servable is not available";
    exit_error.set_error_code(status.StatusCode());
    exit_error.set_error_msg(status.StatusMessage());
    for (auto &task_item : job_item.second.task) {
      auto instance = reply->add_instances();
      auto error = reply->add_error_msg();
      if (task_item.error.error_code() != 0) {
        *error = task_item.error;
        if (!has_error) {
          has_error = true;
          detect_error = task_item.error;
        }
      } else if (task_item.output != nullptr) {
        *instance = *task_item.output;
        has_reply = true;
      } else {
        *error = exit_error;
      }
    }
    if (!has_error && !has_reply) {
      job_item.second.reply->clear_instances();
      job_item.second.reply->clear_error_msg();
      auto error_msg = job_item.second.reply->add_error_msg();
      *error_msg = exit_error;
    } else if (!has_reply) {
      job_item.second.reply->clear_instances();
      job_item.second.reply->clear_error_msg();
      auto error_msg = job_item.second.reply->add_error_msg();
      *error_msg = detect_error;
    }
    job_item.second.callback();
  }
  job_.clear();
  pid_process_.clear();
  task_wait_queue_ = std::queue<std::pair<uint64_t, uint64_t>>();
  worker_wait_map_.clear();
}

ModelThread::~ModelThread() { Clear(); }

Status ModelThread::AddWorker(uint64_t pid, const std::shared_ptr<WorkerContext> &notify) {
  {
    std::unique_lock<std::mutex> lock(lock_);
    auto it = pid_process_.find(pid);
    if (it != pid_process_.end()) {
      MSI_LOG(INFO) << "pid is existed: " << pid;
      return FAILED;
    }
    pid_process_.insert(std::make_pair(pid, notify));
    worker_wait_map_[pid] = static_cast<int64_t>(round_);
  }
  SendTasks();
  return SUCCESS;
}

Status ModelThread::DelWorker(uint64_t pid) {
  {
    std::unique_lock<std::mutex> lock(lock_);
    auto it = pid_process_.find(pid);
    if (it == pid_process_.end()) {
      MSI_LOG(INFO) << "pid not existed: " << pid;
      return FAILED;
    }
    pid_process_.erase(it);
    auto worker_it = worker_wait_map_.find(pid);
    if (worker_it == worker_wait_map_.end()) {
      MSI_LOG(INFO) << "pid not existed in worker wait map: " << pid;
      return FAILED;
    }
    worker_wait_map_.erase(worker_it);
    for (auto &job_item : job_) {
      auto job_id = job_item.first;
      auto &task_list = job_item.second.task;
      for (size_t i = 0; i < task_list.size(); ++i) {
        if (task_list[i].pid == pid) {
          auto task_id = i;
          task_wait_queue_.push(std::make_pair(job_id, task_id));
        }
      }
    }
    if (pid_process_.empty()) {
      InnerClear();
    }
  }
  SendTasks();
  return SUCCESS;
}

Status ModelThread::FindProcessQueue(uint64_t *pid) {
  int64_t max_free_slot = 0;
  uint64_t cur_pid = 0;
  for (auto &item : worker_wait_map_) {
    auto slot = item.second;
    if (slot <= 0 || slot < max_free_slot) {
      continue;
    }
    if (slot > max_free_slot || (cur_pid <= last_worker_pid_ && item.first > last_worker_pid_)) {
      max_free_slot = slot;
      cur_pid = item.first;
    }
  }
  if (cur_pid != 0) {
    worker_wait_map_[cur_pid]--;
    last_worker_pid_ = cur_pid;
    *pid = cur_pid;
    return SUCCESS;
  }
  return FAILED;
}

Status ModelThread::PushTasks(const proto::PredictRequest &request, proto::PredictReply *reply,
                              PredictOnFinish callback) {
  auto status = GrpcTensorHelper::CheckRequestInstances(request, method_info_.input_names);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Check request failed";
    return status;
  }
  std::unique_lock<std::mutex> lock(lock_);
  if (pid_process_.empty()) {
    RequestSpec request_spec;
    GrpcTensorHelper::GetRequestSpec(request, &request_spec);
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Request " << request_spec.Repr() << ", servable is not available";
  }
  std::vector<const proto::Instance *> instances_data;
  for (auto &item : request.instances()) {
    // cppcheck-suppress useStlAlgorithm
    instances_data.push_back(&item);
  }
  auto it = job_.find(job_id_);
  if (it != job_.end()) {
    MSI_LOG(ERROR) << "job_id has existed: " << job_id_;
    return FAILED;
  }
  Job job;
  job.job_id = job_id_;
  job.wait_task_num = instances_data.size();
  job.callback = callback;
  job.request = &request;
  job.reply = reply;
  job.task.resize(instances_data.size());
  for (unsigned int i = 0; i < instances_data.size(); i++) {
    Task &task = job.task[i];
    task.input = instances_data[i];
    task.pid = 0;
    task_wait_queue_.push(std::make_pair(job_id_, i));
  }
  job_.insert(std::make_pair(job_id_, job));
  job_id_++;
  return SUCCESS;
}

Status ModelThread::DispatchAsync(const proto::PredictRequest &request, proto::PredictReply *reply,
                                  PredictOnFinish callback) {
  auto status = PushTasks(request, reply, std::move(callback));
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Push tasks into queue failed";
    return status;
  }
  SendTasks();
  return SUCCESS;
}

Status ModelThread::Combine(const std::vector<std::pair<uint64_t, uint64_t>> &ids, uint64_t pid,
                            proto::PredictRequest *msg) {
  std::vector<const proto::Instance *> inputs;
  // ids->inputs
  for (auto it = begin(ids); it != end(ids); it++) {
    uint64_t job_id = it->first;
    uint64_t task_id = it->second;
    job_[job_id].task[task_id].pid = pid;
    inputs.push_back(job_[job_id].task[task_id].input);
  }
  return GrpcTensorHelper::CreatePredictRequestFromInstances(spec_, inputs, msg);
}

void ModelThread::SendTasks() {
  while (true) {
    std::shared_ptr<PredictContext> context;
    std::shared_ptr<WorkerContext> worker;
    {  // pop tasks
      std::unique_lock<std::mutex> lock(lock_);
      if (task_wait_queue_.empty()) {
        return;
      }
      uint64_t pid;
      auto status = FindProcessQueue(&pid);
      if (status != SUCCESS) {
        return;
      }
      context = std::make_shared<PredictContext>();
      std::vector<std::pair<uint64_t, uint64_t>> &inputs = context->inputs;
      for (uint64_t i = 0; i < batch_size_; i++) {
        if (task_wait_queue_.empty()) {
          break;
        }
        inputs.push_back(task_wait_queue_.front());
        task_wait_queue_.pop();
      }
      context->pid = pid;
      Combine(inputs, pid, &context->request);  // inputs string->InstanceData,task pid status
      worker = pid_process_[pid];
    }
    // send request
    PredictOnFinish callback = [context, worker, this]() {
      bool worker_not_available = false;
      for (auto &error : context->reply.error_msg()) {
        if (error.error_code() == WORKER_UNAVAILABLE) {
          worker_not_available = true;
          break;
        }
      }
      if (worker_not_available) {
        worker->NotifyNotAvailable();
      } else {
        Commit(context);
      }
    };
    auto status = worker->DispatchAsync(context->request, &context->reply, callback);
    if (status != SUCCESS) {
      auto error_msg = context->reply.add_error_msg();
      error_msg->set_error_code(WORKER_UNAVAILABLE);
      error_msg->set_error_msg(status.StatusMessage());
      worker->NotifyNotAvailable();
    }
  }
}

void ModelThread::OnTasksFinished(const std::shared_ptr<PredictContext> &context) {
  std::unique_lock<std::mutex> lock(lock_);
  const auto pid = context->pid;
  const auto &inputs = context->inputs;
  if (pid_process_.find(pid) != pid_process_.end()) {
    worker_wait_map_[pid]++;
  }
  std::vector<proto::ErrorMsg> error;
  std::vector<const proto::Instance *> output;
  auto status = GrpcTensorHelper::CreateInstanceFromPredictReply(spec_, context->reply, &error, &output);
  if (status != SUCCESS) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
             << "Get reply failed, servable name: " << spec_.servable_name << ", method name: " << spec_.method_name
             << ", version number: " << spec_.version_number;
  }
  if (!output.empty() && output.size() != inputs.size()) {
    status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
             << "The instance count " << output.size() << " of reply is not equal to the count " << inputs.size()
             << " of request";
  }
  if (status != SUCCESS) {
    output.clear();
    error.clear();
    proto::ErrorMsg error_msg;
    error_msg.set_error_code(status.StatusCode());
    error_msg.set_error_msg(status.StatusMessage());
    error.push_back(error_msg);
  }
  for (unsigned int i = 0; i < inputs.size(); i++) {
    uint64_t task_id = inputs[i].second;
    uint64_t job_id = inputs[i].first;
    auto iter2 = job_.find(job_id);
    if (iter2 == job_.end()) {
      MSI_LOG_ERROR << "job_id not exist: " << job_id;
      continue;
    }
    auto &job_item = iter2->second;
    // collect result
    auto &task_item = job_item.task[task_id];
    task_item.pid = 0;
    if (i < output.size()) {
      task_item.output = output[i];
    }
    if (error.empty()) {
      task_item.error.set_error_code(0);
    } else if (error.size() == 1) {
      task_item.error = error[0];
    } else {
      task_item.error = error[i];
    }
    job_item.wait_task_num--;
    job_item.reply_context_list.push_back(context);
    if (job_item.wait_task_num == 0) {
      // reply job
      std::vector<const proto::Instance *> out;
      std::vector<proto::ErrorMsg> error_reply;
      for (auto &item : job_item.task) {
        out.push_back(item.output);
        error_reply.push_back(item.error);
      }
      GrpcTensorHelper::CreatePredictReplyFromInstances(*job_item.request, error_reply, out, job_item.reply);
      job_item.callback();
      job_.erase(iter2);
    }
  }
}

void ModelThread::Commit(const std::shared_ptr<PredictContext> &context) {
  OnTasksFinished(context);
  SendTasks();
}

}  // namespace serving
}  // namespace mindspore
