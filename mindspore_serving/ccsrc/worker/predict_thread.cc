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
#include <utility>
#include "worker/task_queue.h"
#include "worker/preprocess.h"
#include "worker/postprocess.h"

namespace mindspore::serving {

serving::PredictThread::PredictThread() {}
PredictThread::~PredictThread() { Stop(); }

Status PredictThread::PushPredictTask(const std::vector<Instance> &inputs) {
  if (!is_running_) {
    MSI_LOG_EXCEPTION << "Predict task thread has not been started";
  }
  std::unique_lock<std::mutex> lock{m_lock_};
  bool empty_before = predict_buffer_.empty();
  for (auto &item : inputs) {
    predict_buffer_.push(item);
  }
  if (empty_before) {
    cond_var_.notify_one();
  }
  return SUCCESS;
}

void PredictThread::ThreadFunc(PredictThread *queue) { queue->Predict(); }

void PredictThread::Predict() {
  while (true) {
    std::vector<Instance> instances;
    {
      std::unique_lock<std::mutex> lock{m_lock_};
      if (!is_running_) {  // before start, after stop
        break;
      }
      if (predict_buffer_.empty()) {
        cond_var_.wait(lock, [this] { return !is_running_ || !predict_buffer_.empty(); });
        if (!is_running_) {
          return;
        }
      }
      for (uint32_t i = 0; i < batch_size_ && !predict_buffer_.empty(); i++) {
        instances.push_back(predict_buffer_.front());
        predict_buffer_.pop();
      }
    }
    MSI_TIME_STAMP_START(InvokePredict)
    predict_fun_(instances);
    MSI_TIME_STAMP_END(InvokePredict)
  }
}

void PredictThread::Stop() {
  {
    std::unique_lock<std::mutex> lock{m_lock_};
    if (!is_running_) {
      return;
    }
    is_running_ = false;
    cond_var_.notify_all();
  }
  if (predict_thread_.joinable()) {
    try {
      predict_thread_.join();
    } catch (const std::system_error &) {
    } catch (...) {
    }
  }
}

void PredictThread::Start(PredictFun predict_fun, uint32_t batch_size) {
  if (is_running_) {
    return;
  }
  predict_fun_ = std::move(predict_fun);
  batch_size_ = batch_size;
  is_running_ = true;  // set true before predict_thread_ start
  predict_thread_ = std::thread(ThreadFunc, this);
}

}  // namespace mindspore::serving
