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

#ifndef MINDSPORE_SERVING_WORKER_PREDICT_THREAD_H
#define MINDSPORE_SERVING_WORKER_PREDICT_THREAD_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include "common/instance.h"

namespace mindspore::serving {

using PredictFun = std::function<void(const std::vector<Instance> &inputs)>;
class MS_API PredictThread {
 public:
  PredictThread();
  ~PredictThread();

  Status PushPredictTask(const std::vector<Instance> &inputs);
  void Start(PredictFun predict_fun, uint32_t batch_size);
  void Stop();

 private:
  std::thread predict_thread_;
  std::queue<Instance> predict_buffer_;
  PredictFun predict_fun_;
  uint32_t batch_size_ = 0;

  std::mutex m_lock_;
  std::condition_variable cond_var_;
  std::atomic<bool> is_stoped_{false};
  std::atomic<bool> has_started = false;

  static void ThreadFunc(PredictThread *queue);
  void Predict();
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_WORKER_PREDICT_THREAD_H
