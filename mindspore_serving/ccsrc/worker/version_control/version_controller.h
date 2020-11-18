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
#ifndef MINDSPORE_SERVING_VERSOIN_CONTROLLER_H_
#define MINDSPORE_SERVING_VERSOIN_CONTROLLER_H_

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include "common/serving_common.h"

namespace mindspore {
namespace serving {
class VersionController {
 public:
  VersionController();
  ~VersionController();
  void StartPollModelPeriodic();
  void StopPollModelPeriodic();

 private:
  std::thread poll_model_thread_;

  std::mutex m_lock_;
  std::condition_variable cond_var_;
  std::atomic<bool> is_stoped_ = false;
  std::atomic<bool> has_started = false;
  uint32_t poll_model_wait_seconds_ = 5;
  void ThreadFunc();
};

}  // namespace serving
}  // namespace mindspore

#endif  // !MINDSPORE_SERVING_VERSOIN_CONTROLLER_H_
