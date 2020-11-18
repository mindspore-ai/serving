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

#include "worker/version_control/version_controller.h"
#include <string>
#include <iostream>
#include <ctime>
#include <memory>
#include <chrono>
#include "worker/worker.h"

namespace mindspore {
namespace serving {

void VersionController::ThreadFunc() {
  while (!is_stoped_.load()) {
    std::unique_lock<std::mutex> lock{m_lock_};
    cond_var_.wait_for(lock, std::chrono::seconds(poll_model_wait_seconds_), [this] { return is_stoped_.load(); });
    if (is_stoped_) {
      break;
    }
    Worker::GetInstance().Update();
  }
}

VersionController::VersionController() = default;

VersionController::~VersionController() = default;

void VersionController::StartPollModelPeriodic() {
  if (has_started) {
    return;
  }
  poll_model_thread_ = std::thread(&VersionController::ThreadFunc, this);
  // poll_model_thread_.detach();
  has_started = true;
}

void VersionController::StopPollModelPeriodic() {
  if (is_stoped_) {
    return;
  }
  is_stoped_.store(true);
  cond_var_.notify_all();
  if (poll_model_thread_.joinable()) {
    try {
      poll_model_thread_.join();
    } catch (const std::system_error &) {
    } catch (...) {
    }
  }
}
}  // namespace serving
}  // namespace mindspore
