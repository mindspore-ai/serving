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

#include "common/exit_handle.h"
#include <signal.h>
#include <utility>

namespace mindspore {
namespace serving {

ExitHandle &ExitHandle::Instance() {
  static ExitHandle instance;
  return instance;
}

void ExitHandle::InitSignalHandle() {
  if (!has_inited_.test_and_set()) {
    signal(SIGINT, HandleSignal);
    signal(SIGTERM, HandleSignal);
  }
}

void ExitHandle::MasterWait() {
  InitSignalHandle();
  auto exit_future = master_exit_requested_.get_future();
  exit_future.wait();
}

void ExitHandle::WorkerWait() {
  InitSignalHandle();
  auto exit_future = worker_exit_requested_.get_future();
  exit_future.wait();
}

void ExitHandle::Stop() { HandleSignal(0); }

bool ExitHandle::HasStopped() { return is_exit_; }

void ExitHandle::HandleSignal(int sig) {
  auto &instance = Instance();
  if (!instance.has_exited_.test_and_set()) {
    instance.master_exit_requested_.set_value();
    instance.worker_exit_requested_.set_value();
    instance.is_exit_.store(true);
  }
}

}  // namespace serving
}  // namespace mindspore
