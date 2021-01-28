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

ExitSignalHandle &ExitSignalHandle::Instance() {
  static ExitSignalHandle instance;
  return instance;
}

void ExitSignalHandle::InitSignalHandle() {
  if (!has_inited_.test_and_set()) {
    signal(SIGINT, HandleSignal);
    signal(SIGTERM, HandleSignal);
  }
}

// waiting ctrl+c or stop message to exit,
// if no server is running or server has exited, there is no need to wait
void ExitSignalHandle::MasterWait() {
  if (!is_running_) {
    MSI_LOG_INFO << "Exit Handle has not started or has exited";
    return;
  }
  auto exit_future = master_exit_requested_.get_future();
  exit_future.wait();
}

// waiting ctrl+c or stop message to exit,
// if no server is running or server has exited, there is no need to wait
void ExitSignalHandle::WorkerWait() {
  if (!is_running_) {
    MSI_LOG_INFO << "Exit Handle has not started or has exited";
    return;
  }
  auto exit_future = worker_exit_requested_.get_future();
  exit_future.wait();
}

// waiting ctrl+c or stop message to exit,
// if no server is running or server has exited, there is no need to wait
void ExitSignalHandle::AgentWait() {
  if (!is_running_) {
    MSI_LOG_INFO << "Exit Handle has not started or has exited";
    return;
  }
  auto exit_future = agent_exit_requested_.get_future();
  exit_future.wait();
}

void ExitSignalHandle::Start() {
  if (is_running_) {
    return;
  }
  is_running_ = true;
  master_exit_requested_ = std::promise<void>();
  worker_exit_requested_ = std::promise<void>();
  agent_exit_requested_ = std::promise<void>();
  has_exited_.clear();
  InitSignalHandle();
}

void ExitSignalHandle::Stop() { HandleSignal(0); }

bool ExitSignalHandle::HasStopped() { return !is_running_; }

void ExitSignalHandle::HandleSignal(int sig) {
  auto &instance = Instance();
  instance.HandleSignalInner();
}

void ExitSignalHandle::HandleSignalInner() {
  if (!has_exited_.test_and_set()) {
    master_exit_requested_.set_value();
    worker_exit_requested_.set_value();
    agent_exit_requested_.set_value();
    is_running_ = false;
  }
}

}  // namespace serving
}  // namespace mindspore
