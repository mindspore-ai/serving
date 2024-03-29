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

#ifndef MINDSPORE_SERVING_EXIT_HANDLE_H
#define MINDSPORE_SERVING_EXIT_HANDLE_H
#include <functional>
#include <atomic>
#include <future>
#include "common/serving_common.h"

namespace mindspore {
namespace serving {
// Handle Ctrl+C signal. When the master or worker is waiting for the Ctrl+C signal,
// it can continue to perform subsequent operations, such as cleaning.
class MS_API ExitSignalHandle {
 public:
  static ExitSignalHandle &Instance();
  void InitSignalHandle();
  void MasterWait();
  void WorkerWait();
  void AgentWait();
  void Start();
  void Stop();
  bool HasStopped() const;

 private:
  std::promise<void> master_exit_requested_;
  std::promise<void> worker_exit_requested_;
  std::promise<void> agent_exit_requested_;
  std::atomic_flag has_exited_ = true;
  std::atomic_flag has_inited_ = ATOMIC_FLAG_INIT;
  std::atomic_bool is_running_ = false;
  int exit_signal_ = 0;

  static void HandleSignal(int sig);
  void HandleSignalInner(int sig);
};
}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_EXIT_HANDLE_H
