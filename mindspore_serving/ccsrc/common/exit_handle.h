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

class MS_API ExitHandle {
 public:
  static ExitHandle &Instance();
  void InitSignalHandle();
  void MasterWait();
  void WorkerWait();
  void Stop();
  bool HasStopped();

 private:
  std::promise<void> master_exit_requested_;
  std::promise<void> worker_exit_requested_;
  std::atomic_flag has_exited_ = ATOMIC_FLAG_INIT;
  std::atomic_flag has_inited_ = ATOMIC_FLAG_INIT;
  std::atomic<bool> is_exit_ = false;

  static void HandleSignal(int sig);
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_EXIT_HANDLE_H
