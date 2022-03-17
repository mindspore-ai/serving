/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_SERVING_THREAD_POOL_H_
#define MINDSPORE_SERVING_THREAD_POOL_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace mindspore::serving {
using ThreadTask = std::function<void()>;

class ThreadPool {
 public:
  explicit ThreadPool(uint32_t size = 4);

  ~ThreadPool();

  template <class Func, class... Args>
  auto commit(Func &&func, Args &&... args) -> std::future<decltype(func(args...))> {
    using retType = decltype(func(args...));
    std::future<retType> fail_future;
    if (is_stoped_.load()) {
      return fail_future;
    }

    auto bindFunc = std::bind(std::forward<Func>(func), std::forward<Args>(args)...);
    auto task = std::make_shared<std::packaged_task<retType()>>(bindFunc);
    if (task == nullptr) {
      return fail_future;
    }
    std::future<retType> future = task->get_future();
    {
      std::lock_guard<std::mutex> lock{m_lock_};
      (void)tasks_.emplace([task]() { (*task)(); });
    }
    cond_var_.notify_one();
    return future;
  }

  static void ThreadFunc(ThreadPool *thread_pool);

 private:
  std::vector<std::thread> pool_;
  std::queue<ThreadTask> tasks_;
  std::mutex m_lock_;
  std::condition_variable cond_var_;
  std::atomic<bool> is_stoped_;
  std::atomic<uint32_t> idle_thrd_num_;
};
}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_THREAD_POOL_H_
