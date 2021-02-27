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

#ifndef MINDSPORE_STUB_SERVING_UTILS_H
#define MINDSPORE_STUB_SERVING_UTILS_H

#include <memory>
#include <atomic>
#include <string>
#include <vector>
#include <set>
#include "utils/log_adapter.h"

namespace mindspore {

class FuncGraph {};
using FuncGraphPtr = std::shared_ptr<FuncGraph>;

namespace common {
static inline const char *SafeCStr(const std::string &str) {
  const int CACHED_STR_NUM = 1 << 8;
  const int CACHED_STR_MASK = CACHED_STR_NUM - 1;
  std::vector<std::string> STR_HOLDER(CACHED_STR_NUM);

  static std::atomic<uint32_t> index{0};
  uint32_t cur_index = index++;
  cur_index = cur_index & CACHED_STR_MASK;
  STR_HOLDER[cur_index] = str;
  return STR_HOLDER[cur_index].c_str();
}

}  // namespace common
}  // namespace mindspore

#endif  // MINDSPORE_STUB_SERVING_UTILS_H
