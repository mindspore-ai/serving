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

#ifndef MINDSPORE_SERVING_COMMON_UTILS_H
#define MINDSPORE_SERVING_COMMON_UTILS_H

#include <string>
#include "common/status.h"

namespace mindspore {
namespace serving {
namespace common {

static inline std::string GetEnv(const std::string &envvar) {
  const char *value = ::getenv(envvar.c_str());

  if (value == nullptr) {
    return std::string();
  }

  return std::string(value);
}

Status CheckAddress(const std::string &address, const std::string &server_tag, std::string *ip, uint16_t *port);

bool DirOrFileExist(const std::string &file_path);

}  // namespace common
}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_COMMON_UTILS_H
