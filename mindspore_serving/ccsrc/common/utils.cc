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
#include "common/utils.h"
#include <unistd.h>
#include <string>
#include <algorithm>

namespace mindspore::serving::common {

Status CheckAddress(const std::string &address, const std::string &server_tag, std::string *ip, uint16_t *port) {
  Status status;
  auto position = address.find_last_of(':');
  if (position == std::string::npos) {
    status = INFER_STATUS_LOG_ERROR(FAILED)
             << "Serving Error: The format of the " << server_tag << " address '" << address << "' is illegal";
    return status;
  }
  if (position == 0 || position == address.size() - 1) {
    status = INFER_STATUS_LOG_ERROR(FAILED)
             << "Serving Error: Missing ip or port of the " << server_tag << " address '" << address << "'";
    return status;
  }
  if (ip != nullptr) {
    *ip = address.substr(0, position);
  }
  try {
    auto port_number = std::stoi(address.substr(position + 1, address.size()));
    if (port_number < 1 || port_number > 65535) {
      status = INFER_STATUS_LOG_ERROR(FAILED) << "Serving Error: The port of the " << server_tag << " address '"
                                              << address << "' is out of legal range [1 ~ 65535]";
      return status;
    }
    if (port != nullptr) {
      *port = port_number;
    }
  } catch (const std::invalid_argument &) {
    status = INFER_STATUS_LOG_ERROR(FAILED)
             << "Serving Error: The type of " << server_tag << " address '" << address << "' port is not a number";
    return status;
  } catch (const std::out_of_range &) {
    status = INFER_STATUS_LOG_ERROR(FAILED) << "Serving Error: The port of the " << server_tag << " address '"
                                            << address << "' is out of legal range [1 ~ 65535]";
    return status;
  }
  return SUCCESS;
}

bool DirOrFileExist(const std::string &file_path) {
  int ret = access(file_path.c_str(), 0);
  return (ret == -1) ? false : true;
}
}  // namespace mindspore::serving::common
