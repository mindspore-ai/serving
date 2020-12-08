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

#ifndef MINDSPORE_SERVING_STATUS_H
#define MINDSPORE_SERVING_STATUS_H

#include <chrono>
#include <string>
#include <sstream>

#include "common/log.h"

namespace mindspore::serving {

enum StatusCode { SUCCESS = 0, FAILED, INVALID_INPUTS, SYSTEM_ERROR };

class Status {
 public:
  Status() : status_code_(FAILED) {}
  Status(enum StatusCode status_code, const std::string &status_msg = "")
      : status_code_(status_code), status_msg_(status_msg) {}
  bool IsSuccess() const { return status_code_ == SUCCESS; }
  enum StatusCode StatusCode() const { return status_code_; }
  std::string StatusMessage() const { return status_msg_; }
  bool operator==(const Status &other) const { return status_code_ == other.status_code_; }
  bool operator==(enum StatusCode other_code) const { return status_code_ == other_code; }
  bool operator!=(const Status &other) const { return status_code_ != other.status_code_; }
  bool operator!=(enum StatusCode other_code) const { return status_code_ != other_code; }
  operator bool() const = delete;
  Status &operator<(const LogStream &stream) noexcept __attribute__((visibility("default"))) {
    status_msg_ = stream.sstream_->str();
    return *this;
  }
  Status &operator=(const std::string &msg) noexcept __attribute__((visibility("default"))) {
    status_msg_ = msg;
    return *this;
  }

 private:
  enum StatusCode status_code_;
  std::string status_msg_;
};

#define MSI_TIME_STAMP_START(name) auto time_start_##name = std::chrono::steady_clock::now();
#define MSI_TIME_STAMP_END(name)                                                                             \
  {                                                                                                          \
    auto time_end_##name = std::chrono::steady_clock::now();                                                 \
    auto time_cost = std::chrono::duration<double, std::milli>(time_end_##name - time_start_##name).count(); \
    MSI_LOG_INFO << #name " Time Cost # " << time_cost << " ms ---------------------";                       \
  }

#define INFER_STATUS(code) mindspore::serving::Status(code) < mindspore::serving::LogStream()
#define ERROR_INFER_STATUS(status, type, msg) \
  MSI_LOG_ERROR << msg;                       \
  status = mindspore::serving::Status(type, msg)

#define INFER_STATUS_LOG_ERROR(code) mindspore::serving::Status(code) = MSI_LOG_ERROR
#define INFER_STATUS_LOG_WARNING(code) mindspore::serving::Status(code) = MSI_LOG_WARNING

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_STATUS_H
