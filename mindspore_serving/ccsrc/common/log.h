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

#ifndef MINDSPORE_SERVING_LOG_H
#define MINDSPORE_SERVING_LOG_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <sstream>
#include <memory>
#include <string>

namespace mindspore::serving {
#define MS_API __attribute__((visibility("default")))

#define SERVING_LOG_HDR_FILE_REL_PATH "mindspore_serving/ccsrc/common/log.h"

// Get start index of file relative path in __FILE__
static constexpr int GetRelPathPos() noexcept {
  return sizeof(__FILE__) > sizeof(SERVING_LOG_HDR_FILE_REL_PATH)
           ? sizeof(__FILE__) - sizeof(SERVING_LOG_HDR_FILE_REL_PATH)
           : 0;
}

#define SERVING_FILE_NAME                                                                     \
  (sizeof(__FILE__) > GetRelPathPos() ? static_cast<const char *>(__FILE__) + GetRelPathPos() \
                                      : static_cast<const char *>(__FILE__))

class LogStream {
 public:
  LogStream() { sstream_ = std::make_shared<std::stringstream>(); }
  ~LogStream() = default;

  template <typename T>
  LogStream &operator<<(const T &val) noexcept {
    (*sstream_) << val;
    return *this;
  }

  template <typename T>
  LogStream &operator<<(const std::vector<T> &val) noexcept {
    (*sstream_) << "[";
    for (size_t i = 0; i < val.size(); i++) {
      (*this) << val[i];
      if (i + 1 < val.size()) {
        (*sstream_) << ", ";
      }
    }
    (*sstream_) << "]";
    return *this;
  }

  template <typename K, typename V>
  LogStream &operator<<(const std::unordered_map<K, V> &val) noexcept {
    (*sstream_) << "{";
    for (auto &item : val) {
      (*this) << "{" << item.first << ": " << item.second << "} ";
    }
    (*sstream_) << "}";
    return *this;
  }

  template <typename K, typename V>
  LogStream &operator<<(const std::map<K, V> &val) noexcept {
    (*sstream_) << "{";
    for (auto &item : val) {
      (*this) << "{" << item.first << ": " << item.second << "} ";
    }
    (*sstream_) << "}";
    return *this;
  }

  LogStream &operator<<(std::ostream &func(std::ostream &os)) noexcept {
    (*sstream_) << func;
    return *this;
  }

  friend class LogWriter;
  friend class Status;

 private:
  std::shared_ptr<std::stringstream> sstream_;
};

enum ERROR_LEVEL {
  LOG_DEBUG,
  LOG_INFO,
  LOG_WARNING,
  LOG_ERROR,
  LOG_EXCEPTION,
};

class MS_API LogWriter {
 public:
  LogWriter(const char *file, int line, const char *func, ERROR_LEVEL log_level)
      : file_(file), line_(line), func_(func), log_level_(log_level) {}
  ~LogWriter() = default;

  std::string operator<(const LogStream &stream) const noexcept __attribute__((visibility("default"))) {
    std::ostringstream msg;
    msg << stream.sstream_->rdbuf();
    OutputLog(msg);
    return msg.str();
  }

  void operator^(const LogStream &stream) const __attribute__((noreturn, visibility("default"))) {
    std::ostringstream msg;
    msg << stream.sstream_->rdbuf();
    OutputLog(msg);
    throw std::runtime_error(msg.str());
  }

 private:
  void OutputLog(const std::ostringstream &msg) const;

  const char *file_;
  int line_;
  const char *func_;
  ERROR_LEVEL log_level_;
};

#define MSILOG_IF(level)                                                                                      \
  mindspore::serving::LogWriter(SERVING_FILE_NAME, __LINE__, __FUNCTION__, mindspore::serving::LOG_##level) < \
    mindspore::serving::LogStream()

#define MSILOG_THROW                                                                                            \
  mindspore::serving::LogWriter(SERVING_FILE_NAME, __LINE__, __FUNCTION__, mindspore::serving::LOG_EXCEPTION) ^ \
    mindspore::serving::LogStream()

#define MSI_LOG(level) MSI_LOG_##level

#define MSI_LOG_DEBUG MSILOG_IF(DEBUG)
#define MSI_LOG_INFO MSILOG_IF(INFO)
#define MSI_LOG_WARNING MSILOG_IF(WARNING)
#define MSI_LOG_ERROR MSILOG_IF(ERROR)
#define MSI_LOG_EXCEPTION MSILOG_THROW

#define MSI_EXCEPTION_IF_NULL(ptr)                                   \
  do {                                                               \
    if ((ptr) == nullptr) {                                          \
      MSI_LOG_EXCEPTION << ": The pointer[" << #ptr << "] is null."; \
    }                                                                \
  } while (0)

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_LOG_H
