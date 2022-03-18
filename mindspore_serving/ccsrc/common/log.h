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
#include <utility>

namespace mindspore::serving {
#define MS_API __attribute__((visibility("default")))

#define SERVING_LOG_HDR_FILE_REL_PATH "mindspore_serving/ccsrc/common/log.h"

// Get start index of file relative path in __FILE__
static constexpr size_t GetRelPathPos() noexcept {
  return sizeof(__FILE__) > sizeof(SERVING_LOG_HDR_FILE_REL_PATH)
           ? sizeof(__FILE__) - sizeof(SERVING_LOG_HDR_FILE_REL_PATH)
           : 0;
}

#define SERVING_FILE_NAME                                                        \
  (sizeof(__FILE__) > mindspore::serving::GetRelPathPos()                        \
     ? static_cast<const char *>(__FILE__) + mindspore::serving::GetRelPathPos() \
     : static_cast<const char *>(__FILE__))

class AsStringHelper {
 public:
  template <typename T>
  static std::string AsString(const T &val) noexcept {
    std::stringstream ss;
    ss << val;
    return ss.str();
  }
  static std::string AsString(const bool &val) noexcept { return val ? "true" : "false"; }

  template <typename T>
  static std::string AsString(const std::vector<T> &val) noexcept {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < val.size(); i++) {
      ss << AsString(val[i]);
      if (i + 1 < val.size()) {
        ss << ", ";
      }
    }
    ss << "]";
    return ss.str();
  }

  template <typename K, typename V>
  static std::string AsString(const std::unordered_map<K, V> &val) noexcept {
    return AsStringMap(val);
  }
  template <typename K, typename V>
  static std::string AsString(const std::map<K, V> &val) noexcept {
    return AsStringMap(val);
  }
  template <typename K, typename V>
  static std::string AsString(const std::vector<std::pair<K, V>> &val) noexcept {
    return AsStringMap(val);
  }

 private:
  template <typename T>
  static std::string AsStringMap(const T &val) noexcept {
    std::stringstream ss;
    ss << "{";
    size_t index = 0;
    for (auto &item : val) {
      ss << AsString(item.first) << ": " << AsString(item.second);
      if (index + 1 < val.size()) {
        ss << ", ";
      }
      index += 1;
    }
    ss << "}";
    return ss.str();
  }
};

class LogStream {
 public:
  LogStream() { sstream_ = std::make_shared<std::stringstream>(); }
  ~LogStream() = default;

  template <typename T>
  LogStream &operator<<(const T &val) noexcept {
    (*sstream_) << val;
    return *this;
  }

  LogStream &operator<<(const bool &val) noexcept {
    (*sstream_) << (val ? "true" : "false");
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
    return OutputMap(val);
  }

  template <typename K, typename V>
  LogStream &operator<<(const std::map<K, V> &val) noexcept {
    return OutputMap(val);
  }

  LogStream &operator<<(std::ostream &func(std::ostream &os)) noexcept {
    (*sstream_) << func;
    return *this;
  }

  friend class LogWriter;
  friend class Status;

 private:
  std::shared_ptr<std::stringstream> sstream_;
  template <typename T>
  LogStream &OutputMap(const T &val) noexcept {
    (*sstream_) << "{";
    size_t index = 0;
    for (auto &item : val) {
      (*this) << item.first << ": " << item.second;
      if (index + 1 < val.size()) {
        (*sstream_) << ", ";
      }
      index += 1;
    }
    (*sstream_) << "}";
    return *this;
  }
};

enum MsLogLevel {
  LOG_DEBUG,
  LOG_INFO,
  LOG_WARNING,
  LOG_ERROR,
  LOG_EXCEPTION,
};

class MS_API LogWriter {
 public:
  LogWriter(const char *file, int line, const char *func, MsLogLevel log_level)
      : file_(file), line_(line), func_(func), log_level_(log_level) {}
  ~LogWriter() = default;

  std::string operator<(const LogStream &stream) const noexcept __attribute__((visibility("default"))) {
    std::ostringstream msg;
    msg << stream.sstream_->rdbuf();
    auto msg_str = GetOutputMsg(msg);
    OutputLog(msg_str);
    return msg_str;
  }

  void operator^(const LogStream &stream) const __attribute__((noreturn, visibility("default"))) {
    std::ostringstream msg;
    msg << stream.sstream_->rdbuf();
    auto msg_str = GetOutputMsg(msg);
    OutputLog(msg_str);
    throw std::runtime_error(msg_str);
  }

  std::string GetOutputMsg(const std::ostringstream &msg) const {
    std::string msg_str = msg.str();
    constexpr int max_log_size = 384;
    constexpr int msg_log_start_size = 192;
    if (msg_str.length() > max_log_size) {
      msg_str = msg_str.substr(0, msg_log_start_size) + "..." + msg_str.substr(msg_str.length() - msg_log_start_size);
    }
    return msg_str;
  }

 private:
  void OutputLog(const std::string &msg_str) const;

  const char *file_;
  int line_;
  const char *func_;
  MsLogLevel log_level_;
};

extern int g_ms_serving_log_level MS_API;

#define MSILOG_IF(level, condition)                                                       \
  !(condition) ? std::string()                                                            \
               : mindspore::serving::LogWriter(SERVING_FILE_NAME, __LINE__, __FUNCTION__, \
                                               mindspore::serving::LOG_##level) < mindspore::serving::LogStream()

#define MSILOG_NOIF(level)                                                                                    \
  mindspore::serving::LogWriter(SERVING_FILE_NAME, __LINE__, __FUNCTION__, mindspore::serving::LOG_##level) < \
    mindspore::serving::LogStream()

inline bool IS_OUTPUT_ON(enum MsLogLevel level) { return static_cast<int>(level) >= g_ms_serving_log_level; }

#define MSILOG_THROW                                                                                            \
  mindspore::serving::LogWriter(SERVING_FILE_NAME, __LINE__, __FUNCTION__, mindspore::serving::LOG_EXCEPTION) ^ \
    mindspore::serving::LogStream()

#define MSI_LOG(level) MSI_LOG_##level

#define MSI_LOG_DEBUG MSILOG_IF(DEBUG, mindspore::serving::IS_OUTPUT_ON(mindspore::serving::LOG_DEBUG))
#define MSI_LOG_INFO MSILOG_IF(INFO, mindspore::serving::IS_OUTPUT_ON(mindspore::serving::LOG_INFO))
#define MSI_LOG_WARNING MSILOG_IF(WARNING, mindspore::serving::IS_OUTPUT_ON(mindspore::serving::LOG_WARNING))
#define MSI_LOG_ERROR MSILOG_IF(ERROR, mindspore::serving::IS_OUTPUT_ON(mindspore::serving::LOG_ERROR))

#define MSI_LOG_EXCEPTION MSILOG_THROW

#define MSI_EXCEPTION_IF_NULL(ptr)                                   \
  do {                                                               \
    if ((ptr) == nullptr) {                                          \
      MSI_LOG_EXCEPTION << ": The pointer[" << #ptr << "] is null."; \
    }                                                                \
  } while (0)

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_LOG_H
