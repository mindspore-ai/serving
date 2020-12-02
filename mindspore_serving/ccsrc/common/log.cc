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
#include "common/log.h"

#include <sys/time.h>
#include "glog/logging.h"

namespace mindspore {
namespace serving {

#undef Dlog
#define Dlog(module_id, level, format, ...)                   \
  do {                                                        \
    DlogInner((module_id), (level), (format), ##__VA_ARGS__); \
  } while (0)

static std::string GetTime() {
#define BUFLEN 80
  static char buf[BUFLEN];
#if defined(_WIN32) || defined(_WIN64)
  time_t time_seconds = time(0);
  struct tm now_time;
  localtime_s(&now_time, &time_seconds);
  sprintf_s(buf, BUFLEN, "%d-%d-%d %d:%d:%d", now_time.tm_year + 1900, now_time.tm_mon + 1, now_time.tm_mday,
            now_time.tm_hour, now_time.tm_min, now_time.tm_sec);
#else
  struct timeval cur_time;
  (void)gettimeofday(&cur_time, nullptr);

  struct tm now;
  (void)localtime_r(&cur_time.tv_sec, &now);
  (void)strftime(buf, BUFLEN, "%Y-%m-%d-%H:%M:%S", &now);  // format date and time
  // set micro-second
  buf[27] = '\0';
  int idx = 26;
  auto num = cur_time.tv_usec;
  for (int i = 5; i >= 0; i--) {
    buf[idx--] = static_cast<char>(num % 10 + '0');
    num /= 10;
    if (i % 3 == 0) {
      buf[idx--] = '.';
    }
  }
#endif
  return std::string(buf);
}

static std::string GetProcName() {
#if defined(__APPLE__) || defined(__FreeBSD__)
  const char *appname = getprogname();
#elif defined(_GNU_SOURCE)
  const char *appname = program_invocation_name;
#else
  const char *appname = "?";
#endif
  // some times, the appname is an absolute path, its too long
  std::string app_name(appname);
  std::size_t pos = app_name.rfind("/");
  if (pos == std::string::npos) {
    return app_name;
  }
  if (pos + 1 >= app_name.size()) {
    return app_name;
  }
  return app_name.substr(pos + 1);
}

static std::string GetLogLevel(ERROR_LEVEL level) {
  switch (level) {
    case LOG_DEBUG:
      return "DEBUG";
    case LOG_INFO:
      return "INFO";
    case LOG_WARNING:
      return "WARNING";
    case LOG_ERROR:
    default:
      return "ERROR";
  }
}

// convert MsLogLevel to corresponding glog level
static int GetGlogLevel(ERROR_LEVEL level) {
  switch (level) {
    case LOG_DEBUG:
    case LOG_INFO:
      return google::GLOG_INFO;
    case LOG_WARNING:
      return google::GLOG_WARNING;
    case LOG_ERROR:
    default:
      return google::GLOG_ERROR;
  }
}

void LogWriter::OutputLog(const std::ostringstream &msg) const {
  auto submodule_name = "Serving";
  google::LogMessage("", 0, GetGlogLevel(log_level_)).stream()
    << "[" << GetLogLevel(log_level_) << "] " << submodule_name << "(" << getpid() << "," << GetProcName()
    << "):" << GetTime() << " "
    << "[" << file_ << ":" << line_ << "] " << func_ << "] " << msg.str() << std::endl;
}

}  // namespace serving
}  // namespace mindspore
