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
#include <thread>
#include "glog/logging.h"
#include "common/utils.h"

namespace mindspore {
namespace serving {

int g_ms_serving_log_level = LOG_WARNING;

#undef Dlog
#define Dlog(module_id, level, format, ...)                   \
  do {                                                        \
    DlogInner((module_id), (level), (format), ##__VA_ARGS__); \
  } while (0)

static std::string GetTimeString() {
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
  const std::string appname = getprogname();
#elif defined(_GNU_SOURCE)
  const std::string appname = program_invocation_name;
#else
  const std::string appname = "?";
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

static std::string GetLogLevel(MsLogLevel level) {
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
static int GetGlogLevel(MsLogLevel level) {
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

// get threshold level
static int GetThresholdLevel(const std::string &threshold) {
  if (threshold.empty()) {
    return google::GLOG_WARNING;
  } else if (threshold == std::to_string(LOG_DEBUG) || threshold == std::to_string(LOG_INFO)) {
    return google::GLOG_INFO;
  } else if (threshold == std::to_string(LOG_WARNING)) {
    return google::GLOG_WARNING;
  } else if (threshold == std::to_string(LOG_ERROR)) {
    return google::GLOG_ERROR;
  } else {
    return google::GLOG_WARNING;
  }
}

void LogWriter::OutputLog(const std::string &msg_str) const {
  if (log_level_ < g_ms_serving_log_level) {
    return;
  }
  auto submodule_name = "SERVING";
  google::LogMessage("", 0, GetGlogLevel(log_level_)).stream()
    << "[" << GetLogLevel(log_level_) << "] " << submodule_name << "(" << getpid() << "," << std::hex
    << std::this_thread::get_id() << std::dec << "," << GetProcName() << "):" << GetTimeString() << " "
    << "[" << file_ << ":" << line_ << "] " << func_ << "] " << msg_str << std::endl;
}

static MsLogLevel GetGlobalLogLevel() { return static_cast<MsLogLevel>(FLAGS_v); }

enum class LogConfigToken : size_t {
  INVALID,      // indicate invalid token
  LEFT_BRACE,   // '{'
  RIGHT_BRACE,  // '}'
  VARIABLE,     // '[A-Za-z][A-Za-z0-9_]*'
  NUMBER,       // [0-9]+
  COMMA,        // ','
  COLON,        // ':'
  EOS,          // End Of String, '\0'
  NUM_LOG_CFG_TOKENS
};

static const char *g_tok_names[static_cast<size_t>(LogConfigToken::NUM_LOG_CFG_TOKENS)] = {
  "invalid",        // indicate invalid token
  "{",              // '{'
  "}",              // '}'
  "variable",       // '[A-Za-z][A-Za-z0-9_]*'
  "number",         // [0-9]+
  ",",              // ','
  ":",              // ':'
  "end-of-string",  // End Of String, '\0'
};

static inline bool IsAlpha(char ch) { return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z'); }

static inline bool IsDigit(char ch) { return ch >= '0' && ch <= '9'; }

class LogConfigLexer {
 public:
  explicit LogConfigLexer(const std::string &text) : buffer_(text) { cur_idx_ = 0; }
  ~LogConfigLexer() = default;

  // skip white space, and return the first char after white space
  char SkipWhiteSpace() {
    while (cur_idx_ < buffer_.size()) {
      char ch = buffer_[cur_idx_];
      if (ch == ' ' || ch == '\t') {
        ++cur_idx_;
        continue;
      }
      return ch;
    }
    return '\0';
  }

  LogConfigToken GetNext(std::string *const ptr) {
    char ch = SkipWhiteSpace();
    // clang-format off
    static const std::map<char, LogConfigToken> single_char_map = {
      {'{', LogConfigToken::LEFT_BRACE},
      {'}', LogConfigToken::RIGHT_BRACE},
      {',', LogConfigToken::COMMA},
      {':', LogConfigToken::COLON},
      {'\0', LogConfigToken::EOS},
    };
    // clang-format on

    auto iter = single_char_map.find(ch);
    if (iter != single_char_map.end()) {
      if (ptr != nullptr) {
        *ptr = std::string() + ch;
      }
      ++cur_idx_;
      return iter->second;
    } else if (IsAlpha(ch)) {
      std::ostringstream oss;
      do {
        oss << ch;
        ch = buffer_[++cur_idx_];
      } while (cur_idx_ < buffer_.size() && (IsAlpha(ch) || IsDigit(ch) || ch == '_'));
      if (ptr != nullptr) {
        *ptr = std::string(oss.str());
      }
      return LogConfigToken::VARIABLE;
    } else if (IsDigit(ch)) {
      std::ostringstream oss;
      do {
        oss << ch;
        ch = buffer_[++cur_idx_];
      } while (cur_idx_ < buffer_.size() && IsDigit(ch));
      if (ptr != nullptr) {
        *ptr = std::string(oss.str());
      }
      return LogConfigToken::NUMBER;
    }
    return LogConfigToken::INVALID;
  }

 private:
  std::string buffer_;
  size_t cur_idx_;
};

class LogConfigParser {
 public:
  explicit LogConfigParser(const std::string &cfg) : lexer(cfg) {}
  ~LogConfigParser() = default;

  bool Expect(LogConfigToken expected, LogConfigToken tok) const {
    if (expected != tok) {
      MSI_LOG(WARNING) << "Parse submodule log configuration text error, expect `"
                       << g_tok_names[static_cast<size_t>(expected)] << "`, but got `"
                       << g_tok_names[static_cast<size_t>(tok)] << "`. The whole configuration will be ignored.";
      return false;
    }
    return true;
  }

  // The text of config MS_SUBMODULE_LOG_v is in the form {submodule1:log_level1,submodule2:log_level2,...}.
  // Valid values of log levels are: 0 - debug, 1 - info, 2 - warning, 3 - error
  // e.g. MS_SUBMODULE_LOG_v={PARSER:0, ANALYZER:2, PIPELINE:1}
  std::map<std::string, std::string> Parse() {
    std::map<std::string, std::string> log_levels;

    bool flag_error = false;
    std::string text;
    auto tok = lexer.GetNext(&text);
    // empty string
    if (tok == LogConfigToken::EOS) {
      return log_levels;
    }

    if (!Expect(LogConfigToken::LEFT_BRACE, tok)) {
      return log_levels;
    }

    do {
      std::string key, val;
      tok = lexer.GetNext(&key);
      if (!Expect(LogConfigToken::VARIABLE, tok)) {
        flag_error = true;
        break;
      }

      tok = lexer.GetNext(&text);
      if (!Expect(LogConfigToken::COLON, tok)) {
        flag_error = true;
        break;
      }

      tok = lexer.GetNext(&val);
      if (!Expect(LogConfigToken::NUMBER, tok)) {
        flag_error = true;
        break;
      }

      log_levels[key] = val;
      tok = lexer.GetNext(&text);
    } while (tok == LogConfigToken::COMMA);

    if (!flag_error && !Expect(LogConfigToken::RIGHT_BRACE, tok)) {
      flag_error = true;
    }

    if (flag_error) {
      log_levels.clear();
    }
    return log_levels;
  }

 private:
  LogConfigLexer lexer;
};

bool ParseLogLevel(const std::string &str_level, MsLogLevel *ptr_level) {
  if (str_level.size() == 1) {
    int ch = str_level.c_str()[0];
    ch = ch - '0';  // subtract ASCII code of '0', which is 48
    if (ch >= LOG_DEBUG && ch <= LOG_ERROR) {
      if (ptr_level != nullptr) {
        *ptr_level = static_cast<MsLogLevel>(ch);
      }
      return true;
    }
  }
  return false;
}

void InitSubModulesLogLevel() {
  // initialize submodule's log level using global
  auto global_log_level = GetGlobalLogLevel();
  g_ms_serving_log_level = global_log_level;

  // set submodule's log level
  auto submodule = common::GetEnv("MS_SUBMODULE_LOG_v");
  MSI_LOG(DEBUG) << "MS_SUBMODULE_LOG_v=`" << submodule << "`";
  LogConfigParser parser(submodule);
  auto configs = parser.Parse();
  for (const auto &cfg : configs) {
    if (cfg.first == "SERVING") {
      MsLogLevel submodule_log_level;
      if (!ParseLogLevel(cfg.second, &submodule_log_level)) {
        MSI_LOG(WARNING) << "Illegal log level value " << cfg.second << " for " << cfg.first << ", ignore it.";
        continue;
      }
      g_ms_serving_log_level = submodule_log_level;
    }
  }
}

void common_log_init(void) {
  // do not use glog predefined log prefix
  FLAGS_log_prefix = false;
  // disable log buffer, real-time output
  FLAGS_logbufsecs = 0;
  // set default log level to WARNING
  if (common::GetEnv("GLOG_v").empty()) {
    FLAGS_v = mindspore::serving::LOG_WARNING;
  }

  // set default log file mode to 0640
  if (common::GetEnv("GLOG_logfile_mode").empty()) {
    FLAGS_logfile_mode = 0640;
  }
  std::string logtostderr = common::GetEnv("GLOG_logtostderr");
  // default print log to screen
  if (logtostderr.empty()) {
    FLAGS_logtostderr = true;
  } else if (logtostderr == "0" && common::GetEnv("GLOG_log_dir").empty()) {
    FLAGS_logtostderr = true;
    MSI_LOG(WARNING) << "`GLOG_log_dir` is not set, output log to screen.";
  }

  // default GLOG_stderrthreshold level to WARNING
  auto threshold = common::GetEnv("GLOG_stderrthreshold");
  FLAGS_stderrthreshold = GetThresholdLevel(threshold);

  mindspore::serving::InitSubModulesLogLevel();
}

}  // namespace serving
}  // namespace mindspore

extern "C" {
#if defined(_WIN32) || defined(_WIN64)
__attribute__((constructor)) void mindspore_serving_log_init(void) {
#else
void mindspore_serving_log_init(void) {
#endif
  static bool is_glog_inited = false;
  if (!is_glog_inited) {
#if !defined(_WIN32) && !defined(_WIN64)
    google::InitGoogleLogging("mindspore_serving");
#endif
    is_glog_inited = true;
  }
  mindspore::serving::common_log_init();
}
}
