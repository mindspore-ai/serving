/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef _MSC_VER
#include <sys/time.h>
#endif
#include <string>
#include <iomanip>
#include <sstream>
#include "utils/log_adapter.h"

namespace mindspore {
static const std::vector<std::string> sub_module_names = {
  "UNKNOWN",            // SM_UNKNOWN
  "CORE",               // SM_CORE
  "ANALYZER",           // SM_ANALYZER
  "COMMON",             // SM_COMMON
  "DEBUG",              // SM_DEBUG
  "OFFLINE_DEBUG",      // SM_OFFLINE_DEBUG
  "DEVICE",             // SM_DEVICE
  "GE_ADPT",            // SM_GE_ADPT
  "IR",                 // SM_IR
  "KERNEL",             // SM_KERNEL
  "MD",                 // SM_MD
  "ME",                 // SM_ME
  "EXPRESS",            // SM_EXPRESS
  "OPTIMIZER",          // SM_OPTIMIZER
  "PARALLEL",           // SM_PARALLEL
  "PARSER",             // SM_PARSER
  "PIPELINE",           // SM_PIPELINE
  "PRE_ACT",            // SM_PRE_ACT
  "PYNATIVE",           // SM_PYNATIVE
  "SESSION",            // SM_SESSION
  "UTILS",              // SM_UTILS
  "VM",                 // SM_VM
  "PROFILER",           // SM_PROFILER
  "PS",                 // SM_PS
  "FL",                 // SM_FL
  "LITE",               // SM_LITE
  "ARMOUR",             // SM_ARMOUR
  "HCCL_ADPT",          // SM_HCCL_ADPT
  "MINDQUANTUM",        // SM_MINDQUANTUM
  "RUNTIME_FRAMEWORK",  // SM_RUNTIME_FRAMEWORK
  "GE",                 // SM_GE
};

const std::string GetSubModuleName(SubModuleId module_id) {
  return sub_module_names[static_cast<size_t>(module_id % NUM_SUBMODUES)];
}

std::string GetTimeString() {
#if defined(_WIN32) || defined(_WIN64)
  time_t time_seconds = time(0);
  struct tm now_time;
  localtime_s(&now_time, &time_seconds);
  constexpr int base_year = 1900;
  std::stringstream ss;
  ss << now_time.tm_year + base_year << "-" << now_time.tm_mon + 1 << "-" << now_time.tm_mday << " " << now_time.tm_hour
     << ":" << now_time.tm_min << ":" << now_time.tm_sec;
  return ss.str();
#else
  constexpr auto BUFLEN = 80;
  char buf[BUFLEN] = {'\0'};
  struct timeval cur_time;
  (void)gettimeofday(&cur_time, nullptr);

  struct tm now;
  constexpr int width = 3;
  constexpr int64_t time_convert_unit = 1000;
  (void)localtime_r(&cur_time.tv_sec, &now);
  (void)strftime(buf, BUFLEN, "%Y-%m-%d-%H:%M:%S", &now);  // format date and time
  std::stringstream ss;
  ss << "." << std::setfill('0') << std::setw(width) << cur_time.tv_usec / time_convert_unit << "." << std::setfill('0')
     << std::setw(width) << cur_time.tv_usec % time_convert_unit;
  return std::string(buf) + ss.str();
#endif
}
}  // namespace mindspore
