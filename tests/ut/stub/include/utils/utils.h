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

#include <unistd.h>
#include <memory>
#include <atomic>
#include <string>
#include <vector>
#include <set>
#include <fstream>
#include "utils/log_adapter.h"

namespace mindspore {

class FuncGraph {
 public:
  explicit FuncGraph(const std::string &file_name) : file_name_(file_name) {}
  const std::string file_name_;
};
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

static inline bool DirOrFileExist(const std::string &file_path) {
  int ret = access(file_path.c_str(), 0);
  return ret != -1;
}

}  // namespace common

static inline size_t IntToSize(int i) { return static_cast<size_t>(i); }

typedef unsigned char Byte;

static inline std::unique_ptr<Byte[]> Decrypt(size_t *decrypt_len, const std::string &encrypt_data_path,
                                              const Byte *key, const size_t key_len, const std::string &dec_mode) {
  auto bytes = new Byte[10];
  return std::unique_ptr<Byte[]>(bytes);
}

static inline std::unique_ptr<Byte[]> Decrypt(size_t *decrypt_len, const Byte *model_data, const size_t data_size,
                                              const Byte *key, const size_t key_len, const std::string &dec_mode) {
  auto bytes = new Byte[10];
  return std::unique_ptr<Byte[]>(bytes);
}

static inline bool IsCipherFile(const std::string &file_path) { return false; }

static inline bool IsCipherFile(const Byte *model_data) { return false; }

static inline std::shared_ptr<FuncGraph> LoadMindIR(const std::string &file_name, bool is_lite,
                                                    const unsigned char *dec_key, const size_t key_len,
                                                    const std::string &dec_mode) {
  std::ifstream ifs(file_name);
  if (!ifs.good()) {
    MS_LOG(ERROR) << "File: " << file_name << " is not exist";
    return nullptr;
  }
  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "File: " << file_name << "open failed";
    return nullptr;
  }
  return std::make_shared<FuncGraph>(file_name);
}

static inline std::vector<std::shared_ptr<FuncGraph>> LoadMindIRs(
  const std::vector<std::string> file_names, bool is_lite = false, const unsigned char *dec_key = nullptr,
  const size_t key_len = 0, const std::string &dec_mode = std::string("AES-GCM")) {
  std::vector<std::shared_ptr<FuncGraph>> graphs;
  for (auto &file_name : file_names) {
    std::ifstream ifs(file_name);
    if (!ifs.good()) {
      MS_LOG(ERROR) << "File: " << file_name << " is not exist";
      return {};
    }
    if (!ifs.is_open()) {
      MS_LOG(ERROR) << "File: " << file_name << "open failed";
      return {};
    }
    graphs.push_back(std::make_shared<FuncGraph>(file_name));
  }
  return graphs;
}

static inline std::shared_ptr<FuncGraph> ConvertStreamToFuncGraph(const char *buf, const size_t buf_size,
                                                                  bool is_lite = false) {
  return std::make_shared<FuncGraph>("");
}

class MSTensor::Impl {
 public:
  Impl() = default;
  virtual ~Impl() = default;

  virtual const std::string &Name() const = 0;
  virtual enum DataType DataType() const = 0;
  virtual const std::vector<int64_t> &Shape() const = 0;

  virtual std::shared_ptr<const void> Data() const = 0;
  virtual void *MutableData() = 0;
  virtual size_t DataSize() const = 0;

  virtual bool IsDevice() const = 0;

  virtual std::shared_ptr<Impl> Clone() const = 0;
};

}  // namespace mindspore

#endif  // MINDSPORE_STUB_SERVING_UTILS_H
