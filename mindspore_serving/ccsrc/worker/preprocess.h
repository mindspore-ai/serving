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

#ifndef MINDSPORE_SERVING_PREPROCESS_H
#define MINDSPORE_SERVING_PREPROCESS_H

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <utility>
#include "common/serving_common.h"
#include "common/instance.h"

namespace mindspore::serving {

class PreprocessBase : public std::enable_shared_from_this<PreprocessBase> {
 public:
  PreprocessBase() = default;
  virtual ~PreprocessBase() = default;

  virtual Status Preprocess(const std::string &preprocess_name, const InstanceData &input, InstanceData &output) = 0;
  virtual size_t GetInputsCount(const std::string &preprocess_name) const = 0;
  virtual size_t GetOutputsCount(const std::string &preprocess_name) const = 0;
  virtual bool IsPythonPreprocess() const { return false; }
};

class MS_API PreprocessStorage {
 public:
  void Register(const std::string &preprocess_name, std::shared_ptr<PreprocessBase> preprocess);

  std::shared_ptr<PreprocessBase> GetPreprocess(const std::string &preprocess_name) const;

  static PreprocessStorage &Instance();

 private:
  std::unordered_map<std::string, std::shared_ptr<PreprocessBase>> preprocess_map_;
};

class RegPreprocess {
 public:
  RegPreprocess(const std::string &preprocess_name, std::shared_ptr<PreprocessBase> preprocess);
};

#define REGISTER_PREPROCESS(cls_name, preprocess_name) \
  static RegPreprocess g_register_preprocess_##cls_name(preprocess_name, std::make_shared<cls_name>());

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_PREPROCESS_H
