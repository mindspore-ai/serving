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

#ifndef MINDSPORE_SERVING_WORKER_STAGE_FUNCTION_PY_H
#define MINDSPORE_SERVING_WORKER_STAGE_FUNCTION_PY_H

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <utility>
#include "common/serving_common.h"
#include "common/instance.h"

namespace mindspore::serving {

class CppStageFunctionBase : public std::enable_shared_from_this<CppStageFunctionBase> {
 public:
  CppStageFunctionBase() = default;
  virtual ~CppStageFunctionBase() = default;

  virtual Status Call(const std::string &func_name, const InstanceData &input, InstanceData *output) = 0;
  virtual size_t GetInputsCount(const std::string &func_name) const = 0;
  virtual size_t GetOutputsCount(const std::string &func_name) const = 0;
};

class CppStageFunctionStorage {
 public:
  bool Register(const std::string &func_name, std::shared_ptr<CppStageFunctionBase> function);
  void Unregister(const std::string &func_name);

  std::shared_ptr<CppStageFunctionBase> GetFunction(const std::string &func_name) const;

  static CppStageFunctionStorage &Instance();

 private:
  std::unordered_map<std::string, std::shared_ptr<CppStageFunctionBase>> function_map_;
};

class CppRegStageFunction {
 public:
  CppRegStageFunction(const std::string &func_name, std::shared_ptr<CppStageFunctionBase> function);
  ~CppRegStageFunction();

 private:
  std::string func_name_;
  bool register_success_ = false;
};

#define REGISTER_STAGE_FUNCTION(cls_name, func_name) \
  static CppRegStageFunction g_register_stage_function_##cls_name(func_name, std::make_shared<cls_name>());

class MS_API PyStageFunctionStorage {
 public:
  static std::shared_ptr<PyStageFunctionStorage> Instance();

  void Register(const std::string &func_name, size_t inputs_count, size_t outputs_count);

  bool HasPyFunction(const std::string &func_name);
  bool GetPyFunctionInfo(const std::string &func_name, size_t *inputs_count, size_t *outputs_count);

  std::vector<size_t> GetPyCppFunctionInfo(const std::string &func_name) const;

  PyStageFunctionStorage();
  ~PyStageFunctionStorage();

 private:
  std::unordered_map<std::string, std::pair<size_t, size_t>> function_infos_;
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_WORKER_STAGE_FUNCTION_PY_H
