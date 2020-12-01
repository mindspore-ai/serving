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

#ifndef MINDSPORE_SERVING_PREPROCESS_PY_H
#define MINDSPORE_SERVING_PREPROCESS_PY_H

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <utility>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "worker/preprocess.h"
#include "python/tensor_py.h"

namespace py = pybind11;

namespace mindspore::serving {

class PyPreprocess : public PreprocessBase {
 public:
  Status Preprocess(const std::string &preprocess_name, const InstanceData &input, InstanceData *output) override {
    MSI_LOG_EXCEPTION << "PyPreprocess::Preprocess not expected to be invoked, preprocess name " << preprocess_name;
  }
  size_t GetInputsCount(const std::string &preprocess_name) const override;
  size_t GetOutputsCount(const std::string &preprocess_name) const override;

  bool IsPythonPreprocess() const override { return true; }
};

class MS_API PyPreprocessStorage {
 public:
  static std::shared_ptr<PyPreprocessStorage> Instance();

  void Register(const std::string &preprocess_name, size_t inputs_count, size_t outputs_count);

  bool GetPyPreprocessInfo(const std::string &preprocess_name, size_t *inputs_count, size_t *outputs_count);

  std::vector<size_t> GetPyCppPreprocessInfo(const std::string &preprocess_name) const;

  PyPreprocessStorage();
  ~PyPreprocessStorage();

 private:
  std::unordered_map<std::string, std::pair<size_t, size_t>> preprocess_infos_;
  std::shared_ptr<PyPreprocess> py_preprocess_ = std::make_shared<PyPreprocess>();
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_PREPROCESS_PY_H
