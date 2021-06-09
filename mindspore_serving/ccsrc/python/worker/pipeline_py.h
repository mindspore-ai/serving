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

#ifndef MINDSPORE_SERVING_PIPELINE_PY_H
#define MINDSPORE_SERVING_PIPELINE_PY_H

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <utility>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "worker/pipeline.h"
#include "python/tensor_py.h"

namespace py = pybind11;

namespace mindspore::serving {

class MS_API PyPipelineStorage {
 public:
  static std::shared_ptr<PyPipelineStorage> Instance();

  void Register(const PipelineSignature &method);
  void CheckServable(const RequestSpec &request_spec);
  py::tuple Run(const RequestSpec &request_spec, const py::tuple &args);
  py::tuple RunAsync(const RequestSpec &request_spec, const py::tuple &args);
  PyPipelineStorage();
  ~PyPipelineStorage();

 private:
  std::unordered_map<std::string, std::pair<size_t, size_t>> pipeline_infos_;
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_PIPELINE_PY_H
