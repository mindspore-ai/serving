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

#ifndef MINDSPORE_SERVING_WORKER_SERVABLE_PY_H
#define MINDSPORE_SERVING_WORKER_SERVABLE_PY_H

#include <string>
#include "common/servable.h"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "python/tensor_py.h"

namespace py = pybind11;

namespace mindspore::serving {
class MS_API PyServableRegister {
 public:
  static void RegisterMethod(const MethodSignature &method);

  static void DeclareModel(const ModelMeta &servable);
  static void DeclareDistributedModel(const ModelMeta &servable);

  static void RegisterInputOutputInfo(const std::string &model_key, size_t inputs_count, size_t outputs_count,
                                      uint64_t subgraph = 0);

  // input args: list<list>, output: tuple<tuple>
  static py::tuple Run(const std::string &model_key, const py::tuple &args, uint64_t subgraph);
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_WORKER_SERVABLE_PY_H
