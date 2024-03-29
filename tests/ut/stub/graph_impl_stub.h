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

#ifndef MINDSPORE_SERVING_GRAPH_IMPL_STUB_H
#define MINDSPORE_SERVING_GRAPH_IMPL_STUB_H

#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "include/api/status.h"
#include "include/api/graph.h"
#include "cxx_api/graph/graph_impl.h"
#include "cxx_api/model/model_impl.h"

namespace mindspore {
class GraphImplStubAdd : public GraphCell::GraphImpl {
 public:
  GraphImplStubAdd() = default;
  ~GraphImplStubAdd() = default;

  Status Run(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) override;
  Status Load(uint32_t device_id) override;

  std::vector<MSTensor> GetInputs() override;
  std::vector<MSTensor> GetOutputs() override;
  bool CheckDeviceSupport(mindspore::DeviceType device_type) override;
 private:
  std::vector<MSTensor> inputs_;
  std::vector<MSTensor> outputs_;
  uint64_t input_count = 2;
  uint64_t output_count = 1;
  bool sub_ = false;  // add or sub op

  void Init(const std::vector<int64_t> &add_shape);
  void LoadInner();
  Status CheckContext();
};

}  // namespace mindspore

#endif  // MINDSPORE_SERVING_GRAPH_IMPL_STUB_H
