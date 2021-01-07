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

namespace mindspore::api {
class GraphImplStubAdd : public GraphCell::GraphImpl {
 public:
  GraphImplStubAdd();
  explicit GraphImplStubAdd(const std::vector<int64_t> &add_shape);
  ~GraphImplStubAdd() override;

  Status Run(const std::vector<Buffer> &inputs, std::vector<Buffer> *outputs) override;
  Status Load() override;
  Status GetInputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                       std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) override;
  Status GetOutputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                        std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) override;

 private:
  std::vector<std::string> input_names_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<DataType> input_data_types_;
  std::vector<size_t> input_mem_sizes_;

  std::vector<std::string> output_names_;
  std::vector<std::vector<int64_t>> output_shapes_;
  std::vector<DataType> output_data_types_;
  std::vector<size_t> output_mem_sizes_;

  void Init(const std::vector<int64_t> &add_shape);
};

}  // namespace mindspore::api

#endif  // MINDSPORE_SERVING_GRAPH_IMPL_STUB_H
