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
#include "cxx_api/graph/ms/ms_graph_impl.h"
#include <algorithm>
#include "include/api/context.h"
#include "cxx_api/factory.h"
#include "stub/graph_impl_stub.h"

namespace mindspore::api {
API_FACTORY_REG(GraphCell::GraphImpl, Ascend910, MsGraphImpl);

std::shared_ptr<GraphCell::GraphImpl> MsGraphImpl::graph_imp_stub_ = std::make_shared<GraphImplStubAdd>();

MsGraphImpl::MsGraphImpl() {}

MsGraphImpl::~MsGraphImpl() {}

Status MsGraphImpl::GetInputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                                  std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) {
  if (!graph_imp_stub_) {
    return FAILED;
  }
  return graph_imp_stub_->GetInputsInfo(names, shapes, data_types, mem_sizes);
}

Status MsGraphImpl::GetOutputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                                   std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) {
  if (!graph_imp_stub_) {
    return FAILED;
  }
  return graph_imp_stub_->GetOutputsInfo(names, shapes, data_types, mem_sizes);
}

Status MsGraphImpl::Load() { return SUCCESS; }

Status MsGraphImpl::Run(const std::vector<Buffer> &inputs, std::vector<Buffer> *outputs) {
  if (!graph_imp_stub_) {
    return FAILED;
  }
  return graph_imp_stub_->Run(inputs, outputs);
}
}  // namespace mindspore::api
