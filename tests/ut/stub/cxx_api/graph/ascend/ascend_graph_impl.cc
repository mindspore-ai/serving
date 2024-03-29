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
#include "cxx_api/graph/ascend/ascend_graph_impl.h"
#include <algorithm>
#include "include/api/context.h"
#include "cxx_api/factory.h"
#include "stub/graph_impl_stub.h"

namespace mindspore {
API_FACTORY_REG(GraphCell::GraphImpl, AscendGraphImpl);

AscendGraphImpl::AscendGraphImpl() { graph_imp_stub_ = std::make_shared<GraphImplStubAdd>(); }

AscendGraphImpl::~AscendGraphImpl() {}

std::vector<MSTensor> AscendGraphImpl::GetInputs() { return graph_imp_stub_->GetInputs(); }

std::vector<MSTensor> AscendGraphImpl::GetOutputs() { return graph_imp_stub_->GetOutputs(); }

Status AscendGraphImpl::Load(uint32_t device_id) {
  graph_imp_stub_->SetGraph(graph_);
  graph_imp_stub_->SetContext(graph_context_);
  return graph_imp_stub_->Load(device_id);
}

Status AscendGraphImpl::Run(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  return graph_imp_stub_->Run(inputs, outputs);
}

bool AscendGraphImpl::CheckDeviceSupport(mindspore::DeviceType device_type) {
  return graph_imp_stub_->CheckDeviceSupport(device_type);
}

}  // namespace mindspore
