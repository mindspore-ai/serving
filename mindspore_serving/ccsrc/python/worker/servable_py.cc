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
#include "python/worker/servable_py.h"
#include <string>
#include <sstream>
#include <vector>
#include "worker/servable_register.h"
#include "worker/worker.h"

namespace mindspore::serving {
void PyServableRegister::RegisterMethod(const MethodSignature &method) {
  auto status = ServableRegister::Instance().RegisterMethod(method);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
}
void PyServableRegister::DeclareModel(const ModelMeta &servable) {
  auto status = ServableRegister::Instance().DeclareModel(servable);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
}
void PyServableRegister::DeclareDistributedModel(const ModelMeta &servable) {
  auto status = ServableRegister::Instance().DeclareDistributedModel(servable);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
}
void PyServableRegister::RegisterInputOutputInfo(const std::string &model_key, size_t inputs_count,
                                                 size_t outputs_count, uint64_t subgraph) {
  auto status = ServableRegister::Instance().RegisterInputOutputInfo(model_key, inputs_count, outputs_count, subgraph);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
}

py::tuple PyServableRegister::Run(const std::string &model_key, const py::tuple &args, uint64_t subgraph) {
  std::stringstream model_stream;
  if (subgraph == 0) {
    model_stream << "Model(" << model_key << ").call()";
  } else {
    model_stream << "Model(" << model_key << ", subgraph=" << subgraph << ").call()";
  }
  const std::string model_str = model_stream.str();
  RequestSpec request;
  auto const &signature = ServableRegister::Instance().GetServableSignature();
  auto model_meta = signature.GetModelDeclare(model_key);
  if (model_meta == nullptr) {
    MSI_LOG_EXCEPTION << model_str
                      << " failed: the model is not declared, ensure that interface 'declare_model' can take effect "
                         "when importing servable_config.py by the serving server";
  }
  auto &common_meta = model_meta->common_meta;
  auto input_it = common_meta.inputs_count.find(subgraph);
  if (input_it == common_meta.inputs_count.end()) {
    MSI_LOG_EXCEPTION << model_str << " failed: The model does not have subgraph of index " << subgraph
                      << ", the subgraph count of the model is " << common_meta.inputs_count.size();
  }
  auto input_count = input_it->second;

  request.servable_name = ServableRegister::Instance().GetServableSignature().servable_name;
  request.method_name = ServableRegister::Instance().GetCallModelMethodName(model_key, subgraph);

  std::vector<InstanceData> inputs;
  auto inputs_args = py::cast<py::tuple>(args);
  for (size_t i = 0; i < inputs_args.size(); i++) {
    auto input = PyTensor::AsInstanceData(py::cast<py::tuple>(inputs_args[i]));
    if (input.size() != input_count) {
      MSI_LOG_EXCEPTION << model_str << " failed: The inputs count " << input.size() << " of instance " << i
                        << " is not equal to the inputs count " << input_count << " of the model";
    }
    inputs.push_back(input);
  }
  std::vector<InstancePtr> outs;
  {
    py::gil_scoped_release release;
    auto status = Worker::GetInstance().Run(request, inputs, &outs);
    if (status != SUCCESS || outs.size() == 0) {
      MSI_LOG_EXCEPTION << model_str << " failed: " << status.StatusMessage();
    }
  }
  py::tuple outputs(outs.size());
  for (size_t i = 0; i < outs.size(); i++) {
    auto &out = outs[i];
    if (out->error_msg != SUCCESS) {
      MSI_LOG_EXCEPTION << model_str << " failed: " << out->error_msg.StatusMessage();
    }
    outputs[i] = PyTensor::AsNumpyTuple(out->data);
  }
  return outputs;
}
}  // namespace mindspore::serving
