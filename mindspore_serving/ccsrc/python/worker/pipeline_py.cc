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

#include "python/worker/pipeline_py.h"
#include <pybind11/embed.h>

namespace mindspore::serving {

std::shared_ptr<PyPipelineStorage> PyPipelineStorage::Instance() {
  static std::shared_ptr<PyPipelineStorage> instance;
  if (instance == nullptr) {
    instance = std::make_shared<PyPipelineStorage>();
  }
  return instance;
}

void PyPipelineStorage::Register(const PipelineSignature &pipeline) {
  MSI_LOG_INFO << "Register python pipeline " << pipeline.pipeline_name;
  auto status = PipelineStorage::Instance().RegisterPipeline(pipeline);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
  auto pipeline_name = pipeline.pipeline_name;
  size_t inputs_count = pipeline.inputs.size();
  size_t outputs_count = pipeline.outputs.size();
  pipeline_infos_[pipeline_name] = std::make_pair(inputs_count, outputs_count);
}

void PyPipelineStorage::CheckServable(const RequestSpec &request_spec) {
  auto status = PipelineStorage::Instance().CheckServable(request_spec);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "CheckServable failed: " << status.StatusMessage();
  }
}

std::vector<py::tuple> PyPipelineStorage::Run(const RequestSpec &request_spec, const std::vector<py::tuple> &args) {
  std::vector<InstanceData> inputs;
  std::vector<py::tuple> outputs;
  for (auto &arg : args) {
    auto input = PyTensor::AsInstanceData(py::cast<py::tuple>(arg));
    inputs.push_back(input);
  }
  std::vector<InstancePtr> outs;
  {
    py::gil_scoped_release release;
    auto status = PipelineStorage::Instance().Run(request_spec, inputs, &outs);
    if (status != SUCCESS || outs.size() == 0) {
      MSI_LOG_EXCEPTION << "Run failed: " << status.StatusMessage();
    }
  }
  for (auto &out : outs) {
    if (out->error_msg != SUCCESS) {
      MSI_LOG_EXCEPTION << "Run failed: " << out->error_msg.StatusMessage();
    }
    py::tuple output = PyTensor::AsNumpyTuple(out->data);
    outputs.push_back(output);
  }
  return outputs;
}

std::vector<py::tuple> PyPipelineStorage::RunAsync(const RequestSpec &request_spec,
                                                   const std::vector<py::tuple> &args) {
  std::vector<InstanceData> inputs;
  std::vector<py::tuple> outputs;
  for (auto &arg : args) {
    auto input = PyTensor::AsInstanceData(py::cast<py::tuple>(arg));
    inputs.push_back(input);
  }
  std::vector<InstancePtr> outs;
  {
    py::gil_scoped_release release;
    auto status = PipelineStorage::Instance().RunAsync(request_spec, inputs, &outs);
    if (status != SUCCESS) {
      MSI_LOG_EXCEPTION << "Run failed: " << status.StatusMessage();
    }
  }
  for (auto &out : outs) {
    if (out->error_msg != SUCCESS) {
      MSI_LOG_EXCEPTION << "Run failed: " << out->error_msg.StatusMessage();
    }
    py::tuple output = PyTensor::AsNumpyTuple(out->data);
    outputs.push_back(output);
  }
  return outputs;
}

PyPipelineStorage::PyPipelineStorage() {}

PyPipelineStorage::~PyPipelineStorage() = default;

}  // namespace mindspore::serving
