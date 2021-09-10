/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "worker/model_loader_base.h"
#include "common/buffer_tensor.h"

namespace mindspore::serving {

Status DirectModelLoaderBase::Predict(const std::vector<InstanceData> &inputs, std::vector<ResultInstance> *outputs,
                                      uint64_t subgraph) {
  MSI_EXCEPTION_IF_NULL(outputs);
  if (subgraph >= model_info_.sub_graph_infos.size()) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Invalid input subgraph index " << subgraph << ", model info: " << model_key_
           << ", subgraph count: " << model_info_.sub_graph_infos.size();
  }
  Status status;
  std::vector<TensorBasePtr> predict_outputs;
  auto &subgraph_info = model_info_.sub_graph_infos[subgraph];
  status = PrePredict(subgraph_info, model_info_.batch_size, inputs);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Call Pre Predict failed, model info " << model_key_;
    return status;
  }
  status = Predict(subgraph_info.input_buffers, &predict_outputs, subgraph);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Predict failed, model info " << model_key_;
    return status;
  }
  status = PostPredict(subgraph_info, model_info_.batch_size, inputs, predict_outputs, outputs);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Call Post Predict failed, model info " << model_key_;
    return status;
  }
  return SUCCESS;
}

Status DirectModelLoaderBase::PrePredict(const ModelExecutorSubgraphInfo &subgraph_info, uint32_t model_batch_size,
                                         const std::vector<InstanceData> &instances) {
  auto input_batch_size = static_cast<uint32_t>(instances.size());
  if (input_batch_size == 0 || input_batch_size > model_batch_size) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
           << "Invalid input batch size " << input_batch_size << ", model batch size " << model_batch_size;
  }
  auto &input_infos = subgraph_info.input_infos;
  auto &input_buffers = subgraph_info.input_buffers;

  for (size_t i = 0; i < input_infos.size(); i++) {
    auto &tensor = input_buffers[i];
    auto data_size = tensor->data_size();
    auto dst_buffer = reinterpret_cast<uint8_t *>(tensor->mutable_data());
    if (input_infos[i].is_no_batch_dim) {
      if (data_size != instances[0][i]->data_size()) {
        return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Input " << i << " data size " << instances[0][i]->data_size()
                                                    << "does not match size " << data_size << " defined in model";
      }
      memcpy_s(dst_buffer, data_size, instances[0][i]->data(), data_size);
      continue;
    }
    auto item_size = static_cast<size_t>(data_size / model_batch_size);
    for (uint32_t k = 0; k < input_batch_size; k++) {
      if (i >= instances[k].size()) {
        return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << " Batch index " << k << " does not have input " << i;
      }
      if (item_size != instances[k][i]->data_size()) {
        return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
               << "Input " << i << " Batch index " << k << " input data size " << instances[k][i]->data_size()
               << "does not match size " << item_size << " defined in model";
      }
      memcpy_s(dst_buffer + k * item_size, data_size - k * item_size, instances[k][i]->data(), item_size);
    }
    for (uint32_t k = input_batch_size; k < model_batch_size; k++) {
      memcpy_s(dst_buffer + k * item_size, data_size - k * item_size, instances[0][i]->data(), item_size);
    }
  }
  return SUCCESS;
}

Status DirectModelLoaderBase::PostPredict(const ModelExecutorSubgraphInfo &subgraph_info, uint32_t model_batch_size,
                                          const std::vector<InstanceData> &instances,
                                          const std::vector<TensorBasePtr> &predict_result,
                                          std::vector<ResultInstance> *instance_result) {
  auto input_batch_size = static_cast<uint32_t>(instances.size());
  if (input_batch_size == 0 || input_batch_size > model_batch_size) {
    MSI_LOG_ERROR << "Input batch size " << input_batch_size << " invalid, model batch size " << model_batch_size;
    return SYSTEM_ERROR;
  }
  if (predict_result.size() != subgraph_info.output_infos.size()) {
    MSI_LOG_ERROR << "Output result count " << predict_result.size() << " not equal to outputs count "
                  << subgraph_info.output_infos.size();
    return SYSTEM_ERROR;
  }
  std::vector<ResultInstance> results_data(input_batch_size);
  auto &output = subgraph_info.output_infos;
  for (size_t i = 0; i < predict_result.size(); i++) {
    auto &item = predict_result[i];
    auto &output_info = output[i];
    if (item->data_size() != output_info.tensor_info.size) {
      MSI_LOG_ERROR << "Output result " << i << " data size " << item->data_size() << " not equal to size "
                    << output_info.tensor_info.size << " in output_infos_ ";
      return SYSTEM_ERROR;
    }
    auto item_size = output_info.size_one_batch;
    auto shape = output_info.shape_one_batch;
    auto data_type = output_info.tensor_info.data_type;
    auto src_buffer = const_cast<uint8_t *>(item->data());
    for (uint32_t k = 0; k < input_batch_size; k++) {
      auto tensor =
        std::make_shared<BufferTensorWithOwner>(item, data_type, shape, src_buffer + item_size * k, item_size, true);
      results_data[k].data.push_back(tensor);
    }
  }
  *instance_result = results_data;
  return SUCCESS;
}

Status DirectModelLoaderBase::AfterLoadModel() {
  InitModelExecuteInfo();
  return SUCCESS;
}

void DirectModelLoaderBase::InitModelExecuteInfo() {
  auto graph_num = GetGraphNum();
  model_info_.sub_graph_infos.resize(graph_num);
  model_info_.batch_size = GetBatchSize();

  for (uint64_t i = 0; i < graph_num; i++) {
    auto input_infos = GetInputInfos(i);
    auto output_infos = GetOutputInfos(i);
    auto &subgraph_info = model_info_.sub_graph_infos[i];
    subgraph_info.input_infos = input_infos;
    for (auto &item : output_infos) {
      TensorInfoOutput info;
      info.tensor_info = item;
      if (item.is_no_batch_dim) {
        info.shape_one_batch = item.shape;
        info.size_one_batch = item.size;
      } else {
        info.shape_one_batch = item.shape;
        info.shape_one_batch.erase(info.shape_one_batch.begin());
        // the batch size has been checked in WorkerExecutor
        info.size_one_batch = item.size / model_info_.batch_size;
      }
      subgraph_info.output_infos.push_back(info);
    }
    // init input buffer
    subgraph_info.input_buffers.clear();
    for (auto &input_info : subgraph_info.input_infos) {
      auto tensor = std::make_shared<Tensor>();
      tensor->set_data_type(input_info.data_type);
      tensor->set_shape(input_info.shape);
      tensor->resize_data(input_info.size);
      subgraph_info.input_buffers.push_back(tensor);
    }
  }
}

}  // namespace mindspore::serving
