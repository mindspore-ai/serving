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

#ifndef MINDSPORE_SERVING_REMOTE_CALL_MODEL_H
#define MINDSPORE_SERVING_REMOTE_CALL_MODEL_H
#include <map>
#include <string>
#include <vector>
#include <memory>
#include "worker/model_loader_base.h"

namespace mindspore::serving {

struct RemoteCallModelContext {
  uint32_t version_number;
  std::string model_name;
  uint64_t subgraph;
  std::vector<std::string> request_memory;
  std::vector<std::string> reply_memory;
  std::vector<TensorInfo> input_infos;
  std::vector<TensorInfoOutput> output_infos;
};

class MS_API RemoteCallModel : public ModelLoaderBase {
 public:
  static Status InitRemote(const std::string &servable_name, uint32_t version_number, const std::string &master_address,
                           std::map<std::string, std::shared_ptr<ModelLoaderBase>> *models);

  std::vector<TensorInfo> GetInputInfos(uint64_t subgraph = 0) const override;
  std::vector<TensorInfo> GetOutputInfos(uint64_t subgraph = 0) const override;
  uint64_t GetBatchSize() const override;
  uint64_t GetGraphNum() const override;
  void Clear() override;

  Status Predict(const std::vector<InstanceData> &inputs, std::vector<ResultInstance> *outputs,
                 uint64_t subgraph = 0) override;
  Status AfterLoadModel() override { return SUCCESS; }
  bool OwnDevice() const override { return false; }

 private:
  std::string model_key_;
  uint64_t batch_size_ = 0;
  std::vector<RemoteCallModelContext> subgraph_contexts_;

  Status InitModelExecuteInfo();
  Status InitModel(const std::string &model_key, uint32_t version_number, const ModelInfo &model_info);
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_REMOTE_CALL_MODEL_H
