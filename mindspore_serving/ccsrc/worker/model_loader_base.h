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

#ifndef MINDSPORE_SERVING_WORKER_SERVABLE_BASE_H
#define MINDSPORE_SERVING_WORKER_SERVABLE_BASE_H

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>

#include "common/serving_common.h"
#include "common/instance_data.h"
#include "common/servable.h"
#include "worker/inference/inference.h"

namespace mindspore::serving {
class ModelLoaderBase {
 public:
  ModelLoaderBase() = default;
  virtual ~ModelLoaderBase() = default;

  virtual std::vector<TensorInfo> GetInputInfos(uint64_t subgraph) const = 0;
  virtual std::vector<TensorInfo> GetOutputInfos(uint64_t subgraph) const = 0;
  virtual uint64_t GetBatchSize() const = 0;
  virtual uint64_t GetGraphNum() const = 0;
  virtual void Clear() = 0;

  virtual Status Predict(const std::vector<InstanceData> &inputs, std::vector<ResultInstance> *outputs,
                         uint64_t subgraph) = 0;
  virtual Status AfterLoadModel() = 0;
  virtual bool OwnDevice() const = 0;
};

struct ModelExecutorSubgraphInfo {
  std::vector<TensorInfo> input_infos;
  std::vector<TensorInfoOutput> output_infos;
  std::vector<TensorBasePtr> input_buffers;
};

struct ModelExecutorInfo {
  std::vector<ModelExecutorSubgraphInfo> sub_graph_infos;
  uint64_t batch_size = 0;
};

class MS_API DirectModelLoaderBase : public ModelLoaderBase {
 public:
  virtual Status Predict(const std::vector<TensorBasePtr> &input, std::vector<TensorBasePtr> *output,
                         uint64_t subgraph) = 0;

  Status Predict(const std::vector<InstanceData> &inputs, std::vector<ResultInstance> *outputs,
                 uint64_t subgraph) override;

  Status AfterLoadModel() override;
  bool OwnDevice() const override { return true; }

 private:
  std::string model_key_;
  ModelExecutorInfo model_info_;

  void InitModelExecuteInfo();
  Status PrePredict(const ModelExecutorSubgraphInfo &subgraph_info, uint64_t model_batch_size,
                    const std::vector<InstanceData> &instances);
  Status PostPredict(const ModelExecutorSubgraphInfo &subgraph_info, uint64_t model_batch_size,
                     const std::vector<InstanceData> &instances, const std::vector<TensorBasePtr> &predict_result,
                     std::vector<ResultInstance> *instance_result);
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_WORKER_SERVABLE_BASE_H
