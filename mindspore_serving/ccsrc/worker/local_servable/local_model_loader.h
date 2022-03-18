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

#ifndef MINDSPORE_SERVING_WORKER_ASCEND_SERVABLE_H
#define MINDSPORE_SERVING_WORKER_ASCEND_SERVABLE_H

#include <memory>
#include <vector>
#include <string>
#include <map>

#include "common/serving_common.h"
#include "common/instance.h"
#include "common/servable.h"
#include "mindspore_serving/ccsrc/worker/model_loader_base.h"
#include "worker/inference/inference.h"

namespace mindspore::serving {
class MS_API LocalModelLoader : public DirectModelLoaderBase {
 public:
  LocalModelLoader() = default;
  ~LocalModelLoader() noexcept override;

  Status Predict(const std::vector<TensorBasePtr> &input, std::vector<TensorBasePtr> *output,
                 uint64_t subgraph) override;

  std::vector<TensorInfo> GetInputInfos(uint64_t subgraph) const override;
  std::vector<TensorInfo> GetOutputInfos(uint64_t subgraph) const override;
  uint64_t GetBatchSize() const override;
  uint64_t GetGraphNum() const override;

  Status LoadModel(const std::string &servable_directory, const std::string &servable_name, uint64_t version_number,
                   const ModelMeta &model_meta, const std::string &dec_key, const std::string &dec_mode);
  Status InitDevice(ModelType model_type);
  void Clear() override;

  std::string GetModelKey() const { return model_meta_.common_meta.model_key; }

 private:
  ServableLoadSpec base_spec_;
  ModelMeta model_meta_;
  uint64_t graph_num_ = 0;
  std::shared_ptr<InferenceBase> model_session_ = nullptr;

  bool model_loaded_ = false;

  Status LoadModel(uint64_t version, const std::string &dec_key, const std::string &dec_mode);
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_WORKER_ASCEND_SERVABLE_H
