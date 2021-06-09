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
#include "worker/sevable_base.h"
#include "worker/inference/inference.h"

namespace mindspore::serving {

class MS_API LocalModelServable : public ServableBase {
 public:
  LocalModelServable() = default;
  ~LocalModelServable() override;

  Status Predict(const std::vector<TensorBasePtr> &input, std::vector<TensorBasePtr> *output,
                 uint64_t subgraph = 0) override;

  std::vector<TensorInfo> GetInputInfos(uint64_t subgraph = 0) const override;
  std::vector<TensorInfo> GetOutputInfos(uint64_t subgraph = 0) const override;
  uint64_t GetBatchSize(uint64_t subgraph = 0) const override;

  Status StartServable(const std::string &servable_directory, const std::string &servable_name,
                       uint64_t version_number);
  Status InitDevice(ModelType model_type, const std::map<std::string, std::string> &other_options);
  std::string GetServableDirectory() const override;
  std::string GetServableName() const override;
  uint64_t GetServableVersion() const override;
  uint64_t GetConfigVersion() const override;
  uint64_t GetGraphNum() const override;
  void Clear() override;

 private:
  ServableLoadSpec base_spec_;
  std::string servable_name_;
  uint64_t running_version_number_ = 0;
  uint64_t graph_num_ = 0;
  std::shared_ptr<InferenceBase> session_ = nullptr;
  std::string version_strategy_;
  bool model_loaded_ = false;

  void GetVersions(const ServableLoadSpec &servable_spec, std::vector<uint64_t> *real_versions);
  Status LoadServableConfig(const ServableLoadSpec &servable_spec, const std::string &version_strategy,
                            std::vector<uint64_t> *real_version_number);
  Status LoadModel(uint64_t version);
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_WORKER_ASCEND_SERVABLE_H
