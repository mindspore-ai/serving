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
#include "worker/inference/mindspore_model_wrap.h"

namespace mindspore::serving {

class MS_API AscendModelServable : public ServableBase {
 public:
  AscendModelServable() = default;
  ~AscendModelServable() override;

  Status Predict(const std::vector<TensorBasePtr> &input, std::vector<TensorBasePtr> *output) override;

  std::vector<TensorInfo> GetInputInfos() const override;
  std::vector<TensorInfo> GetOutputInfos() const override;
  uint64_t GetBatchSize() const override;
  TensorBasePtr MakeInferenceTensor(DataType data_type, const std::vector<int64_t> &shape) const override;

  Status StartServable(const std::string &servable_directory, const std::string &servable_name,
                       uint32_t version_number);
  Status InitDevice(ModelType model_type, const std::map<std::string, std::string> &other_options);
  WorkerSpec GetWorkerSpec() const override { return worker_spec_; }

 private:
  LoadServableSpec base_spec_;
  WorkerSpec worker_spec_;
  MindSporeModelWrap session_;
  std::string version_strategy_;
  bool model_loaded_ = false;

  void GetVersions(const LoadServableSpec &servable_spec, std::vector<uint64_t> *real_versions);
  Status LoadServableConfig(const LoadServableSpec &servable_spec, const std::string &version_strategy,
                            std::vector<uint64_t> *real_version_number);
  Status LoadModel(uint64_t version);
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_WORKER_ASCEND_SERVABLE_H
