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

#ifndef MINDSPORE_SERVING_SERVABLE_REGISTER_H
#define MINDSPORE_SERVING_SERVABLE_REGISTER_H

#include <map>
#include <string>
#include <memory>
#include <utility>
#include <vector>

#include "common/servable.h"
#include "worker/model_loader_base.h"

namespace mindspore {
namespace serving {
class MS_API ServableRegister {
 public:
  static ServableRegister &Instance();
  const ServableSignature &GetServableSignature() const { return servable_signatures_; }

  // register_method
  Status RegisterMethod(const MethodSignature &method);
  // call_model
  Status RegisterInputOutputInfo(const std::string &model_key, size_t inputs_count, size_t outputs_count,
                                 uint64_t subgraph = 0);
  // declare_model
  Status DeclareModel(ModelMeta model);
  Status DeclareDistributedModel(ModelMeta model);

  static std::string GetCallModelMethodName(const std::string &model_key, uint64_t subgraph);

  Status InitOnModelsLoad(const std::map<std::string, std::shared_ptr<ModelLoaderBase>> &models);

 private:
  ServableSignature servable_signatures_;
  Status RegisterOneCallModelMethod(const std::string &model_key, uint64_t input_count, uint64_t output_count,
                                    uint64_t subgraph);
  Status CheckModels(const std::map<std::string, std::shared_ptr<ModelLoaderBase>> &models);
  Status InitCallModelMethods(const std::map<std::string, std::shared_ptr<ModelLoaderBase>> &models);
  Status CheckMethods();
  Status InitMethodBatchSize(const std::map<std::string, std::shared_ptr<ModelLoaderBase>> &models);
  Status CheckOneMethod(const MethodSignature &method);
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_SERVABLE_REGISTER_H
