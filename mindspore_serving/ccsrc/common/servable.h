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

#ifndef MINDSPORE_SERVING_SERVABLE_H
#define MINDSPORE_SERVING_SERVABLE_H

#include <utility>
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include "common/serving_common.h"
#include "worker/inference/inference.h"

namespace mindspore::serving {

enum PredictPhaseTag {
  kPredictPhaseTag_Input,
  kPredictPhaseTag_Preproces,
  kPredictPhaseTag_Predict,
  kPredictPhaseTag_Postprocess,
  kPredictPhaseTag_Output,
};

struct MethodSignature {
  std::string method_name;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;

  std::string preprocess_name;
  // the output index of the 'predict phase'
  std::vector<std::pair<PredictPhaseTag, uint64_t>> preprocess_inputs;

  std::string postprocess_name;
  std::vector<std::pair<PredictPhaseTag, uint64_t>> postprocess_inputs;

  std::string servable_name;
  std::vector<std::pair<PredictPhaseTag, uint64_t>> servable_inputs;

  std::vector<std::pair<PredictPhaseTag, uint64_t>> returns;
};

struct LoadServableSpec {
  std::string servable_directory;
  std::string servable_name;
  uint64_t version_number = 0;
  std::string Repr() const;
};

struct WorkerMethodInfo {
  std::string name;
  std::vector<std::string> input_names;
};

struct WorkerSpec {
  std::string servable_name;
  uint64_t version_number = 0;
  std::string worker_address;
  std::vector<WorkerMethodInfo> methods;
  std::string Repr() const;
};

struct RequestSpec {
  std::string servable_name;
  std::string method_name;
  uint64_t version_number = 0;  // not specified
  std::string Repr() const;
};

enum ServableType {
  kServableTypeUnknown = 0,
  kServableTypeLocal = 1,
  kServableTypeDistributed = 2,
};

struct CommonServableMeta {
  std::string servable_name;
  bool with_batch_dim = true;  // whether there is batch dim in model's inputs/outputs
  std::vector<int> without_batch_dim_inputs;
  size_t inputs_count = 0;
  size_t outputs_count = 0;
};

struct MS_API LocalServableMeta {
  std::string servable_file;                        // file name
  ModelType model_format = api::kUnknownType;       // OM, MindIR
  std::map<std::string, std::string> load_options;  // Acl options
  void SetModelFormat(const std::string &format);
};

struct DistributedServableMeta {
  size_t rank_size = 0;
  size_t stage_size = 0;
};

struct MS_API ServableMeta {
  ServableType servable_type = kServableTypeUnknown;
  CommonServableMeta common_meta;
  LocalServableMeta local_meta;
  DistributedServableMeta distributed_meta;

  std::string Repr() const;
};

struct ServableSignature {
  ServableMeta servable_meta;
  std::vector<MethodSignature> methods;

  Status Check() const;
  bool GetMethodDeclare(const std::string &method_name, MethodSignature *method);

 private:
  Status CheckPreprocessInput(const MethodSignature &method, size_t *pre) const;
  Status CheckPredictInput(const MethodSignature &method, size_t pre) const;
  Status CheckPostprocessInput(const MethodSignature &method, size_t pre, size_t *post) const;
  Status CheckReturn(const MethodSignature &method, size_t pre, size_t post) const;
};

class MS_API ServableStorage {
 public:
  void Register(const ServableSignature &def);
  Status RegisterMethod(const MethodSignature &method);

  bool GetServableDef(const std::string &model_name, ServableSignature *def) const;

  Status DeclareServable(ServableMeta servable);
  Status DeclareDistributedServable(ServableMeta servable);

  Status RegisterInputOutputInfo(const std::string &servable_name, size_t inputs_count, size_t outputs_count);
  std::vector<size_t> GetInputOutputInfo(const std::string &servable_name) const;
  void Clear();

  static ServableStorage &Instance();

 private:
  std::unordered_map<std::string, ServableSignature> servable_signatures_map_;
};

static inline LogStream &operator<<(LogStream &stream, PredictPhaseTag data_type) {
  switch (data_type) {
    case kPredictPhaseTag_Input:
      stream << "Input";
      break;
    case kPredictPhaseTag_Preproces:
      stream << "Preprocess";
      break;
    case kPredictPhaseTag_Predict:
      stream << "Predict";
      break;
    case kPredictPhaseTag_Postprocess:
      stream << "Postprocess";
      break;
    case kPredictPhaseTag_Output:
      stream << "Output";
      break;
  }
  return stream;
}

struct WorkerAgentSpec {
  std::string agent_address;
  uint32_t rank_id = 0;
  std::vector<TensorInfo> input_infos;
  std::vector<TensorInfo> output_infos;
  uint32_t batch_size = 0;
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_SERVABLE_H
