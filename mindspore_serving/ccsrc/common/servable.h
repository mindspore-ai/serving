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

enum MethodStageType {
  kMethodStageTypeNone = 0,
  kMethodStageTypePyFunction,
  kMethodStageTypeCppFunction,
  kMethodStageTypeModel,
  kMethodStageTypeReturn,
};

struct MethodStage {
  std::string method_name;
  uint64_t stage_index = 0;
  std::string stage_key;  // function name, model name
  std::string tag;
  MethodStageType stage_type;
  uint64_t subgraph = 0;                                  // when model
  std::vector<std::pair<size_t, uint64_t>> stage_inputs;  // first: input- 0, stage- 1~n, second: output index
  // will be updated when model loaded
  uint64_t batch_size = 0;
};

static const uint64_t kStageStartIndex = 1;
struct MS_API MethodSignature {
  std::string servable_name;
  std::string method_name;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;

  std::map<size_t, MethodStage> stage_map;  // stage_index, MethodStage

  void AddStageFunction(const std::string &func_name, const std::vector<std::pair<size_t, uint64_t>> &stage_inputs,
                        uint64_t batch_size = 0, const std::string &tag = "");
  void AddStageModel(const std::string &model_key, const std::vector<std::pair<size_t, uint64_t>> &stage_inputs,
                     uint64_t subgraph = 0, const std::string &tag = "");
  void SetReturn(const std::vector<std::pair<size_t, uint64_t>> &return_inputs);
  // the max stage is return, when reach max stage, all stage works done
  size_t GetStageMax() const;

 private:
  // stage index begin with 1, 0 reserve for input, include function, model, return stage
  size_t stage_index = kStageStartIndex;
};

struct ServableLoadSpec {
  std::string servable_directory;
  std::string servable_name;
  uint64_t version_number = 0;
  std::string Repr() const;
};

struct ServableMethodInfo {
  std::string name;
  std::vector<std::string> input_names;
  bool only_model_stage = false;
};

struct ModelSubgraphInfo {
  std::vector<TensorInfo> input_infos;
  std::vector<TensorInfo> output_infos;
};

struct ModelInfo {
  std::vector<ModelSubgraphInfo> sub_graph_infos;
  uint64_t batch_size = 0;
};

struct ServableRegSpec {
  std::string servable_name;
  uint64_t version_number = 0;
  uint64_t batch_size = 0;
  bool own_device = true;
  std::vector<ServableMethodInfo> methods;
  std::map<std::string, ModelInfo> models;
};

struct WorkerRegSpec {
  uint64_t worker_pid = 0;
  std::string worker_address;
  ServableRegSpec servable_spec;
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

struct CommonModelMeta {
  std::string servable_name;
  // used to identify model, for local model: ";".join(model_files), for distributed model: servable name
  std::string model_key;
  bool with_batch_dim = true;  // whether there is batch dim in model's inputs/outputs
  std::vector<int> without_batch_dim_inputs;
  std::map<uint64_t, size_t> inputs_count;
  std::map<uint64_t, size_t> outputs_count;
};

struct MS_API LocalModelMeta {
  std::vector<std::string> model_files;              // file names
  ModelType model_format = ModelType::kUnknownType;  // OM, MindIR, MindIR_Opt
  ModelContext model_context;
  std::string config_file;
  void SetModelFormat(const std::string &format);
};

struct DistributedModelMeta {
  size_t rank_size = 0;
  size_t stage_size = 0;
};

struct MS_API ModelMeta {
  CommonModelMeta common_meta;
  LocalModelMeta local_meta;
  DistributedModelMeta distributed_meta;
};

struct MS_API ServableSignature {
  ServableType servable_type = kServableTypeUnknown;
  std::string servable_name;
  std::vector<ModelMeta> model_metas;
  std::vector<MethodSignature> methods;
  const MethodSignature *GetMethodDeclare(const std::string &method_name) const;
  const ModelMeta *GetModelDeclare(const std::string &model_key) const;
};

struct WorkerAgentSpec {
  std::string agent_address;
  uint32_t rank_id = 0;
  std::vector<TensorInfo> input_infos;
  std::vector<TensorInfo> output_infos;
  uint32_t batch_size = 0;
  uint64_t subgraph = 0;
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_SERVABLE_H
