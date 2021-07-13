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

#ifndef MINDSPORE_SERVING_PROTO_TENSOR_H_
#define MINDSPORE_SERVING_PROTO_TENSOR_H_

#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <map>
#include "common/serving_common.h"
#include "proto/ms_service.pb.h"
#include "proto/ms_master.pb.h"
#include "proto/ms_distributed.pb.h"
#include "common/instance.h"
#include "common/servable.h"
#include "common/shared_memory.h"

namespace mindspore::serving {

class MS_API ProtoTensor : public TensorBase {
 public:
  // the other's lifetime must longer than this object
  explicit ProtoTensor(proto::Tensor *other);
  ~ProtoTensor();

  DataType data_type() const override;
  void set_data_type(DataType type) override;
  std::vector<int64_t> shape() const override;
  void set_shape(const std::vector<int64_t> &shape) override;
  const uint8_t *data() const override;
  size_t data_size() const override;
  bool resize_data(size_t data_len) override;
  uint8_t *mutable_data() override;

  void clear_bytes_data() override;
  void add_bytes_data(const uint8_t *data, size_t bytes_len) override;
  size_t bytes_data_size() const override;
  void get_bytes_data(size_t index, const uint8_t **data, size_t *bytes_len) const override;

  static proto::DataType TransDataType2Proto(DataType data_type);
  static DataType TransDataType2Inference(proto::DataType data_type);

  void SetSharedMemory(const proto::ShmTensorData &shm_data_proto);
  Status AttachSharedMemory() const;

 private:
  // if tensor_ is reference from other ms_serving::Tensor, the other's lifetime must
  // longer than this object
  proto::Tensor *tensor_;
  mutable bool has_attached_shm_ = false;
  mutable SharedMemoryAttachItem shm_attach_;
};

class MS_API GrpcTensorHelper {
 public:
  static void GetRequestSpec(const proto::PredictRequest &request, RequestSpec *request_spec);
  static void ConvertProtoWorkerSpec(const proto::RegisterRequest &proto_request, WorkerRegSpec *worker_spec);
  static void ConvertWorkerSpec(const WorkerRegSpec &worker_spec, proto::RegisterRequest *proto_request);
  static void ConvertProtoModelInfos(const proto::ModelInfos &proto_model_infos,
                                     std::map<std::string, ModelInfo> *model_infos);
  static void ConvertModelInfos(const std::map<std::string, ModelInfo> &model_infos,
                                proto::ModelInfos *proto_model_infos);
  static Status CreateInstanceFromRequest(const MethodSignature &method, const proto::PredictRequest &request,
                                          std::vector<InstanceData> *results);
  static void CreateReplyFromInstances(const proto::PredictRequest &request, const MethodSignature &method,
                                       const std::vector<InstancePtr> &instances, proto::PredictReply *reply);
  static void CreateReplyFromErrorMsg(const Status &error_msg, proto::PredictReply *reply);
  static void CopyFromAgentSpec(const proto::AgentSpec &request, WorkerAgentSpec *worker_specs);
  static void CopyFromWorkerAgentSpec(const std::vector<WorkerAgentSpec> &worker_specs,
                                      proto::AgentRegisterRequest *request);
  static Status CreatePredictRequestFromInstances(const RequestSpec &request_spec,
                                                  const std::vector<const proto::Instance *> &instances,
                                                  proto::PredictRequest *request);
  static Status CreatePredictReplyFromInstances(const proto::PredictRequest &request,
                                                const std::vector<proto::ErrorMsg> &errors,
                                                const std::vector<const proto::Instance *> &instances,
                                                proto::PredictReply *reply);
  static Status CreateInstanceFromPredictReply(const RequestSpec &request_spec, const proto::PredictReply &reply,
                                               std::vector<proto::ErrorMsg> *error,
                                               std::vector<const proto::Instance *> *results);

  static Status CheckRequestInstances(const proto::PredictRequest &request,
                                      const std::vector<std::string> &input_names);

 private:
  static Status CreateInstanceFromRequestInstances(const proto::PredictRequest &request, const MethodSignature &method,
                                                   std::vector<InstanceData> *results);
  static Status CheckRequestTensor(const proto::Tensor &tensor);
  static Status CreateReplyFromInstancesInner(const proto::PredictRequest &request, const MethodSignature &method,
                                              const std::vector<InstancePtr> &instances, proto::PredictReply *reply);
};

extern MS_API LogStream &operator<<(serving::LogStream &stream, proto::DataType data_type);

}  // namespace mindspore::serving
#endif  // MINDSPORE_SERVING_PROTO_TENSOR_H_
