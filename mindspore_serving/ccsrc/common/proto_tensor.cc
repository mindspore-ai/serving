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

#include "common/proto_tensor.h"
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include "common/buffer_tensor.h"
#include "common/servable.h"
#include "master/dispacther.h"
#include "common/shared_memory.h"

using std::string;
using std::unordered_map;
using std::vector;

namespace mindspore::serving {

const size_t kMaxShapeElementCount = INT32_MAX;

ProtoTensor::ProtoTensor(proto::Tensor *other) : tensor_(other) {}

ProtoTensor::~ProtoTensor() {}

DataType ProtoTensor::data_type() const {
  MSI_EXCEPTION_IF_NULL(tensor_);
  return TransDataType2Inference(tensor_->dtype());
}

void ProtoTensor::set_data_type(DataType data_type) {
  MSI_EXCEPTION_IF_NULL(tensor_);
  tensor_->set_dtype(TransDataType2Proto(data_type));
}

std::vector<int64_t> ProtoTensor::shape() const {
  MSI_EXCEPTION_IF_NULL(tensor_);
  std::vector<int64_t> result;
  auto dims = tensor_->shape().dims();
  std::transform(dims.begin(), dims.end(), std::back_inserter(result), [](const int64_t dim) { return dim; });
  return result;
}

void ProtoTensor::set_shape(const std::vector<int64_t> &shape) {
  MSI_EXCEPTION_IF_NULL(tensor_);
  auto tensor_shape = tensor_->mutable_shape();
  tensor_shape->Clear();
  size_t element_count = 1;
  for (auto dim : shape) {
    if (dim < 0 || (dim > 0 && element_count > kMaxShapeElementCount / dim)) {
      MSI_LOG_ERROR << "failed to set shape, invalid dim num " << dim;
      tensor_shape->Clear();
      return;
    }
    element_count *= dim;
    tensor_shape->add_dims(dim);
  }
}

bool ProtoTensor::resize_data(size_t data_len) {
  MSI_EXCEPTION_IF_NULL(tensor_);
  if (tensor_->has_shm_data()) {
    if (data_len == tensor_->shm_data().data_size()) {
      return true;
    }
    MSI_LOG_EXCEPTION << "Cannot resize shared memory data size from " << tensor_->shm_data().data_size() << " to "
                      << data_len;
  }
  string *buffer = tensor_->mutable_data();
  if (buffer == nullptr) {
    MSI_LOG_ERROR << "invalid buffer data";
    return false;
  }
  buffer->resize(data_len);
  return true;
}

size_t ProtoTensor::data_size() const {
  MSI_EXCEPTION_IF_NULL(tensor_);
  if (tensor_->has_shm_data()) {
    return tensor_->shm_data().data_size();
  }
  return tensor_->data().size();
}

uint8_t *ProtoTensor::mutable_data() {
  MSI_EXCEPTION_IF_NULL(tensor_);
  if (data_size() == 0) {
    return nullptr;
  }
  if (tensor_->has_shm_data()) {
    auto status = AttachSharedMemory();
    if (status != SUCCESS) {
      return nullptr;
    }
    return shm_attach_.offset_address;
  }
  return reinterpret_cast<uint8_t *>(tensor_->mutable_data()->data());
}

const uint8_t *ProtoTensor::data() const {
  MSI_EXCEPTION_IF_NULL(tensor_);
  if (data_size() == 0) {
    return nullptr;
  }
  if (tensor_->has_shm_data()) {
    auto status = AttachSharedMemory();
    if (status != SUCCESS) {
      return nullptr;
    }
    return shm_attach_.offset_address;
  }
  return reinterpret_cast<const uint8_t *>(tensor_->data().data());
}

void ProtoTensor::clear_bytes_data() {
  MSI_EXCEPTION_IF_NULL(tensor_);
  return tensor_->mutable_bytes_val()->Clear();
}

void ProtoTensor::add_bytes_data(const uint8_t *data, size_t bytes_len) {
  MSI_EXCEPTION_IF_NULL(tensor_);
  tensor_->add_bytes_val(data, bytes_len);
}

size_t ProtoTensor::bytes_data_size() const {
  MSI_EXCEPTION_IF_NULL(tensor_);
  return tensor_->bytes_val().size();
}

void ProtoTensor::get_bytes_data(size_t index, const uint8_t **data, size_t *bytes_len) const {
  MSI_EXCEPTION_IF_NULL(data);
  MSI_EXCEPTION_IF_NULL(bytes_len);
  MSI_EXCEPTION_IF_NULL(tensor_);
  if (index >= static_cast<size_t>(tensor_->bytes_val().size())) {
    MSI_LOG_EXCEPTION << "visit invalid index " << index << " total size " << tensor_->bytes_val().size();
  }
  auto &bytes = tensor_->bytes_val(index);
  *data = reinterpret_cast<const uint8_t *>(bytes.data());
  *bytes_len = bytes.size();
}

proto::DataType ProtoTensor::TransDataType2Proto(DataType data_type) {
  const std::unordered_map<DataType, proto::DataType> id2type_map{
    {serving::kMSI_Unknown, proto::MS_UNKNOWN}, {serving::kMSI_Bool, proto::MS_BOOL},
    {serving::kMSI_Float64, proto::MS_FLOAT64}, {serving::kMSI_Int8, proto::MS_INT8},
    {serving::kMSI_Uint8, proto::MS_UINT8},     {serving::kMSI_Int16, proto::MS_INT16},
    {serving::kMSI_Uint16, proto::MS_UINT16},   {serving::kMSI_Int32, proto::MS_INT32},
    {serving::kMSI_Uint32, proto::MS_UINT32},   {serving::kMSI_Int64, proto::MS_INT64},
    {serving::kMSI_Uint64, proto::MS_UINT64},   {serving::kMSI_Float16, proto::MS_FLOAT16},
    {serving::kMSI_Float32, proto::MS_FLOAT32}, {serving::kMSI_String, proto::MS_STRING},
    {serving::kMSI_Bytes, proto::MS_BYTES},
  };
  auto it = id2type_map.find(data_type);
  if (it == id2type_map.end()) {
    MSI_LOG_WARNING << "failed to set data type, undefined data type " << data_type;
    return proto::MS_UNKNOWN;
  } else {
    return it->second;
  }
}

DataType ProtoTensor::TransDataType2Inference(proto::DataType data_type) {
  const std::unordered_map<proto::DataType, DataType> type2id_map{
    {proto::MS_UNKNOWN, kMSI_Unknown}, {proto::MS_BOOL, kMSI_Bool},       {proto::MS_INT8, kMSI_Int8},
    {proto::MS_UINT8, kMSI_Uint8},     {proto::MS_INT16, kMSI_Int16},     {proto::MS_UINT16, kMSI_Uint16},
    {proto::MS_INT32, kMSI_Int32},     {proto::MS_UINT32, kMSI_Uint32},   {proto::MS_INT64, kMSI_Int64},
    {proto::MS_UINT64, kMSI_Uint64},   {proto::MS_FLOAT16, kMSI_Float16}, {proto::MS_FLOAT32, kMSI_Float32},
    {proto::MS_FLOAT64, kMSI_Float64}, {proto::MS_STRING, kMSI_String},   {proto::MS_BYTES, kMSI_Bytes},
  };
  auto it = type2id_map.find(data_type);
  if (it == type2id_map.end()) {
    MSI_LOG_WARNING << "failed to get data type, undefined data type " << data_type;
    return kMSI_Unknown;
  } else {
    return it->second;
  }
}

void ProtoTensor::SetSharedMemory(const proto::ShmTensorData &shm_data) { *tensor_->mutable_shm_data() = shm_data; }

Status ProtoTensor::AttachSharedMemory() const {
  if (has_attached_shm_) {
    return SUCCESS;
  }
  if (tensor_ == nullptr) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "The proto tensor object cannot be nullptr";
  }
  if (!tensor_->has_shm_data()) {
    return SUCCESS;
  }
  const proto::ShmTensorData &shm_data = tensor_->shm_data();
  auto status = SharedMemoryManager::Instance().Attach(shm_data.memory_key(), shm_data.bytes_size(),
                                                       shm_data.data_offset(), shm_data.data_size(), &shm_attach_);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Attach shared memory failed, memory key: " << shm_data.memory_key()
                  << ", bytes size: " << shm_data.bytes_size() << ", data offset: " << shm_data.data_offset()
                  << ", data size: " << shm_data.data_size();
    return status;
  }
  has_attached_shm_ = true;
  return SUCCESS;
}

void GrpcTensorHelper::GetRequestSpec(const proto::PredictRequest &request, RequestSpec *request_spec) {
  MSI_EXCEPTION_IF_NULL(request_spec);
  request_spec->servable_name = request.servable_spec().name();
  request_spec->method_name = request.servable_spec().method_name();
  request_spec->version_number = request.servable_spec().version_number();
}

void GrpcTensorHelper::ConvertProtoWorkerSpec(const proto::RegisterRequest &proto_request, WorkerRegSpec *worker_spec) {
  MSI_EXCEPTION_IF_NULL(worker_spec);
  auto &proto_worker_spec = proto_request.worker_spec();
  worker_spec->worker_address = proto_worker_spec.address();
  worker_spec->worker_pid = proto_worker_spec.worker_pid();

  auto &proto_spec = proto_worker_spec.servable_spec();
  auto &servable_spec = worker_spec->servable_spec;
  servable_spec.servable_name = proto_spec.name();
  servable_spec.version_number = proto_spec.version_number();
  servable_spec.batch_size = proto_spec.batch_size();
  servable_spec.own_device = proto_spec.own_device();
  for (const auto &proto_method : proto_spec.methods()) {
    ServableMethodInfo method_info;
    method_info.name = proto_method.name();
    method_info.only_model_stage = proto_method.only_model_stage();
    for (auto &name : proto_method.input_names()) {
      method_info.input_names.push_back(name);
    }
    servable_spec.methods.push_back(method_info);
  }
  ConvertProtoModelInfos(proto_spec.model_infos(), &servable_spec.models);
}

void GrpcTensorHelper::ConvertWorkerSpec(const WorkerRegSpec &worker_spec, proto::RegisterRequest *proto_request) {
  auto proto_worker_spec = proto_request->mutable_worker_spec();
  proto_worker_spec->set_address(worker_spec.worker_address);
  proto_worker_spec->set_worker_pid(worker_spec.worker_pid);

  auto proto_spec = proto_worker_spec->mutable_servable_spec();
  const auto &spec = worker_spec.servable_spec;
  proto_spec->set_name(spec.servable_name);
  proto_spec->set_version_number(spec.version_number);
  proto_spec->set_batch_size(spec.batch_size);
  proto_spec->set_own_device(spec.own_device);
  for (auto &method : spec.methods) {
    auto proto_method = proto_spec->add_methods();
    proto_method->set_name(method.name);
    proto_method->set_only_model_stage(method.only_model_stage);
    for (auto &name : method.input_names) {
      proto_method->add_input_names(name);
    }
  }
  ConvertModelInfos(spec.models, proto_spec->mutable_model_infos());
}

void GrpcTensorHelper::ConvertProtoModelInfos(const proto::ModelInfos &proto_model_infos,
                                              std::map<std::string, ModelInfo> *model_infos) {
  MSI_EXCEPTION_IF_NULL(model_infos);
  model_infos->clear();
  auto convert_tensor_info = [](const proto::TensorInfo &proto_tensor_info) -> TensorInfo {
    TensorInfo tensor_info;
    tensor_info.is_no_batch_dim = proto_tensor_info.is_no_batch_dim();
    tensor_info.size = proto_tensor_info.size();
    tensor_info.data_type = ProtoTensor::TransDataType2Inference(proto_tensor_info.dtype());
    auto &proto_shape = proto_tensor_info.shape().dims();
    std::copy(proto_shape.begin(), proto_shape.end(), std::back_inserter(tensor_info.shape));
    return tensor_info;
  };
  for (const auto &proto_model_it : proto_model_infos.model_infos()) {
    auto &model_key = proto_model_it.first;
    auto &proto_model = proto_model_it.second;
    ModelInfo &model_info = (*model_infos)[model_key];
    model_info.batch_size = proto_model.batch_size();
    for (auto &proto_subgraph : proto_model.subgraph_infos()) {
      ModelSubgraphInfo subgraph_info;
      for (auto &input_tensor : proto_subgraph.inputs()) {
        subgraph_info.input_infos.push_back(convert_tensor_info(input_tensor));
      }
      for (auto &output_tensor : proto_subgraph.outputs()) {
        subgraph_info.output_infos.push_back(convert_tensor_info(output_tensor));
      }
      model_info.sub_graph_infos.push_back(subgraph_info);
    }
  }
}

void GrpcTensorHelper::ConvertModelInfos(const std::map<std::string, ModelInfo> &model_infos,
                                         proto::ModelInfos *proto_model_infos) {
  MSI_EXCEPTION_IF_NULL(proto_model_infos);
  proto_model_infos->Clear();
  auto convert_tensor_info = [](const TensorInfo &tensor_info, proto::TensorInfo *proto_tensor_info) {
    proto_tensor_info->set_is_no_batch_dim(tensor_info.is_no_batch_dim);
    proto_tensor_info->set_size(tensor_info.size);
    proto_tensor_info->set_dtype(ProtoTensor::TransDataType2Proto(tensor_info.data_type));
    auto proto_shape = proto_tensor_info->mutable_shape()->mutable_dims();
    for (auto &dim : tensor_info.shape) {
      proto_shape->Add(dim);
    }
  };
  auto &proto_models_items = *(proto_model_infos->mutable_model_infos());
  for (const auto &model_it : model_infos) {
    auto &model_key = model_it.first;
    auto &model_info = model_it.second;
    auto &proto_model = proto_models_items[model_key];
    proto_model.set_batch_size(model_info.batch_size);
    for (auto &subgraph_info : model_info.sub_graph_infos) {
      auto proto_subgraph = proto_model.add_subgraph_infos();
      for (auto &input_tensor : subgraph_info.input_infos) {
        convert_tensor_info(input_tensor, proto_subgraph->add_inputs());
      }
      for (auto &output_tensor : subgraph_info.output_infos) {
        convert_tensor_info(output_tensor, proto_subgraph->add_outputs());
      }
    }
  }
}

Status GrpcTensorHelper::CreateInstanceFromRequest(const MethodSignature &method, const proto::PredictRequest &request,
                                                   vector<InstanceData> *results) {
  MSI_EXCEPTION_IF_NULL(results);
  results->clear();

  Status status;
  if (request.instances_size() == 0) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
           << "Instances count of request cannot be 0, servable: " << method.servable_name
           << ", method: " << method.method_name;
  }
  status = CreateInstanceFromRequestInstances(request, method, results);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Create instances from request instances failed";
    return status;
  }
  return SUCCESS;
}

void GrpcTensorHelper::CreateReplyFromInstances(const proto::PredictRequest &request, const MethodSignature &method,
                                                const vector<InstancePtr> &instances, proto::PredictReply *reply) {
  auto status = CreateReplyFromInstancesInner(request, method, instances, reply);
  if (status != SUCCESS) {
    CreateReplyFromErrorMsg(status, reply);
  }
}

Status GrpcTensorHelper::CreateInstanceFromPredictReply(const RequestSpec &request_spec,
                                                        const proto::PredictReply &reply,
                                                        std::vector<proto::ErrorMsg> *error,
                                                        std::vector<const proto::Instance *> *results) {
  MSI_EXCEPTION_IF_NULL(error);
  MSI_EXCEPTION_IF_NULL(results);
  results->clear();
  error->clear();
  if (reply.instances_size() == 0 && reply.error_msg_size() == 0) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
           << "The instance or error count of reply cannot be 0, servable: " << request_spec.servable_name
           << ", method: " << request_spec.method_name;
  }
  std::copy(reply.error_msg().begin(), reply.error_msg().end(), std::back_inserter(*error));
  for (auto &item : reply.instances()) {
    // cppcheck-suppress useStlAlgorithm
    results->push_back(&item);
  }
  return SUCCESS;
}

Status GrpcTensorHelper::CreatePredictReplyFromInstances(const proto::PredictRequest &request,
                                                         const std::vector<proto::ErrorMsg> &errors,
                                                         const std::vector<const proto::Instance *> &instances,
                                                         proto::PredictReply *reply) {
  MSI_EXCEPTION_IF_NULL(reply);
  for (auto &instance : instances) {
    auto proto_instance = reply->add_instances();
    if (instance) {
      *proto_instance->mutable_items() = instance->items();
    }
  }
  bool all_ok = true;
  bool all_same = true;
  for (auto &error : errors) {
    if (error.error_code() != 0) {
      all_ok = false;
    }
    if (error.error_code() != errors[0].error_code() || error.error_msg() != errors[0].error_msg()) {
      all_same = false;
    }
  }
  if (!all_ok) {
    if (all_same) {
      reply->clear_instances();
      auto proto_error = reply->add_error_msg();
      proto_error->set_error_msg(errors[0].error_msg());
      proto_error->set_error_code(errors[0].error_code());
    } else {
      for (auto &error : errors) {
        auto proto_error = reply->add_error_msg();
        proto_error->set_error_msg(error.error_msg());
        proto_error->set_error_code(error.error_code());
      }
    }
  }
  return SUCCESS;
}

Status GrpcTensorHelper::CreatePredictRequestFromInstances(const RequestSpec &request_spec,
                                                           const std::vector<const proto::Instance *> &instances,
                                                           proto::PredictRequest *request) {
  MSI_EXCEPTION_IF_NULL(request);
  auto proto_spec = request->mutable_servable_spec();
  proto_spec->set_name(request_spec.servable_name);
  proto_spec->set_method_name(request_spec.method_name);
  proto_spec->set_version_number(request_spec.version_number);
  for (auto &instance : instances) {
    auto proto_instance = request->add_instances();
    *proto_instance = *instance;
  }
  return SUCCESS;
}

Status GrpcTensorHelper::CreateReplyFromInstancesInner(const proto::PredictRequest &request,
                                                       const MethodSignature &method,
                                                       const std::vector<InstancePtr> &instances,
                                                       proto::PredictReply *reply) {
  MSI_EXCEPTION_IF_NULL(reply);
  if (instances.empty()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Result instances count invalid, cannot be 0";
  }
  if (instances.size() != static_cast<size_t>(request.instances_size())) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
           << "Result instances number " << instances.size() << " is inconsistent with request instances number "
           << request.instances_size();
  }
  Status status;
  size_t err_cnt = 0;
  for (auto &instance : instances) {
    if (instance->error_msg != SUCCESS) {
      err_cnt++;
    } else if (instance->data.size() != method.outputs.size()) {
      return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
             << "Result data tensor size " << instance->data.size() << " not equal outputs size "
             << method.outputs.size() << " defined in method signature";
    }
  }
  if (err_cnt > 0) {
    for (auto &instance : instances) {
      auto proto_err_msg = reply->add_error_msg();
      proto_err_msg->set_error_code(instance->error_msg.StatusCode());
      if (instance->error_msg == INVALID_INPUTS) {
        proto_err_msg->set_error_msg(instance->error_msg.StatusMessage());
      } else if (instance->error_msg != SUCCESS) {
        proto_err_msg->set_error_msg(instance->error_msg.StatusMessage());
      }
    }
  }
  // create instance reply, same with request
  for (size_t index = 0; index < instances.size(); index++) {
    auto proto_instance = reply->add_instances();
    auto &instance = instances[index];
    if (instance->data.empty()) {
      continue;
    }
    auto &request_output_buffers = request.instances(index).output_buffers();
    auto proto_items = proto_instance->mutable_items();
    for (size_t i = 0; i < method.outputs.size(); i++) {
      auto &output_tensor = instance->data[i];
      auto &output_name = method.outputs[i];

      auto &proto_tensor = (*proto_items)[method.outputs[i]];
      ProtoTensor result_tensor(&proto_tensor);

      auto it = request_output_buffers.find(output_name);
      if (it != request_output_buffers.end()) {
        if (output_tensor->is_bytes_val_data()) {
          return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
                 << "The output shared memory cannot be specified in the request"
                 << " when the data type of output " << output_name << " is " << output_tensor->data_type()
                 << ", output name: " << output_name;
        }
        auto &shm_data = it->second;
        if (shm_data.data_size() != output_tensor->data_size()) {
          return INFER_STATUS_LOG_ERROR(FAILED)
                 << "The data size " << shm_data.data_size() << " of output shared memory "
                 << " is inconsistent with the data size " << output_tensor->data_size()
                 << " of result, output name: " << output_name;
        }
        result_tensor.SetSharedMemory(shm_data);
        status = result_tensor.AttachSharedMemory();
        if (status != SUCCESS) {
          return INFER_STATUS_LOG_ERROR(FAILED)
                 << "Attach output shared memory failed, memory key: " << shm_data.memory_key()
                 << ", bytes size: " << shm_data.bytes_size() << ", data offset: " << shm_data.data_offset()
                 << ", data size: " << shm_data.data_size() << ", output name: " << output_name;
        }
      }
      result_tensor.assign(*output_tensor);
    }
  }
  return SUCCESS;
}

Status GrpcTensorHelper::CreateInstanceFromRequestInstances(const proto::PredictRequest &request,
                                                            const MethodSignature &method,
                                                            std::vector<InstanceData> *results) {
  MSI_EXCEPTION_IF_NULL(results);
  auto servable_name = request.servable_spec().name();
  auto method_name = request.servable_spec().method_name();
  Status status;
  auto &input_names = method.inputs;
  auto &output_names = method.outputs;
  for (auto &proto_instance : request.instances()) {
    InstanceData instance_data;
    for (const auto &input_name : input_names) {
      auto it = proto_instance.items().find(input_name);
      if (it == proto_instance.items().end()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
               << "Cannot find input " << input_name << " in instance input , servable " << servable_name << ", method "
               << method_name;
      }
      auto &tensor_proto = it->second;
      status = CheckRequestTensor(tensor_proto);
      if (status != SUCCESS) {
        auto status2 = INFER_STATUS(INVALID_INPUTS) << "Instances input " << input_name << " check failed";
        MSI_LOG_ERROR << status2.StatusMessage();
        return Status(INVALID_INPUTS, status2.StatusMessage() + ", detail: " + status.StatusMessage());
      }
      auto add_tensor = std::make_shared<ProtoTensor>(const_cast<proto::Tensor *>(&tensor_proto));
      if (tensor_proto.has_shm_data()) {
        status = add_tensor->AttachSharedMemory();
        if (status != SUCCESS) {
          auto &shm_data = tensor_proto.shm_data();
          MSI_LOG_ERROR << "Attach input shared memory failed, memory key: " << shm_data.memory_key()
                        << ", bytes size: " << shm_data.bytes_size() << ", data offset: " << shm_data.data_offset()
                        << ", data size: " << shm_data.data_size() << ", input name: " << input_name;
          return status;
        }
      }
      instance_data.push_back(add_tensor);
    }
    auto &output_buffers = proto_instance.output_buffers();
    if (!output_buffers.empty()) {
      for (auto &buffer : output_buffers) {
        auto it = std::find(output_names.begin(), output_names.end(), buffer.first);
        if (it == output_names.end()) {
          return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
                 << "The name " << buffer.first << " of the output buffers cannot be found in the output names "
                 << output_names << " of the method, servable " << servable_name << ", method " << method_name;
        }
        auto &shm_data = buffer.second;
        SharedMemoryAttachItem item;
        status = SharedMemoryManager::Instance().Attach(shm_data.memory_key(), shm_data.bytes_size(),
                                                        shm_data.data_offset(), shm_data.data_size(), &item);
        if (status != SUCCESS) {
          MSI_LOG_ERROR << "Attach output shared memory failed, memory key: " << shm_data.memory_key()
                        << ", bytes size: " << shm_data.bytes_size() << ", data offset: " << shm_data.data_offset()
                        << ", data size: " << shm_data.data_size() << ", output name: " << buffer.first;
          return status;
        }
      }
    }
    results->push_back(instance_data);
  }
  return SUCCESS;
}

Status GrpcTensorHelper::CheckRequestInstances(const proto::PredictRequest &request,
                                               const std::vector<std::string> &input_names) {
  auto servable_name = request.servable_spec().name();
  auto method_name = request.servable_spec().method_name();
  Status status;
  for (auto &proto_instance : request.instances()) {
    for (const auto &input_name : input_names) {
      auto it = proto_instance.items().find(input_name);
      if (it == proto_instance.items().end()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
               << "Cannot find input " << input_name << " in instance input , servable " << servable_name << ", method "
               << method_name;
      }
      status = CheckRequestTensor(it->second);
      if (status != SUCCESS) {
        auto status2 = INFER_STATUS(INVALID_INPUTS) << "Instances input " << input_name << " check failed";
        MSI_LOG_ERROR << status2.StatusMessage();
        return Status(INVALID_INPUTS, status2.StatusMessage() + ", detail: " + status.StatusMessage());
      }
    }
  }
  return SUCCESS;
}

void GrpcTensorHelper::CopyFromAgentSpec(const proto::AgentSpec &specs, WorkerAgentSpec *worker_specs) {
  worker_specs->rank_id = specs.rank_id();
  worker_specs->batch_size = specs.batch_size();
  for (auto &in : specs.inputs()) {
    TensorInfo info;
    info.data_type = ProtoTensor::TransDataType2Inference(in.dtype());
    info.size = in.size();
    info.is_no_batch_dim = in.is_no_batch_dim();
    for (auto &dim : in.shape().dims()) {
      info.shape.push_back(dim);
    }
    worker_specs->input_infos.push_back(info);
  }
  for (auto &out : specs.outputs()) {
    TensorInfo info;
    info.data_type = ProtoTensor::TransDataType2Inference(out.dtype());
    info.size = out.size();
    info.is_no_batch_dim = out.is_no_batch_dim();
    for (auto &dim : out.shape().dims()) {
      info.shape.push_back(dim);
    }
    worker_specs->output_infos.push_back(info);
  }
}

void GrpcTensorHelper::CopyFromWorkerAgentSpec(const std::vector<WorkerAgentSpec> &worker_specs,
                                               proto::AgentRegisterRequest *request) {
  for (size_t i = 0; i < worker_specs.size(); i++) {
    auto &spec = worker_specs[i];
    auto worker_spec = request->add_agent_spec();
    worker_spec->set_rank_id(spec.rank_id);
    worker_spec->set_batch_size(spec.batch_size);
    for (auto &method : spec.input_infos) {
      auto proto_method = worker_spec->add_inputs();
      proto_method->set_dtype(ProtoTensor::TransDataType2Proto(method.data_type));
      proto_method->set_size(method.size);
      proto_method->set_is_no_batch_dim(method.is_no_batch_dim);
      auto proto_shape = proto_method->mutable_shape();
      for (auto &dim : method.shape) {
        proto_shape->add_dims(dim);
      }
    }
    for (auto &method : spec.output_infos) {
      auto proto_method = worker_spec->add_outputs();
      proto_method->set_dtype(ProtoTensor::TransDataType2Proto(method.data_type));
      proto_method->set_size(method.size);
      proto_method->set_is_no_batch_dim(method.is_no_batch_dim);
      auto proto_shape = proto_method->mutable_shape();
      for (auto &dim : method.shape) {
        proto_shape->add_dims(dim);
      }
    }
  }
}

Status GrpcTensorHelper::CheckRequestTensor(const proto::Tensor &tensor) {
  Status status;
  ProtoTensor tensor_input(const_cast<proto::Tensor *>(&tensor));
  auto shape = tensor_input.shape();
  if (tensor.dtype() == proto::MS_BYTES || tensor.dtype() == proto::MS_STRING) {
    if (tensor.bytes_val_size() != 1) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
             << "Instance tensor check failed: bytes or string type shape batch size can only be 1";
    }
    if (!(shape.size() == 1 && shape[0] == 1) && !shape.empty()) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Instance tensor check failed: bytes or string type input "
                                                    << " shape can only be (1,) or empty, but given shape is " << shape;
    }
  } else {
    bool zero_dim = false;
    for (auto &shape_item : tensor.shape().dims()) {
      if (shape_item < 0 || zero_dim) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Tensor check failed: input "
                                                      << " shape " << shape << " invalid";
      }
      if (shape_item == 0) {
        zero_dim = true;
      }
    }
    auto item_size = tensor_input.itemsize();
    if (item_size == 0) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
             << "Tensor check failed: input data type " << tensor.dtype() << " invalid";
    }
    size_t element_num = tensor_input.element_cnt();
    auto expect_data_size = element_num * item_size;
    if (tensor.tensor_data_case() == proto::Tensor::TensorDataCase::kShmData) {
      if (expect_data_size != tensor.shm_data().data_size()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
               << "Tensor check failed: input shared memory data size " << tensor.shm_data().data_size()
               << " not equal to expected size " << expect_data_size;
      }
    } else {
      if (expect_data_size != tensor.data().size()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
               << "Tensor check failed: input data size " << tensor.data().size() << " invalid";
      }
    }
  }
  return SUCCESS;
}

void GrpcTensorHelper::CreateReplyFromErrorMsg(const Status &error_msg, proto::PredictReply *reply) {
  MSI_EXCEPTION_IF_NULL(reply);
  if (error_msg == SUCCESS) {
    return;
  }
  reply->clear_error_msg();
  reply->clear_instances();
  auto proto_error_msg = reply->add_error_msg();
  proto_error_msg->set_error_code(error_msg.StatusCode());
  std::string error_msg_str = error_msg.StatusMessage();
  if (error_msg_str.empty()) {
    proto_error_msg->set_error_msg("Predict failed");
  } else {
    proto_error_msg->set_error_msg(error_msg_str);
  }
}

serving::LogStream &operator<<(serving::LogStream &stream, proto::DataType data_type) {
  const std::map<proto::DataType, std::string> type_name_map{
    {proto::MS_UNKNOWN, "proto::MS_UNKNOWN"}, {proto::MS_BOOL, "proto::kMSI_Bool"},
    {proto::MS_INT8, "proto::MS_INT8"},       {proto::MS_UINT8, "proto::MS_UINT8"},
    {proto::MS_INT16, "proto::MS_INT16"},     {proto::MS_UINT16, "proto::MS_UINT16"},
    {proto::MS_INT32, "proto::MS_INT32"},     {proto::MS_UINT32, "proto::MS_UINT32"},
    {proto::MS_INT64, "proto::MS_INT64"},     {proto::MS_UINT64, "proto::MS_UINT64"},
    {proto::MS_FLOAT16, "proto::MS_FLOAT16"}, {proto::MS_FLOAT32, "proto::MS_FLOAT32"},
    {proto::MS_FLOAT64, "proto::MS_FLOAT64"}, {proto::MS_STRING, "proto::MS_STRING"},
    {proto::MS_BYTES, "proto::MS_BYTES"},
  };
  auto it = type_name_map.find(data_type);
  if (it != type_name_map.end()) {
    stream << it->second;
  } else {
    stream << "proto::MS_UNKNOWN";
  }
  return stream;
}

}  // namespace mindspore::serving
