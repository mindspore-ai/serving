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
    if (dim == 0 && shape.size() == 1) {
      MSI_LOG_INFO << "Detect shape (0,) ";
      tensor_shape->add_dims(0);
      return;
    }
    if (dim <= 0 || element_count > kMaxShapeElementCount / dim) {
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
  return tensor_->data().size();
}

uint8_t *ProtoTensor::mutable_data() {
  MSI_EXCEPTION_IF_NULL(tensor_);
  if (data_size() == 0) {
    return nullptr;
  }
  return reinterpret_cast<uint8_t *>(tensor_->mutable_data()->data());
}

const uint8_t *ProtoTensor::data() const {
  MSI_EXCEPTION_IF_NULL(tensor_);
  if (data_size() == 0) {
    return nullptr;
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

void GrpcTensorHelper::GetRequestSpec(const proto::PredictRequest &request, RequestSpec *request_spec) {
  MSI_EXCEPTION_IF_NULL(request_spec);
  request_spec->servable_name = request.servable_spec().name();
  request_spec->method_name = request.servable_spec().method_name();
  request_spec->version_number = request.servable_spec().version_number();
}

void GrpcTensorHelper::GetWorkerSpec(const proto::RegisterRequest &request, std::vector<WorkerSpec> *worker_specs) {
  MSI_EXCEPTION_IF_NULL(worker_specs);
  for (auto &proto_spec : request.worker_spec()) {
    WorkerSpec worker_spec;
    worker_spec.worker_address = request.address();
    worker_spec.servable_name = proto_spec.name();
    worker_spec.version_number = proto_spec.version_number();
    for (const auto &proto_method : proto_spec.methods()) {
      WorkerMethodInfo method_info;
      method_info.name = proto_method.name();
      for (auto &name : proto_method.input_names()) {
        method_info.input_names.push_back(name);
      }
      worker_spec.methods.push_back(method_info);
    }
    worker_specs->push_back(worker_spec);
  }
}

void GrpcTensorHelper::GetWorkerSpec(const proto::AddWorkerRequest &request, WorkerSpec *worker_spec) {
  MSI_EXCEPTION_IF_NULL(worker_spec);
  worker_spec->worker_address = request.address();
  worker_spec->servable_name = request.worker_spec().name();
  worker_spec->version_number = request.worker_spec().version_number();
  for (const auto &proto_method : request.worker_spec().methods()) {
    WorkerMethodInfo method_info;
    method_info.name = proto_method.name();
    for (auto &name : proto_method.input_names()) {
      method_info.input_names.push_back(name);
    }
    worker_spec->methods.push_back(method_info);
  }
}

void GrpcTensorHelper::GetWorkerSpec(const proto::RemoveWorkerRequest &request, WorkerSpec *worker_spec) {
  MSI_EXCEPTION_IF_NULL(worker_spec);
  worker_spec->worker_address = request.address();
  worker_spec->servable_name = request.worker_spec().name();
  worker_spec->version_number = request.worker_spec().version_number();
  for (const auto &proto_method : request.worker_spec().methods()) {
    WorkerMethodInfo method_info;
    method_info.name = proto_method.name();
    for (auto &name : proto_method.input_names()) {
      method_info.input_names.push_back(name);
    }
    worker_spec->methods.push_back(method_info);
  }
}

Status GrpcTensorHelper::CreateInstanceFromRequest(const proto::PredictRequest &request, RequestSpec *request_spec,
                                                   vector<InstanceData> *results) {
  MSI_EXCEPTION_IF_NULL(request_spec);
  MSI_EXCEPTION_IF_NULL(results);
  results->clear();

  Status status;
  GetRequestSpec(request, request_spec);

  auto servable_name = request_spec->servable_name;
  auto method_name = request_spec->method_name;

  ServableSignature servable_signature;
  if (!ServableStorage::Instance()->GetServableDef(servable_name, &servable_signature)) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Servable " << servable_name << " is not declared";
  }
  MethodSignature method_signature;
  if (!servable_signature.GetMethodDeclare(request_spec->method_name, &method_signature)) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
           << "Method " << method_name << " is not registed for servable " << servable_name;
  }

  // instance
  if (request.instances_size() > 0) {
    status = CreateInstanceFromRequestInstances(request, method_signature.inputs, results);
    if (status != SUCCESS) {
      MSI_LOG_ERROR << "Create instances from request instances failed";
      return status;
    }
  }
  return SUCCESS;
}

Status GrpcTensorHelper::CreateReplyFromInstances(const proto::PredictRequest &request, const vector<Instance> &outputs,
                                                  proto::PredictReply *reply) {
  MSI_EXCEPTION_IF_NULL(reply);
  auto servable_name = request.servable_spec().name();
  auto method_name = request.servable_spec().method_name();
  Status status;
  ServableSignature servable_signature;
  if (!ServableStorage::Instance()->GetServableDef(servable_name, &servable_signature)) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Servable " << servable_name << " is not declared";
  }
  MethodSignature method_signature;
  if (!servable_signature.GetMethodDeclare(method_name, &method_signature)) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
           << "Method " << method_name << " is not registed for servable " << servable_name;
  }
  *reply->mutable_servable_spec() = request.servable_spec();

  size_t err_cnt = 0;
  for (auto &output_intance : outputs) {
    if (output_intance.error_msg != SUCCESS) {
      err_cnt++;
    } else if (output_intance.data.size() != method_signature.outputs.size()) {
      return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
             << "Result data tensor size " << output_intance.data.size() << " not equal outputs size "
             << method_signature.outputs.size() << "defined in method signature";
    }
  }
  if (err_cnt > 0) {
    for (auto &output_intance : outputs) {
      auto proto_err_msg = reply->add_error_msg();
      proto_err_msg->set_error_code(output_intance.error_msg.StatusCode());
      if (output_intance.error_msg == INVALID_INPUTS) {
        proto_err_msg->set_error_msg(output_intance.error_msg.StatusMessage());
      } else if (output_intance.error_msg != SUCCESS) {
        proto_err_msg->set_error_msg(output_intance.error_msg.StatusMessage());
      }
    }
  }
  // create instance reply, same with request
  if (request.instances_size() > 0) {
    for (auto &output_intance : outputs) {
      auto proto_instance = reply->add_instances();
      if (output_intance.data.empty()) {
        continue;
      }
      auto proto_items = proto_instance->mutable_items();
      for (size_t i = 0; i < method_signature.outputs.size(); i++) {
        auto &output_tensor = output_intance.data[i];
        auto &proto_tensor = (*proto_items)[method_signature.outputs[i]];
        ProtoTensor result_tensor(&proto_tensor);
        result_tensor.assgin(*output_tensor);
      }
    }
  }
  return SUCCESS;
}

Status GrpcTensorHelper::CreateInstanceFromRequestInstances(const proto::PredictRequest &request,
                                                            const std::vector<std::string> &input_names,
                                                            std::vector<InstanceData> *results) {
  MSI_EXCEPTION_IF_NULL(results);
  auto servable_name = request.servable_spec().name();
  auto method_name = request.servable_spec().method_name();
  Status status;
  for (auto &proto_instance : request.instances()) {
    InstanceData instance_data;
    for (const auto &input_name : input_names) {
      auto it = proto_instance.items().find(input_name);
      if (it == proto_instance.items().end()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
               << "Cannot find input " << input_name << " in instance input , servable " << servable_name << ", method "
               << method_name;
      }
      status = CheckRequestTensor(it->second, true, 1);
      if (status != SUCCESS) {
        auto status2 = INFER_STATUS(INVALID_INPUTS) << "Instances input " << input_name << " check failed";
        MSI_LOG_ERROR << status2.StatusMessage();
        return Status(INVALID_INPUTS, status2.StatusMessage() + ", detail: " + status.StatusMessage());
      }
      auto add_tensor = std::make_shared<ProtoTensor>(const_cast<proto::Tensor *>(&it->second));
      instance_data.push_back(add_tensor);
    }
    results->push_back(instance_data);
  }
  return SUCCESS;
}

Status GrpcTensorHelper::CheckRequestTensor(const proto::Tensor &tensor, bool is_instance_tensor, uint32_t batch_size) {
  Status status;
  ProtoTensor tensor_input(const_cast<proto::Tensor *>(&tensor));
  auto shape = tensor_input.shape();
  if (tensor.dtype() == proto::MS_BYTES || tensor.dtype() == proto::MS_STRING) {
    if (is_instance_tensor) {
      if (tensor.bytes_val_size() != 1) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
               << "Instance tensor check failed: bytes or string type shape batch size " << batch_size
               << " not equal to bytes val size " << tensor.bytes_val_size();
      }
      if (!(shape.size() == 1 && shape[0] == 1) && !shape.empty()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
               << "Instance tensor check failed: bytes or string type input "
               << " shape can only be (1,) or empty, but given shape is " << shape;
      }
    } else {
      if (static_cast<int64_t>(tensor.bytes_val_size()) != batch_size) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
               << "Inputs tensor check failed: bytes or string type shape batch size " << batch_size
               << " not equal to bytes val size " << tensor.bytes_val_size();
      }
      if (shape.size() != 1) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
               << "Inputs Tensor check failed: bytes or string type input "
               << " shape can only be (batch_size,), but given shape is " << shape;
      }
    }
  } else {
    size_t element_num = tensor_input.element_cnt();
    if (element_num == 0) {  // shape dim invalid or element count > max element num
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Tensor check failed: input "
                                                    << " shape " << tensor_input.shape() << " invalid";
    }
    if (tensor_input.data_type() == kMSI_Unknown || tensor_input.itemsize() == 0) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Tensor check failed: input "
                                                    << " data type " << tensor.dtype() << " invalid";
    }
    if (element_num * tensor_input.itemsize() != tensor_input.data_size()) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Tensor check failed: input "
                                                    << " data size " << tensor.data().size() << " invalid";
    }
  }
  return SUCCESS;
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
