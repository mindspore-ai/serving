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

#include "client/cpp/client.h"
#include <grpcpp/grpcpp.h>
#include <google/protobuf/text_format.h>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <sstream>
#include "proto/ms_service.pb.h"
#include "proto/ms_service.grpc.pb.h"

namespace mindspore {
namespace serving {
namespace client {

Status &Status::operator<<(DataType val) {
  std::unordered_map<DataType, std::string> data_type_map = {
    {DT_UINT8, "uint8"},   {DT_UINT16, "uint16"},   {DT_UINT32, "uint32"},   {DT_UINT64, "uint64"},
    {DT_INT8, "int8"},     {DT_INT16, "int16"},     {DT_INT32, "int32"},     {DT_INT64, "int64"},
    {DT_BOOL, "bool"},     {DT_FLOAT16, "float16"}, {DT_FLOAT32, "float32"}, {DT_FLOAT64, "float64"},
    {DT_STRING, "string"}, {DT_BYTES, "bytes"},     {DT_UNKNOWN, "unknown"},
  };
  auto it = data_type_map.find(val);
  if (it == data_type_map.end()) {
    status_msg_ += "unknown";
  } else {
    status_msg_ += it->second;
  }
  return *this;
}

Status &operator<<(Status &status, proto::DataType val) {
  std::unordered_map<proto::DataType, std::string> data_type_map = {
    {proto::MS_UINT8, "uint8"},     {proto::MS_UINT16, "uint16"},   {proto::MS_UINT32, "uint32"},
    {proto::MS_UINT64, "uint64"},   {proto::MS_INT8, "int8"},       {proto::MS_INT16, "int16"},
    {proto::MS_INT32, "int32"},     {proto::MS_INT64, "int64"},     {proto::MS_BOOL, "bool"},
    {proto::MS_FLOAT16, "float16"}, {proto::MS_FLOAT32, "float32"}, {proto::MS_FLOAT64, "float64"},
    {proto::MS_STRING, "string"},   {proto::MS_BYTES, "bytes"},     {proto::MS_UNKNOWN, "unknown"},
  };
  auto it = data_type_map.find(val);
  if (it == data_type_map.end()) {
    status << "unknown";
  } else {
    status << it->second;
  }
  return status;
}

Status &operator<<(Status &status, grpc::StatusCode val) {
  std::unordered_map<grpc::StatusCode, std::string> data_type_map = {
    {grpc::OK, "OK"},
    {grpc::CANCELLED, "CANCELLED"},
    {grpc::UNKNOWN, "UNKNOWN"},
    {grpc::INVALID_ARGUMENT, "INVALID_ARGUMENT"},
    {grpc::DEADLINE_EXCEEDED, "DEADLINE_EXCEEDED"},
    {grpc::NOT_FOUND, "NOT_FOUND"},
    {grpc::ALREADY_EXISTS, "ALREADY_EXISTS"},
    {grpc::PERMISSION_DENIED, "PERMISSION_DENIED"},
    {grpc::UNAUTHENTICATED, "UNAUTHENTICATED"},
    {grpc::RESOURCE_EXHAUSTED, "RESOURCE_EXHAUSTED"},
    {grpc::FAILED_PRECONDITION, "FAILED_PRECONDITION"},
    {grpc::ABORTED, "ABORTED"},
    {grpc::OUT_OF_RANGE, "OUT_OF_RANGE"},
    {grpc::UNIMPLEMENTED, "UNIMPLEMENTED"},
    {grpc::INTERNAL, "INTERNAL"},
    {grpc::UNAVAILABLE, "UNAVAILABLE"},
    {grpc::DATA_LOSS, "DATA_LOSS"},
  };
  auto it = data_type_map.find(val);
  if (it == data_type_map.end()) {
    status << "unknown";
  } else {
    status << it->second;
  }
  return status;
}

Status MutableTensor::SetBytesData(const std::vector<uint8_t> &val) {
  if (mutable_proto_tensor_ == nullptr) {
    return Status(SYSTEM_ERROR) << "proto tensor cannot be nullptr";
  }
  auto proto_shape = mutable_proto_tensor_->mutable_shape();
  proto_shape->add_dims(1);
  mutable_proto_tensor_->set_dtype(proto::MS_BYTES);
  if (val.empty()) {
    return Status(INVALID_INPUTS) << "Input index bytes val len is empty";
  }
  mutable_proto_tensor_->add_bytes_val(val.data(), val.size());
  return SUCCESS;
}

Status MutableTensor::SetStrData(const std::string &val) {
  if (mutable_proto_tensor_ == nullptr) {
    return Status(SYSTEM_ERROR) << "proto tensor cannot be nullptr";
  }
  auto proto_shape = mutable_proto_tensor_->mutable_shape();
  proto_shape->add_dims(val.size());
  mutable_proto_tensor_->set_dtype(proto::MS_STRING);
  if (val.empty()) {
    return Status(INVALID_INPUTS) << "string index string val len is empty";
  }
  mutable_proto_tensor_->add_bytes_val(val);
  return SUCCESS;
}

Status MutableTensor::SetData(const std::vector<uint8_t> &val, const std::vector<int64_t> &shape) {
  return SetData(val.data(), val.size() * sizeof(uint8_t), shape, DT_UINT8);
}

Status MutableTensor::SetData(const std::vector<uint16_t> &val, const std::vector<int64_t> &shape) {
  return SetData(val.data(), val.size() * sizeof(uint16_t), shape, DT_UINT16);
}

Status MutableTensor::SetData(const std::vector<uint32_t> &val, const std::vector<int64_t> &shape) {
  return SetData(val.data(), val.size() * sizeof(uint32_t), shape, DT_UINT32);
}

Status MutableTensor::SetData(const std::vector<uint64_t> &val, const std::vector<int64_t> &shape) {
  return SetData(val.data(), val.size() * sizeof(uint64_t), shape, DT_UINT64);
}

Status MutableTensor::SetData(const std::vector<int8_t> &val, const std::vector<int64_t> &shape) {
  return SetData(val.data(), val.size() * sizeof(int8_t), shape, DT_INT8);
}

Status MutableTensor::SetData(const std::vector<int16_t> &val, const std::vector<int64_t> &shape) {
  return SetData(val.data(), val.size() * sizeof(int16_t), shape, DT_INT16);
}

Status MutableTensor::SetData(const std::vector<int32_t> &val, const std::vector<int64_t> &shape) {
  return SetData(val.data(), val.size() * sizeof(int32_t), shape, DT_INT32);
}

Status MutableTensor::SetData(const std::vector<int64_t> &val, const std::vector<int64_t> &shape) {
  return SetData(val.data(), val.size() * sizeof(int64_t), shape, DT_INT64);
}

Status MutableTensor::SetData(const std::vector<bool> &val, const std::vector<int64_t> &shape) {
  std::vector<uint8_t> val_uint8;
  std::transform(val.begin(), val.end(), std::back_inserter(val_uint8),
                 [](bool item) { return static_cast<uint8_t>(item); });
  return SetData(val_uint8.data(), val_uint8.size() * sizeof(bool), shape, DT_BOOL);
}

Status MutableTensor::SetData(const std::vector<float> &val, const std::vector<int64_t> &shape) {
  return SetData(val.data(), val.size() * sizeof(float), shape, DT_FLOAT32);
}

Status MutableTensor::SetData(const std::vector<double> &val, const std::vector<int64_t> &shape) {
  return SetData(val.data(), val.size() * sizeof(double), shape, DT_FLOAT64);
}

Status MutableTensor::SetData(const void *data, size_t data_len, const std::vector<int64_t> &shape,
                              DataType data_type) {
  if (mutable_proto_tensor_ == nullptr) {
    return Status(SYSTEM_ERROR) << "proto tensor cannot be nullptr";
  }
  if (data == nullptr || data_len == 0) {
    return Status(INVALID_INPUTS) << "data cannot be nullptr, or data len cannot be 0";
  }
  mutable_proto_tensor_->set_data(data, data_len);
  auto proto_shape = mutable_proto_tensor_->mutable_shape();

  std::unordered_map<DataType, std::pair<proto::DataType, int64_t>> data_type_map = {
    {DT_UINT8, {proto::MS_UINT8, sizeof(uint8_t)}},
    {DT_UINT16, {proto::MS_UINT16, sizeof(uint16_t)}},
    {DT_UINT32, {proto::MS_UINT32, sizeof(uint32_t)}},
    {DT_UINT64, {proto::MS_UINT64, sizeof(uint64_t)}},
    {DT_INT8, {proto::MS_INT8, sizeof(int8_t)}},
    {DT_INT16, {proto::MS_INT16, sizeof(int16_t)}},
    {DT_INT32, {proto::MS_INT32, sizeof(int32_t)}},
    {DT_INT64, {proto::MS_INT64, sizeof(int64_t)}},
    {DT_BOOL, {proto::MS_BOOL, sizeof(bool)}},
    {DT_FLOAT16, {proto::MS_FLOAT16, 2}},
    {DT_FLOAT32, {proto::MS_FLOAT32, 4}},
    {DT_FLOAT64, {proto::MS_FLOAT64, 8}},
  };
  auto it = data_type_map.find(data_type);
  if (it == data_type_map.end()) {
    return Status(INVALID_INPUTS) << "Input unsupported find data type " << data_type;
  }
  mutable_proto_tensor_->set_dtype(it->second.first);

  auto shape_str = [](const std::vector<int64_t> &val) noexcept {
    std::stringstream sstream;
    sstream << "[";
    for (size_t i = 0; i < val.size(); i++) {
      sstream << val[i];
      if (i + 1 < val.size()) {
        sstream << ", ";
      }
    }
    sstream << "]";
    return sstream.str();
  };
  int64_t element_cnt = 1;
  for (auto &item : shape) {
    proto_shape->add_dims(item);
    if (item <= 0 || item >= INT64_MAX || INT64_MAX / element_cnt < item) {
      return Status(INVALID_INPUTS) << "Input input shape invalid " << shape_str(shape);
    }
  }
  auto item_size = it->second.second;
  if (static_cast<int64_t>(data_len) / element_cnt < item_size ||
      element_cnt * item_size != static_cast<int64_t>(data_len)) {
    return Status(INVALID_INPUTS) << "Input input shape " << shape_str(shape) << " does not match data len "
                                  << data_len;
  }
  return SUCCESS;
}

Status Tensor::GetBytesData(std::vector<uint8_t> *val) const {
  if (val == nullptr) {
    return Status(SYSTEM_ERROR) << "input val cannot be nullptr";
  }
  if (proto_tensor_ == nullptr) {
    return Status(SYSTEM_ERROR) << "proto tensor cannot be nullptr";
  }
  if (proto_tensor_->dtype() != proto::MS_BYTES) {
    return Status(INVALID_INPUTS) << "Output data type is not match, its' real data type is " << proto_tensor_->dtype();
  }
  auto &bytes_data = proto_tensor_->bytes_val();
  if (bytes_data.size() != 1) {
    return Status(INVALID_INPUTS) << "Bytes value type size can only be 1";
  }
  val->resize(bytes_data[0].size());
  memcpy(val->data(), val->data(), bytes_data[0].size());
  return SUCCESS;
}

Status Tensor::GetStrData(std::string *val) const {
  if (val == nullptr) {
    return Status(SYSTEM_ERROR) << "input val cannot be nullptr";
  }
  if (proto_tensor_ == nullptr) {
    return Status(SYSTEM_ERROR) << "proto tensor cannot be nullptr";
  }
  if (proto_tensor_->dtype() != proto::MS_STRING) {
    return Status(INVALID_INPUTS) << "Output data type is not match, its' real data type is " << proto_tensor_->dtype();
  }
  auto &bytes_data = proto_tensor_->bytes_val();
  if (bytes_data.size() != 1) {
    return Status(INVALID_INPUTS) << "Bytes value type size can only be 1";
  }
  val->resize(bytes_data[0].size());
  memcpy(val->data(), val->data(), bytes_data[0].size());
  return SUCCESS;
}

template <proto::DataType proto_dtype, class DT>
Status GetInputImp(const proto::Tensor *proto_tensor, std::vector<DT> *val) {
  if (val == nullptr) {
    return Status(SYSTEM_ERROR) << "input val cannot be nullptr";
  }
  if (proto_tensor == nullptr) {
    return Status(SYSTEM_ERROR) << "proto tensor cannot be nullptr";
  }
  if (proto_tensor->dtype() != proto_dtype) {
    return Status(INVALID_INPUTS) << "Output data type is not match, its' real data type is " << proto_tensor->dtype();
  }
  auto data = proto_tensor->data().data();
  auto data_len = proto_tensor->data().length();
  val->resize(data_len / sizeof(DT));
  memcpy(val->data(), data, data_len);
  return SUCCESS;
}

Status Tensor::GetData(std::vector<uint8_t> *val) const { return GetInputImp<proto::MS_UINT8>(proto_tensor_, val); }

Status Tensor::GetData(std::vector<uint16_t> *val) const { return GetInputImp<proto::MS_UINT16>(proto_tensor_, val); }

Status Tensor::GetData(std::vector<uint32_t> *val) const { return GetInputImp<proto::MS_UINT32>(proto_tensor_, val); }

Status Tensor::GetData(std::vector<uint64_t> *val) const { return GetInputImp<proto::MS_UINT64>(proto_tensor_, val); }

Status Tensor::GetData(std::vector<int8_t> *val) const { return GetInputImp<proto::MS_INT8>(proto_tensor_, val); }

Status Tensor::GetData(std::vector<int16_t> *val) const { return GetInputImp<proto::MS_INT16>(proto_tensor_, val); }

Status Tensor::GetData(std::vector<int32_t> *val) const { return GetInputImp<proto::MS_INT32>(proto_tensor_, val); }

Status Tensor::GetData(std::vector<int64_t> *val) const { return GetInputImp<proto::MS_INT64>(proto_tensor_, val); }

Status Tensor::GetData(std::vector<bool> *val) const {
  if (val == nullptr) {
    return Status(SYSTEM_ERROR) << "input val cannot be nullptr";
  }
  std::vector<uint8_t> val_uint8;
  Status status = GetInputImp<proto::MS_BOOL>(proto_tensor_, &val_uint8);
  if (!status.IsSuccess()) {
    return status;
  }
  std::transform(val_uint8.begin(), val_uint8.end(), std::back_inserter(*val), [](uint8_t item) { return item != 0; });
  return SUCCESS;
}

Status Tensor::GetData(std::vector<float> *val) const { return GetInputImp<proto::MS_FLOAT32>(proto_tensor_, val); }

Status Tensor::GetData(std::vector<double> *val) const { return GetInputImp<proto::MS_FLOAT64>(proto_tensor_, val); }

Status Tensor::GetFp16Data(std::vector<uint16_t> *val) const {
  return GetInputImp<proto::MS_FLOAT16>(proto_tensor_, val);
}

DataType Tensor::GetDataType() const {
  if (proto_tensor_ == nullptr) {
    std::cout << "proto tensor cannot be nullptr" << std::endl;
    return DT_UNKNOWN;
  }
  std::unordered_map<proto::DataType, DataType> data_type_map = {
    {proto::MS_UNKNOWN, DT_UNKNOWN}, {proto::MS_UINT8, DT_UINT8},     {proto::MS_UINT16, DT_UINT16},
    {proto::MS_UINT32, DT_UINT32},   {proto::MS_UINT64, DT_UINT64},   {proto::MS_INT8, DT_INT8},
    {proto::MS_INT16, DT_INT16},     {proto::MS_INT32, DT_INT32},     {proto::MS_INT64, DT_INT64},
    {proto::MS_BOOL, DT_BOOL},       {proto::MS_FLOAT16, DT_FLOAT16}, {proto::MS_FLOAT32, DT_FLOAT32},
    {proto::MS_FLOAT64, DT_FLOAT64}, {proto::MS_STRING, DT_STRING},   {proto::MS_BYTES, DT_BYTES},
  };
  auto it_dt = data_type_map.find(proto_tensor_->dtype());
  if (it_dt == data_type_map.end()) {
    std::cout << "Unsupported data type " << proto_tensor_->dtype() << std::endl;
    return DT_UNKNOWN;
  }
  return it_dt->second;
}

std::vector<int64_t> Tensor::GetShape() const {
  if (proto_tensor_ == nullptr) {
    std::cout << "proto tensor cannot be nullptr" << std::endl;
    return std::vector<int64_t>();
  }
  std::vector<int64_t> shape;
  auto &dims = proto_tensor_->shape().dims();
  std::copy(dims.begin(), dims.end(), std::back_inserter(shape));
  return shape;
}

Tensor Instance::Get(const std::string &item_name) const {
  if (proto_instance_ == nullptr) {
    std::cout << "proto instance cannot be nullptr" << std::endl;
    return Tensor(nullptr, nullptr);
  }
  auto &items = proto_instance_->items();
  auto it = items.find(item_name);
  if (it == items.end()) {
    std::cout << "Cannot find item name " << item_name << std::endl;
    return Tensor(nullptr, nullptr);
  }
  return Tensor(message_owner_, &it->second);
}

bool Instance::HasErrorMsg(int64_t *error_code, std::string *error_msg) const {
  if (error_code == nullptr) {
    return false;
  }
  if (error_msg == nullptr) {
    return false;
  }
  if (error_msg_ == nullptr) {
    return false;
  }
  *error_code = error_msg_->error_code();
  *error_msg = error_msg_->error_msg();
  return true;
}

MutableTensor MutableInstance::Add(const std::string &item_name) {
  if (mutable_proto_instance_ == nullptr) {
    std::cout << "proto instance cannot be nullptr" << std::endl;
    return MutableTensor(nullptr, nullptr);
  }
  auto items = mutable_proto_instance_->mutable_items();
  auto &proto_tensor = (*items)[item_name];
  return MutableTensor(message_owner_, &proto_tensor);
}

InstancesRequest::InstancesRequest() { request_ = std::make_shared<proto::PredictRequest>(); }

MutableInstance InstancesRequest::AddInstance() {
  auto proto_instance = request_->add_instances();
  return MutableInstance(request_, proto_instance);
}

InstancesReply::InstancesReply() { reply_ = std::make_shared<proto::PredictReply>(); }

std::vector<Instance> InstancesReply::GetResult() const {
  std::vector<Instance> instances;
  auto &proto_instances = reply_->instances();
  auto &proto_error_msgs = reply_->error_msg();
  for (int i = 0; i < proto_instances.size(); i++) {
    auto &proto_instance = proto_instances[i];
    const proto::ErrorMsg *error_msg = nullptr;
    if (proto_error_msgs.size() == 1) {
      error_msg = &proto_error_msgs[0];
    } else if (proto_error_msgs.size() == proto_instances.size() && proto_error_msgs[i].error_code() != 0) {
      error_msg = &proto_error_msgs[i];
    }
    instances.push_back(Instance(reply_, &proto_instance, error_msg));
  }
  return instances;
}

class ClientImpl {
 public:
  ClientImpl(const std::string &server_ip, uint64_t server_port) {
    std::string target_str = server_ip + ":" + std::to_string(server_port);
    auto channel = grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials());
    stub_ = proto::MSService::NewStub(channel);
  }
  Status Predict(const proto::PredictRequest &request, proto::PredictReply *reply) {
    if (reply == nullptr) {
      return Status(SYSTEM_ERROR, "ClientImpl::Predict input reply cannot be nullptr");
    }
    grpc::ClientContext context;

    // The actual RPC.
    grpc::Status status = stub_->Predict(&context, request, reply);
    if (status.ok()) {
      return SUCCESS;
    } else {
      std::cout << status.error_code() << ": " << status.error_message() << std::endl;
      return Status(FAILED, status.error_message());
    }
  }

 private:
  std::unique_ptr<proto::MSService::Stub> stub_;
};

Client::Client(const std::string &server_ip, uint64_t server_port, const std::string &servable_name,
               const std::string &method_name, uint64_t version_number)
    : server_ip_(server_ip),
      server_port_(server_port),
      servable_name_(servable_name),
      method_name_(method_name),
      version_number_(version_number),
      impl_(std::make_shared<ClientImpl>(server_ip, server_port)) {}

Status Client::SendRequest(const InstancesRequest &request, InstancesReply *reply) {
  if (reply == nullptr) {
    return Status(SYSTEM_ERROR) << "input reply cannot be nullptr";
  }
  proto::PredictRequest *proto_request = request.request_.get();
  proto::PredictReply *proto_reply = reply->reply_.get();
  auto servable_spec = proto_request->mutable_servable_spec();
  servable_spec->set_name(servable_name_);
  servable_spec->set_method_name(method_name_);
  servable_spec->set_version_number(version_number_);

  Status result = impl_->Predict(*proto_request, proto_reply);
  return result;
}

}  // namespace client
}  // namespace serving
}  // namespace mindspore
