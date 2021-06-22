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

#ifndef MINDSPORE_SERVING_CLIENT_H
#define MINDSPORE_SERVING_CLIENT_H

#include <string>
#include <vector>
#include <memory>
#include <sstream>

namespace google {
namespace protobuf {
class Message;
}
}  // namespace google

namespace mindspore {
namespace serving {
#define MS_API __attribute__((visibility("default")))

namespace proto {
class Tensor;
class Instance;
class PredictRequest;
class PredictReply;
class ErrorMsg;
}  // namespace proto

namespace client {

using ProtoMsgOwner = std::shared_ptr<google::protobuf::Message>;

enum DataType {
  DT_UNKNOWN,
  DT_UINT8,
  DT_UINT16,
  DT_UINT32,
  DT_UINT64,
  DT_INT8,
  DT_INT16,
  DT_INT32,
  DT_INT64,
  DT_BOOL,
  DT_FLOAT16,
  DT_FLOAT32,
  DT_FLOAT64,
  DT_STRING,
  DT_BYTES,
};

enum StatusCode { SUCCESS = 0, FAILED, INVALID_INPUTS, SYSTEM_ERROR, UNAVAILABLE };

class MS_API Status {
 public:
  Status() : status_code_(FAILED) {}
  Status(enum StatusCode status_code, const std::string &status_msg = "")  // NOLINT(runtime/explicit)
      : status_code_(status_code), status_msg_(status_msg) {}
  bool IsSuccess() const { return status_code_ == SUCCESS; }
  enum StatusCode StatusCode() const { return status_code_; }
  std::string StatusMessage() { return status_msg_; }
  bool operator==(const Status &other) const { return status_code_ == other.status_code_; }
  bool operator==(enum StatusCode other_code) const { return status_code_ == other_code; }
  bool operator!=(const Status &other) const { return status_code_ != other.status_code_; }
  bool operator!=(enum StatusCode other_code) const { return status_code_ != other_code; }
  operator bool() const = delete;

  template <class T>
  Status &operator<<(T val);
  Status &operator<<(DataType val);
  template <class T>
  Status &operator<<(const std::vector<T> &val);

 private:
  enum StatusCode status_code_;
  std::string status_msg_;
};

class MS_API Tensor {
 public:
  Tensor(const ProtoMsgOwner &owner, const proto::Tensor *proto_tensor)
      : message_owner_(owner), proto_tensor_(proto_tensor) {}
  virtual ~Tensor() = default;
  // Bytes type: for images etc.
  Status GetBytesData(std::vector<uint8_t> *val) const;
  Status GetStrData(std::string *val) const;
  Status GetData(std::vector<uint8_t> *val) const;
  Status GetData(std::vector<uint16_t> *val) const;
  Status GetData(std::vector<uint32_t> *val) const;
  Status GetData(std::vector<uint64_t> *val) const;
  Status GetData(std::vector<int8_t> *val) const;
  Status GetData(std::vector<int16_t> *val) const;
  Status GetData(std::vector<int32_t> *val) const;
  Status GetData(std::vector<int64_t> *val) const;
  Status GetData(std::vector<bool> *val) const;
  Status GetData(std::vector<float> *val) const;
  Status GetData(std::vector<double> *val) const;
  Status GetFp16Data(std::vector<uint16_t> *val) const;
  DataType GetDataType() const;
  std::vector<int64_t> GetShape() const;

  bool IsValid() const { return proto_tensor_ != nullptr; }

 protected:
  ProtoMsgOwner message_owner_;

 private:
  const proto::Tensor *proto_tensor_;
};

class MS_API MutableTensor : public Tensor {
 public:
  MutableTensor(const ProtoMsgOwner &owner, proto::Tensor *proto_tensor)
      : Tensor(owner, proto_tensor), mutable_proto_tensor_(proto_tensor) {}
  ~MutableTensor() = default;

  // Bytes type: for images etc.
  Status SetBytesData(const std::vector<uint8_t> &val);
  Status SetStrData(const std::string &val);

  Status SetData(const std::vector<uint8_t> &val, const std::vector<int64_t> &shape);
  Status SetData(const std::vector<uint16_t> &val, const std::vector<int64_t> &shape);
  Status SetData(const std::vector<uint32_t> &val, const std::vector<int64_t> &shape);
  Status SetData(const std::vector<uint64_t> &val, const std::vector<int64_t> &shape);
  Status SetData(const std::vector<int8_t> &val, const std::vector<int64_t> &shape);
  Status SetData(const std::vector<int16_t> &val, const std::vector<int64_t> &shape);
  Status SetData(const std::vector<int32_t> &val, const std::vector<int64_t> &shape);
  Status SetData(const std::vector<int64_t> &val, const std::vector<int64_t> &shape);
  Status SetData(const std::vector<bool> &val, const std::vector<int64_t> &shape);
  Status SetData(const std::vector<float> &val, const std::vector<int64_t> &shape);
  Status SetData(const std::vector<double> &val, const std::vector<int64_t> &shape);

  Status SetData(const void *data, size_t data_bytes_len, const std::vector<int64_t> &shape, DataType data_type);

 private:
  proto::Tensor *mutable_proto_tensor_;
};

class MS_API Instance {
 public:
  Instance(const ProtoMsgOwner &owner, const proto::Instance *proto_instance, const proto::ErrorMsg *error_msg)
      : message_owner_(owner), proto_instance_(proto_instance), error_msg_(error_msg) {}
  virtual ~Instance() = default;

  Tensor Get(const std::string &item_name) const;

  bool IsValid() const { return proto_instance_ != nullptr; }
  bool HasErrorMsg(int64_t *error_code, std::string *error_msg) const;

 protected:
  ProtoMsgOwner message_owner_;

 private:
  const proto::Instance *proto_instance_;
  const proto::ErrorMsg *error_msg_;
};

class MS_API MutableInstance : public Instance {
 public:
  MutableInstance(const ProtoMsgOwner &owner, proto::Instance *proto_instance)
      : Instance(owner, proto_instance, nullptr), mutable_proto_instance_(proto_instance) {}
  ~MutableInstance() = default;

  MutableTensor Add(const std::string &item_name);

 private:
  proto::Instance *mutable_proto_instance_;
};

class MS_API InstancesRequest {
 public:
  InstancesRequest();
  ~InstancesRequest() = default;
  MutableInstance AddInstance();

 private:
  std::shared_ptr<proto::PredictRequest> request_ = nullptr;
  friend class Client;
};

class MS_API InstancesReply {
 public:
  InstancesReply();
  ~InstancesReply() = default;
  std::vector<Instance> GetResult() const;

 private:
  std::shared_ptr<proto::PredictReply> reply_ = nullptr;
  friend class Client;
};

class ClientImpl;
class MS_API Client {
 public:
  Client(const std::string &server_ip, uint64_t server_port, const std::string &servable_name,
         const std::string &method_name, uint64_t version_number = 0);
  ~Client() = default;

  Status SendRequest(const InstancesRequest &request, InstancesReply *reply);

 private:
  std::string server_ip_;
  uint64_t server_port_;
  std::string servable_name_;
  std::string method_name_;
  uint64_t version_number_ = 0;
  std::shared_ptr<ClientImpl> impl_;
};

template <class T>
Status &Status::operator<<(T val) {
  std::stringstream stringstream;
  stringstream << val;
  status_msg_ += stringstream.str();
  return *this;
}

template <class T>
Status &Status::operator<<(const std::vector<T> &val) {
  operator<<("[");
  for (size_t i = 0; i < val.size(); i++) {
    operator<<(val[i]);
    if (i != val.size() - 1) {
      operator<<(", ");
    }
  }
  operator<<("[");
  return *this;
}

}  // namespace client
}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_CLIENT_H
