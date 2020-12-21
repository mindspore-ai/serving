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

#ifndef MINDSPORE_SERVING_MASTER_HTTP_PROCESS_H
#define MINDSPORE_SERVING_MASTER_HTTP_PROCESS_H

#include <string>
#include <memory>
#include <vector>
#include <nlohmann/json.hpp>
#include "proto/ms_service.pb.h"
#include "master/dispacther.h"
#include "common/proto_tensor.h"
#include "master/restful/restful_request.h"

using nlohmann::json;
using std::string;

namespace mindspore {
namespace serving {
constexpr auto kInstancesRequest = "instances";
constexpr auto kInstancesReply = "instances";
constexpr auto kErrorMsg = "error_msg";
constexpr auto kType = "type";
constexpr auto kShape = "shape";
constexpr auto kB64 = "b64";

enum RequestType { kInstanceType = 0, kInvalidType };
enum InstancesType { kNokeyWay = 0, kKeyWay, kInvalidWay };
enum HTTP_DATA_TYPE { HTTP_DATA_NONE, HTTP_DATA_INT, HTTP_DATA_FLOAT, HTTP_DATA_BOOL, HTTP_DATA_STR, HTTP_DATA_OBJ };
class RestfulService {
 public:
  explicit RestfulService(const std::shared_ptr<Dispatcher> &dispatcher) : dispatcher_(dispatcher) {}
  ~RestfulService() = default;
  Status RunRestful(const std::shared_ptr<RestfulRequest> &restful_request, json *const out_json);

 private:
  Status CheckObjTypeMatchShape(DataType data_type, const std::vector<int64_t> &shape);
  std::string GetString(const uint8_t *ptr, size_t length);
  Status CheckObj(const json &js);
  Status CheckObjType(const std::string &type);
  DataType GetObjDataType(const json &js);
  std::vector<int64_t> GetObjShape(const json &js);
  std::vector<int64_t> GetArrayShape(const json &json_array);
  std::vector<int64_t> GetSpecifiedShape(const json &js);
  DataType GetArrayDataType(const json &json_array, HTTP_DATA_TYPE *type_format);
  Status CheckReqJsonValid(const json &js_msg);
  std::string GetStringByDataType(DataType type);
  bool JsonMatchDataType(const json &js, DataType type);

  template <typename T>
  Status GetScalarData(const json &js, size_t index, bool is_bytes, ProtoTensor *const request_tensor);
  Status GetScalarByType(DataType type, const json &js, size_t index, ProtoTensor *const request_tensor);

  Status RecursiveGetArray(const json &json_data, size_t depth, size_t data_index, HTTP_DATA_TYPE type_format,
                           ProtoTensor *const request_tensor);
  Status GetArrayData(const json &js, size_t data_index, HTTP_DATA_TYPE type, ProtoTensor *const request_tensor);

  Status ParseReqCommonMsg(const std::shared_ptr<RestfulRequest> &restful_request,
                           proto::PredictRequest *const request);
  Status ParseRequest(const std::shared_ptr<RestfulRequest> &restful_request, proto::PredictRequest *const request);
  Status ParseInstancesMsg(const json &js_msg, proto::PredictRequest *const request);
  Status GetInstancesType(const json &instances);
  Status ParseKeyInstances(const json &instances, proto::PredictRequest *const request);
  Status PaserKeyOneInstance(const json &instance_msg, proto::PredictRequest *const request);
  Status ParseItem(const json &value, ProtoTensor *const pb_tensor);

  // parse reply:trans RequestReply to http msg
  RequestType GetReqType(const std::string &str);
  std::string GetReqTypeStr(RequestType req_type);
  Status ParseReply(const proto::PredictReply &reply, json *const out_json);
  Status CheckReply(const ProtoTensor &pb_tensor);
  Status ParseInstancesReply(const proto::PredictReply &reply, json *const out_json);
  Status ParseReplyDetail(const proto::Tensor &tensor, json *const js);
  Status ParseScalar(const ProtoTensor &pb_tensor, size_t index, json *const js);
  Status RecursiveParseArray(const ProtoTensor &pb_tensor, size_t depth, size_t pos, json *const out_json);

  template <typename T>
  Status ParseScalarData(const ProtoTensor &pb_tensor, bool is_bytes, size_t index, json *const js);
  template <typename T>
  bool IsString();
  void ParseErrorMsg(const proto::ErrorMsg &error_msg, json *const js);

  RequestType request_type_{kInvalidType};
  InstancesType instances_type_{kInvalidWay};
  int64_t instances_nums_{0};
  std::shared_ptr<Dispatcher> dispatcher_;
  std::vector<std::string> request_type_list_ = {kInstancesRequest};
};

}  // namespace serving
}  // namespace mindspore
#endif  // MINDSPORE_SERVING_MASTER_HTTP_PROCESS_H
