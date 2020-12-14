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
#include "master/restful/http_process.h"
#include <map>
#include <vector>
#include <functional>
#include <utility>
#include <iostream>
#include <sstream>
#include <algorithm>
#include "common/serving_common.h"
#include "master/restful/http_handle.h"

using mindspore::serving::proto::Instance;
using mindspore::serving::proto::PredictReply;
using mindspore::serving::proto::PredictRequest;

namespace mindspore {
namespace serving {
const int BUF_MAX = 0x7FFFFFFF;

static const std::map<DataType, HTTP_DATA_TYPE> infer_type2_http_type{{DataType::kMSI_Int32, HTTP_DATA_INT},
                                                                      {DataType::kMSI_Float32, HTTP_DATA_FLOAT}};

static const std::map<HTTP_DATA_TYPE, DataType> http_type2_infer_type{{HTTP_DATA_INT, DataType::kMSI_Int32},
                                                                      {HTTP_DATA_FLOAT, DataType::kMSI_Float32},
                                                                      {HTTP_DATA_BOOL, DataType::kMSI_Bool},
                                                                      {HTTP_DATA_STR, DataType::kMSI_String},
                                                                      {HTTP_DATA_OBJ, DataType::kMSI_Bytes}};

static const std::map<std::string, DataType> str2_infer_type{
  {"int8", DataType::kMSI_Int8},     {"int16", DataType::kMSI_Int16},   {"int32", DataType::kMSI_Int32},
  {"int64", DataType::kMSI_Int64},   {"uint8", DataType::kMSI_Uint8},   {"uint16", DataType::kMSI_Uint16},
  {"uint32", DataType::kMSI_Uint32}, {"uint64", DataType::kMSI_Uint64}, {"fp16", DataType::kMSI_Float16},
  {"fp32", DataType::kMSI_Float32},  {"fp64", DataType::kMSI_Float64},  {"bool", DataType::kMSI_Bool},
  {"str", DataType::kMSI_String},    {"bytes", DataType::kMSI_Bytes}};

template <typename T>
bool RestfulService::IsString() {
  return typeid(T).hash_code() == typeid(std::string).hash_code();
}

std::string RestfulService::GetString(const uint8_t *ptr, size_t length) {
  std::string str;
  for (size_t i = 0; i < length; i++) {
    str += ptr[i];
  }
  return str;
}

Status RestfulService::CheckObjTypeMatchShape(DataType data_type, const std::vector<int64_t> &shape) {
  if (data_type == kMSI_String || data_type == kMSI_Bytes) {
    size_t elements_nums = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<size_t>());
    if (elements_nums != 1) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
             << "json object, only support scalar when data type is string or bytes";
    }
  }
  return SUCCESS;
}

RequestType RestfulService::GetReqType(const std::string &str) {
  auto it = std::find(request_type_list_.begin(), request_type_list_.end(), str);
  if (it == request_type_list_.end()) {
    return kInvalidType;
  }

  if (*it == kInstancesRequest) {
    return kInstanceType;
  }

  return kInvalidType;
}

std::string RestfulService::GetReqTypeStr(RequestType req_type) {
  switch (req_type) {
    case kInstanceType:
      return kInstancesRequest;
    default:
      break;
  }
  return "";
}

Status RestfulService::CheckObjType(const string &type) {
  Status status(SUCCESS);
  auto it = str2_infer_type.find(type);

  if (it == str2_infer_type.end()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json object, specified type:"
                                                  << "'" << type << "' is illegal";
  }

  return status;
}

DataType RestfulService::GetObjDataType(const json &js) {
  DataType type = kMSI_Unknown;
  if (!js.is_object()) {
    return type;
  }

  auto it1 = js.find(kType);
  if (it1 == js.end()) {
    type = kMSI_Bytes;
  } else {
    auto type_str = it1.value();
    auto it2 = str2_infer_type.find(type_str);
    if (it2 != str2_infer_type.end()) {
      type = it2->second;
    }
  }

  return type;
}

std::string RestfulService::GetStringByDataType(DataType type) {
  for (const auto &item : str2_infer_type) {
    if (item.second == type) {
      return item.first;
    }
  }

  return "";
}

bool RestfulService::JsonMatchDataType(const json &js, DataType type) {
  bool flag = false;
  if (js.is_number_integer()) {
    if (type >= kMSI_Int8 && type <= kMSI_Uint64) {
      flag = true;
    }
  } else if (js.is_number_float()) {
    if (type >= kMSI_Float16 && type <= kMSI_Float64) {
      flag = true;
    }
  } else if (js.is_string()) {
    // string value can express all kinds type
    flag = true;
  } else if (js.is_boolean()) {
    if (type == kMSI_Bool) {
      flag = true;
    }
  }

  return flag;
}

std::vector<int64_t> RestfulService::GetObjShape(const json &js) {
  std::vector<int64_t> shape;
  auto it = js.find(kShape);
  if (it == js.end()) {
    shape.push_back(1);
  } else {
    shape = GetSpecifiedShape(it.value());
  }

  return shape;
}

std::vector<int64_t> RestfulService::GetArrayShape(const json &json_array) {
  std::vector<int64_t> json_shape;
  const json *tmp_json = &json_array;
  while (tmp_json->is_array()) {
    if (tmp_json->empty()) {
      break;
    }

    json_shape.emplace_back(tmp_json->size());
    tmp_json = &tmp_json->at(0);
  }

  return json_shape;
}

std::vector<int64_t> RestfulService::GetSpecifiedShape(const json &js) {
  std::vector<int64_t> shape;
  if (!js.is_array()) {
    return shape;
  }
  if (js.empty()) {
    return shape;
  }

  for (size_t i = 0; i < js.size(); i++) {
    auto &item = js.at(i);
    if (!item.is_number_unsigned()) {
      return {};
    } else {
      shape.push_back(item.get<uint32_t>());
    }
  }

  return shape;
}

DataType RestfulService::GetArrayDataType(const json &json_array, HTTP_DATA_TYPE *type_format_ptr) {
  MSI_EXCEPTION_IF_NULL(type_format_ptr);
  auto &type_format = *type_format_ptr;
  DataType data_type = kMSI_Unknown;
  const json *tmp_json = &json_array;
  while (tmp_json->is_array()) {
    if (tmp_json->empty()) {
      return data_type;
    }

    tmp_json = &tmp_json->at(0);
  }

  if (tmp_json->is_number_integer()) {
    type_format = HTTP_DATA_INT;
    data_type = http_type2_infer_type.at(type_format);
  } else if (tmp_json->is_number_float()) {
    type_format = HTTP_DATA_FLOAT;
    data_type = http_type2_infer_type.at(type_format);
  } else if (tmp_json->is_boolean()) {
    type_format = HTTP_DATA_BOOL;
    data_type = http_type2_infer_type.at(type_format);
  } else if (tmp_json->is_object()) {
    type_format = HTTP_DATA_OBJ;
    data_type = GetObjDataType(*tmp_json);
  } else if (tmp_json->is_string()) {
    type_format = HTTP_DATA_STR;
    data_type = http_type2_infer_type.at(type_format);
  }

  return data_type;
}

Status RestfulService::CheckReqJsonValid(const json &js_msg) {
  int count = 0;
  for (size_t i = 0; i < request_type_list_.size(); i++) {
    std::string item = request_type_list_.at(i);
    auto it = js_msg.find(item.c_str());
    if (it != js_msg.end()) {
      count++;
      auto request_type = GetReqType(item);
      if (request_type == kInvalidType) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "only support instances mode";
      }

      request_type_ = request_type;
    }
  }

  if (count != 1) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "key 'instances' should exit and only exit one time";
  }
  return SUCCESS;
}

Status RestfulService::GetInstancesType(const json &instances) {
  Status status{SUCCESS};
  // Eg:{"instances" : 1}
  if (!(instances.is_array() || instances.is_object())) {
    instances_type_ = kNokeyWay;
    return status;
  }

  // Eg:{"instances":{"A":1, "B":2}}
  if (instances.is_object()) {
    instances_type_ = kKeyWay;
    return status;
  }

  // array:
  if (instances.empty()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "instances value is array type, but no value";
  }
  auto first_instance = instances.at(0);
  if (first_instance.is_object()) {
    instances_type_ = kKeyWay;
  } else {
    instances_type_ = kNokeyWay;
  }

  return status;
}

Status RestfulService::CheckObj(const json &js) {
  if (!js.is_object()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json is not object" << js.dump();
  }

  if (js.empty()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json object, value is empty";
  }

  // 1)required:b64 2)optional:type 3)optional:shape
  if (js.size() > 3) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
           << "json object, items size is more than 3, only support specified ['b64', 'type', 'shape']";
  }

  int b64_count = 0;
  int shape_count = 0;
  int type_count = 0;
  for (auto item = js.begin(); item != js.end(); ++item) {
    auto key = item.key();
    auto value = item.value();
    if (key != kB64 && key != kType && key != kShape) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
             << "json object, key is not ['b64', 'type', 'shape'], fail key:" << key;
    }
    if (key == kB64) {
      b64_count++;
    } else if (key == kType) {
      if (!value.is_string()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json object, key is 'type', value should be string type";
      }
      auto status = CheckObjType(value);
      if (status != SUCCESS) {
        return status;
      }
      type_count++;
    } else if (key == kShape) {
      if (!value.is_array()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json object, key is 'shape', value should be array type";
      }
      if (value.empty()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json object, key is 'shape', array value should no be empty";
      }
      for (auto it = value.begin(); it != value.end(); ++it) {
        if (!(it->is_number())) {
          return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json object, key is 'shape', array value should be number";
        }
        auto number = it->get<int32_t>();
        if (number <= 0) {
          return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
                 << "json object, key is 'shape', number value should be positive number";
        }
      }
      shape_count++;
    }
  }

  if (b64_count != 1) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json object, 'b64' should be specified only one time";
  }

  if (type_count > 1) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json object, 'type' should be specified no more than one time";
  }

  if (shape_count > 1) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json object, 'shape' should be specified no more than one time";
  }

  return SUCCESS;
}

// 1. parse request common func
Status RestfulService::ParseItem(const json &value, ProtoTensor *const pb_tensor) {
  Status status(SUCCESS);
  if (value.is_number_integer()) {
    DataType type = kMSI_Int32;
    pb_tensor->set_data_type(type);
    pb_tensor->set_shape({1});
    pb_tensor->resize_data(pb_tensor->GetTypeSize(type));
    status = GetScalarByType(type, value, 0, pb_tensor);
  } else if (value.is_number_float()) {
    DataType type = kMSI_Float32;
    pb_tensor->set_data_type(type);
    pb_tensor->set_shape({1});
    pb_tensor->resize_data(pb_tensor->GetTypeSize(type));
    status = GetScalarByType(type, value, 0, pb_tensor);
  } else if (value.is_boolean()) {
    DataType type = kMSI_Bool;
    pb_tensor->set_data_type(type);
    pb_tensor->set_shape({1});
    pb_tensor->resize_data(pb_tensor->GetTypeSize(type));
    status = GetScalarByType(type, value, 0, pb_tensor);
  } else if (value.is_string()) {
    DataType type = kMSI_String;
    pb_tensor->set_data_type(type);
    pb_tensor->set_shape({1});
    status = GetScalarByType(type, value, 0, pb_tensor);
  } else if (value.is_object()) {
    status = CheckObj(value);
    if (status != SUCCESS) {
      return status;
    }

    DataType type = GetObjDataType(value);
    if (type == kMSI_Unknown) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json object, type is unknown";
    }

    std::vector<int64_t> shape = GetObjShape(value);
    if (shape.empty()) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "object json, shape is empty";
    }

    bool is_tensor = false;
    if (type != kMSI_String && type != kMSI_Bytes) {
      is_tensor = true;
    }
    if (is_tensor) {
      size_t shape_size = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<size_t>());
      size_t type_size = pb_tensor->GetTypeSize(type);
      pb_tensor->resize_data(shape_size * type_size);
      MSI_LOG_DEBUG << "shape size:" << shape_size << "; type size:" << type_size
                    << "; data size:" << shape_size * type_size;
    }

    status = CheckObjTypeMatchShape(type, shape);
    if (status != SUCCESS) {
      return status;
    }
    pb_tensor->set_data_type(type);
    pb_tensor->set_shape(shape);
    status = GetScalarByType(serving::kMSI_Bytes, value[kB64], 0, pb_tensor);
  } else if (value.is_array()) {
    HTTP_DATA_TYPE type_format = HTTP_DATA_NONE;
    auto shape = GetArrayShape(value);
    if (shape.empty()) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json array, shape is empty";
    }
    DataType data_type = GetArrayDataType(value, &type_format);
    if (data_type == kMSI_Unknown) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json array, data type is unknown";
    }

    bool is_tensor = false;
    if (data_type != kMSI_String && data_type != kMSI_Bytes) {
      is_tensor = true;
    }

    // intances mode:only support one item
    if (request_type_ == kInstanceType) {
      if (!is_tensor) {
        size_t elements_nums = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<size_t>());
        if (elements_nums != 1) {
          return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json array, string or bytes type only support one item";
        }
      }
    }

    // set real data type
    pb_tensor->set_data_type(data_type);
    pb_tensor->set_shape(shape);

    if (is_tensor) {
      size_t shape_size = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<size_t>());
      size_t type_size = pb_tensor->GetTypeSize(data_type);
      pb_tensor->resize_data(shape_size * type_size);
      MSI_LOG_DEBUG << "shape size:" << shape_size << "; type size:" << type_size
                    << "; data size:" << shape_size * type_size;
    }

    if (type_format == HTTP_DATA_OBJ) {
      if (data_type != kMSI_Bytes && data_type != kMSI_String) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
               << "json array, item is object type, object only support string or bytes type";
      }
    }
    status = RecursiveGetArray(value, 0, 0, type_format, pb_tensor);
    if (status != SUCCESS) {
      return status;
    }
  } else if (value.is_null()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json value is null, it is not supported";
  } else if (value.is_discarded()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json value is discarded type, it is not supported";
  }
  return status;
}

Status RestfulService::RecursiveGetArray(const json &json_data, size_t depth, size_t data_index,
                                         HTTP_DATA_TYPE type_format, ProtoTensor *const request_tensor) {
  Status status(SUCCESS);
  std::vector<int64_t> required_shape = request_tensor->shape();
  if (depth >= required_shape.size()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
           << "current depth:" << depth << " is more than shape dims:" << required_shape.size();
  }
  if (!json_data.is_array()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json type is not array";
  }
  if (json_data.size() != static_cast<size_t>(required_shape[depth])) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
           << "json size is:" << json_data.size() << "; the " << depth << " dim need"
           << " shape size:" << required_shape[depth];
  }
  if (depth + 1 < required_shape.size()) {
    size_t sub_element_cnt =
      std::accumulate(required_shape.begin() + depth + 1, required_shape.end(), 1LL, std::multiplies<size_t>());
    for (size_t k = 0; k < json_data.size(); k++) {
      status =
        RecursiveGetArray(json_data[k], depth + 1, data_index + sub_element_cnt * k, type_format, request_tensor);
      if (status != SUCCESS) {
        return status;
      }
    }
  } else {
    status = GetArrayData(json_data, data_index, type_format, request_tensor);
    if (status != SUCCESS) {
      return status;
    }
  }
  return status;
}

Status RestfulService::GetArrayData(const json &js, size_t data_index, HTTP_DATA_TYPE type,
                                    ProtoTensor *const request_tensor) {
  Status status(SUCCESS);
  size_t element_nums = js.size();
  MSI_LOG_DEBUG << "element nums:" << element_nums << "; data index:" << data_index;
  if (type != HTTP_DATA_OBJ) {
    for (size_t k = 0; k < element_nums; k++) {
      auto &json_data = js[k];
      if (!(json_data.is_number() || json_data.is_boolean() || json_data.is_string())) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json array, data should be number, bool, string or bytes";
      }
      auto flag = JsonMatchDataType(json_data, request_tensor->data_type());
      if (!flag) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json array, elements type is not equal";
      }
      status = GetScalarByType(request_tensor->data_type(), json_data, data_index + k, request_tensor);
      if (status != SUCCESS) {
        return status;
      }
    }
  } else {
    for (size_t k = 0; k < element_nums; k++) {
      auto &json_data = js[k];
      auto value_type = GetObjDataType(json_data);
      // Array:object only support string or bytes
      if (value_type != kMSI_String && value_type != kMSI_Bytes) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json array, object type only support string or bytes type";
      }

      if (value_type != request_tensor->data_type()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json array, elements type is not equal";
      }

      status = GetScalarByType(value_type, json_data[kB64], data_index + k, request_tensor);
      if (status != SUCCESS) {
        return status;
      }
    }
  }
  return status;
}

Status RestfulService::GetScalarByType(DataType type, const json &js, size_t index, ProtoTensor *const request_tensor) {
  Status status(SUCCESS);
  if (type == kMSI_Unknown) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "data type is unknown";
  }
  MSI_LOG_DEBUG << "data type:" << type << " ;real data type:" << request_tensor->data_type()
                << " ;data index:" << index;
  switch (type) {
    case kMSI_Bool:
      status = GetScalarData<bool>(js, index, false, request_tensor);
      break;
    case kMSI_Int8:
      status = GetScalarData<int8_t>(js, index, false, request_tensor);
      break;
    case kMSI_Int16:
      status = GetScalarData<int16_t>(js, index, false, request_tensor);
      break;
    case kMSI_Int32:
      status = GetScalarData<int32_t>(js, index, false, request_tensor);
      break;
    case kMSI_Int64:
      status = GetScalarData<int64_t>(js, index, false, request_tensor);
      break;
    case kMSI_Uint8:
      status = GetScalarData<uint8_t>(js, index, false, request_tensor);
      break;
    case kMSI_Uint16:
      status = GetScalarData<uint16_t>(js, index, false, request_tensor);
      break;
    case kMSI_Uint32:
      status = GetScalarData<uint32_t>(js, index, false, request_tensor);
      break;
    case kMSI_Uint64:
      status = GetScalarData<uint64_t>(js, index, false, request_tensor);
      break;
    case kMSI_Float16:
      status = GetScalarData<float>(js, index, false, request_tensor);
      break;
    case kMSI_Float32:
      status = GetScalarData<float>(js, index, false, request_tensor);
      break;
    case kMSI_Float64:
      status = GetScalarData<double>(js, index, false, request_tensor);
      break;
    case kMSI_String:
      status = GetScalarData<std::string>(js, index, false, request_tensor);
      break;
    case kMSI_Bytes:
      status = GetScalarData<std::string>(js, index, true, request_tensor);
      break;
    default:
      auto type_str = GetStringByDataType(type);
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "data type:" << type_str << " is not supported";
  }
  return status;
}

template <typename T>
Status RestfulService::GetScalarData(const json &js, size_t index, bool is_bytes, ProtoTensor *const request_tensor) {
  Status status(SUCCESS);
  if (IsString<T>()) {
    // 1.string
    if (!js.is_string()) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
             << "get scalar data failed, type is string, but json is not string type";
    }

    auto value = js.get<std::string>();
    if (value.empty()) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "string value is empty";
    }
    if (is_bytes) {
      auto tail_equal_size = GetTailEqualSize(value);
      if (tail_equal_size == UINT32_MAX) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "'" << value << "' is illegal b64 encode string";
      }
      auto origin_size = GetB64OriginSize(value.length(), tail_equal_size);
      std::vector<uint8_t> buffer(origin_size, 0);
      auto target_size = Base64Decode(reinterpret_cast<uint8_t *>(value.data()), value.length(), buffer.data());
      if (target_size != origin_size) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "decode base64 failed, size is not matched.";
      }
      DataType real_type = request_tensor->data_type();
      if (real_type == kMSI_Bytes || real_type == kMSI_String) {
        request_tensor->add_bytes_data(buffer.data(), origin_size);
      } else {
        auto type_size = request_tensor->GetTypeSize(real_type);
        auto element_cnt = request_tensor->element_cnt();
        if (origin_size != type_size * element_cnt) {
          return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
                 << "size is not matched, decode base64 size:" << origin_size
                 << "; Given info: type:" << GetStringByDataType(real_type) << "; type size:" << type_size
                 << "; element nums:" << element_cnt;
        }

        auto data = reinterpret_cast<T *>(request_tensor->mutable_data()) + index;
        memcpy_s(data, origin_size, buffer.data(), buffer.size());
      }
    } else {
      request_tensor->add_bytes_data(reinterpret_cast<uint8_t *>(value.data()), value.length());
    }
  } else {
    DataType data_type = request_tensor->data_type();
    auto flag = JsonMatchDataType(js, data_type);
    if (!flag) {
      auto type_str = GetStringByDataType(data_type);
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
             << "data type and json type is not matched, data type is:" << type_str;
    }

    // 2.number
    if ((js.is_number() || js.is_boolean())) {
      // 1)common number
      auto data = reinterpret_cast<T *>(request_tensor->mutable_data()) + index;
      *data = js.get<T>();
    }
  }

  return status;
}

// 2.main
Status RestfulService::RunRestful(const std::shared_ptr<RestfulRequest> &restful_request, json *const out_json) {
  PredictRequest request;
  PredictReply reply;

  MSI_TIME_STAMP_START(ParseRequest)
  auto status = ParseRequest(restful_request, &request);
  MSI_TIME_STAMP_END(ParseRequest)
  if (status != SUCCESS) {
    std::string error_msg = status.StatusMessage();
    std::string msg = "Paser reqeust failed, " + error_msg;
    status = msg;
    return status;
  }

  MSI_TIME_STAMP_START(Predict)
  status = dispatcher_->Dispatch(request, &reply);
  MSI_TIME_STAMP_END(Predict)
  if (status != SUCCESS) {
    std::string error_msg = status.StatusMessage();
    std::string msg = "Predict failed, " + error_msg;
    status = msg;
    return status;
  }

  MSI_TIME_STAMP_START(CreateReplyJson)
  status = ParseReply(reply, out_json);
  MSI_TIME_STAMP_END(CreateReplyJson)
  if (status != SUCCESS) {
    std::string error_msg = status.StatusMessage();
    std::string msg = "Parse reply failed, " + error_msg;
    status = msg;
    return status;
  }
  return SUCCESS;
}

// 3.parse request
Status RestfulService::ParseRequest(const std::shared_ptr<RestfulRequest> &restful_request,
                                    PredictRequest *const request) {
  Status status(SUCCESS);
  // 1. parse common msg
  status = ParseReqCommonMsg(restful_request, request);
  if (status != SUCCESS) {
    return status;
  }

  // 2. parse json
  auto request_ptr = restful_request->decompose_event_request();
  auto &js_msg = request_ptr->request_message_;
  status = CheckReqJsonValid(js_msg);
  if (status != SUCCESS) {
    return status;
  }

  switch (request_type_) {
    case kInstanceType:
      status = ParseInstancesMsg(js_msg, request);
      break;
    default:
      return INFER_STATUS_LOG_ERROR(FAILED) << "restful reqeust only support instances mode";
  }

  PrintRequest(request);
  return status;
}

Status RestfulService::ParseReqCommonMsg(const std::shared_ptr<RestfulRequest> &restful_request,
                                         PredictRequest *const request) {
  Status status(SUCCESS);
  auto request_ptr = restful_request->decompose_event_request();
  if (request_ptr == nullptr) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "Decompose event request is nullptr");
    return status;
  }
  request->mutable_servable_spec()->set_name(request_ptr->model_name_);
  request->mutable_servable_spec()->set_version_number(request_ptr->version_);
  request->mutable_servable_spec()->set_method_name(request_ptr->service_method_);
  MSI_LOG_INFO << "Restful req, model name:" << request->servable_spec().name();
  MSI_LOG_INFO << "Version number:" << request->servable_spec().version_number();
  MSI_LOG_INFO << "Method name:" << request->servable_spec().method_name();
  return status;
}

Status RestfulService::ParseInstancesMsg(const json &js_msg, PredictRequest *const request) {
  Status status = SUCCESS;
  auto type = GetReqTypeStr(request_type_);
  auto instances = js_msg.find(type);
  if (instances == js_msg.end()) {
    ERROR_INFER_STATUS(status, FAILED, "instances request json should have instances key word");
    return status;
  }

  // get instances way:{key, value} or {value}
  status = GetInstancesType(*instances);
  if (status != SUCCESS) {
    return status;
  }

  switch (instances_type_) {
    case kKeyWay: {
      status = ParseKeyInstances(*instances, request);
      break;
    }
    case kNokeyWay: {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "instances no key mode is not supported";
    }
    case kInvalidWay: {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "invalid request type";
    }
  }
  return status;
}

Status RestfulService::ParseKeyInstances(const json &instances, PredictRequest *const request) {
  Status status(SUCCESS);
  if (instances.is_object()) {
    // one instance:{"instances"：{"A":1, "B": 2}}
    if (instances.empty()) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json object, value is empty";
    }
    status = PaserKeyOneInstance(instances, request);
    if (status != SUCCESS) {
      MSI_LOG_ERROR << "instances:parse one instance failed";
      return status;
    }
    instances_nums_ = 1;
  } else {
    // multi instance:{"instances":[{}, {}]}
    for (size_t i = 0; i < instances.size(); i++) {
      auto &instance = instances.at(i);
      if (!instance.is_object()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json array, instance is not object type";
      }

      if (instance.empty()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "json array, instance is object type, but no value";
      }

      status = PaserKeyOneInstance(instance, request);
      if (status != SUCCESS) {
        return status;
      }
    }
    instances_nums_ = instances.size();
  }
  return status;
}

// instance_mgs:one instance, type is object
Status RestfulService::PaserKeyOneInstance(const json &instance_msg, PredictRequest *const request) {
  Status status(SUCCESS);
  auto instance = request->add_instances();

  for (auto it = instance_msg.begin(); it != instance_msg.end(); ++it) {
    auto key = it.key();
    if (key.empty()) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "string key is empty";
    }
    auto value = it.value();

    auto &map_item = *(instance->mutable_items());
    proto::Tensor &tensor = map_item[key];
    ProtoTensor pb_tensor(&tensor);

    status = ParseItem(value, &pb_tensor);
    if (status != SUCCESS) {
      return status;
    }
  }
  return status;
}

/************************************************************************************/
// 4.parse reply common func
Status RestfulService::ParseReplyDetail(const proto::Tensor &tensor, json *const js) {
  Status status(SUCCESS);
  const ProtoTensor pb_tensor(const_cast<proto::Tensor *>(&tensor));
  auto shape = pb_tensor.shape();
  size_t shape_size = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<size_t>());

  if (shape_size == 1) {
    status = ParseScalar(pb_tensor, 0, js);
    if (status != SUCCESS) {
      return status;
    }
  } else {
    status = CheckReply(pb_tensor);
    if (status != SUCCESS) {
      return status;
    }
    status = RecursiveParseArray(pb_tensor, 0, 0, js);
    if (status != SUCCESS) {
      return status;
    }
  }
  return status;
}

Status RestfulService::ParseScalar(const ProtoTensor &pb_tensor, size_t index, json *const js) {
  Status status(SUCCESS);
  DataType data_type = pb_tensor.data_type();
  if (data_type == kMSI_Unknown) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Data type is unknown";
  }
  MSI_LOG_DEBUG << "Data type:" << data_type << "; index:" << index;
  switch (data_type) {
    case kMSI_Bool:
      status = ParseScalarData<bool>(pb_tensor, false, index, js);
      break;
    case kMSI_Int8:
      status = ParseScalarData<int8_t>(pb_tensor, false, index, js);
      break;
    case kMSI_Int16:
      status = ParseScalarData<int16_t>(pb_tensor, false, index, js);
      break;
    case kMSI_Int32:
      status = ParseScalarData<int32_t>(pb_tensor, false, index, js);
      MSI_LOG_INFO << "parse int32:" << js->get<int32_t>();
      break;
    case kMSI_Int64:
      status = ParseScalarData<int64_t>(pb_tensor, false, index, js);
      MSI_LOG_INFO << "parse int64:" << js->get<int64_t>();
      break;
    case kMSI_Uint8:
      status = ParseScalarData<uint8_t>(pb_tensor, false, index, js);
      break;
    case kMSI_Uint16:
      status = ParseScalarData<uint16_t>(pb_tensor, false, index, js);
      break;
    case kMSI_Uint32:
      status = ParseScalarData<uint32_t>(pb_tensor, false, index, js);
      break;
    case kMSI_Uint64:
      status = ParseScalarData<uint64_t>(pb_tensor, false, index, js);
      break;
    case kMSI_Float16:
      ERROR_INFER_STATUS(status, FAILED, "fp16 reply is not supported");
      break;
    case kMSI_Float32:
      status = ParseScalarData<float>(pb_tensor, false, index, js);
      break;
    case kMSI_Float64:
      status = ParseScalarData<double>(pb_tensor, false, index, js);
      break;
    case kMSI_String:
      status = ParseScalarData<std::string>(pb_tensor, false, index, js);
      break;
    case kMSI_Bytes:
      status = ParseScalarData<std::string>(pb_tensor, true, index, js);
      break;
    default:
      ERROR_INFER_STATUS(status, FAILED, "reply data type is not supported");
      break;
  }
  return status;
}

template <typename T>
Status RestfulService::ParseScalarData(const ProtoTensor &pb_tensor, bool is_bytes, size_t index, json *const js) {
  Status status(SUCCESS);

  if (!IsString<T>()) {
    const T *data = reinterpret_cast<const T *>(pb_tensor.data()) + index;
    T value = *data;
    *js = value;
  } else if (IsString<T>()) {
    if (!is_bytes) {
      auto str_nums = pb_tensor.bytes_data_size();
      if (str_nums == 0) {
        return INFER_STATUS_LOG_ERROR(FAILED) << "reply string, size is 0";
      }
      if (index >= str_nums) {
        return INFER_STATUS_LOG_ERROR(FAILED) << "reply string, index:" << index << " is more than size:" << str_nums;
      }

      std::string value;
      size_t length;
      const uint8_t *ptr = nullptr;
      pb_tensor.get_bytes_data(index, &ptr, &length);
      value.resize(length);
      memcpy_s(value.data(), length, reinterpret_cast<const char *>(ptr), length);
      MSI_LOG_INFO << "Parse string value:" << value;
      *js = value;
    } else {
      auto str_nums = pb_tensor.bytes_data_size();
      if (str_nums == 0) {
        return INFER_STATUS_LOG_ERROR(FAILED) << "reply bytes, size is 0";
      }

      if (index >= str_nums) {
        return INFER_STATUS_LOG_ERROR(FAILED) << "reply bytes, index:" << index << " is more than size:" << str_nums;
      }

      std::string value;
      size_t length;
      const uint8_t *ptr = nullptr;
      pb_tensor.get_bytes_data(index, &ptr, &length);
      value.resize(length);
      memcpy_s(value.data(), length, reinterpret_cast<const char *>(ptr), length);
      MSI_LOG_DEBUG << "bytes type, origin str:" << value;

      auto target_size = GetB64TargetSize(length);
      std::vector<uint8_t> buffer(target_size, 0);
      auto size = Base64Encode(reinterpret_cast<uint8_t *>(value.data()), value.length(), buffer.data());
      if (size != target_size) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "reply bytes, size is not matched, expected size:" << target_size << ", encode size:" << size;
      }
      std::string str = GetString(buffer.data(), buffer.size());
      MSI_LOG_DEBUG << "bytes type, decoded str:" << str;
      (*js)[kB64] = str;
    }
  }
  return status;
}

Status RestfulService::RecursiveParseArray(const ProtoTensor &pb_tensor, size_t depth, size_t pos,
                                           json *const out_json) {
  Status status(SUCCESS);
  std::vector<int64_t> required_shape = pb_tensor.shape();
  if (depth >= 10) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "result shape dims is larger than 10";
  }
  if (depth == required_shape.size() - 1) {
    for (int i = 0; i < required_shape[depth]; i++) {
      out_json->push_back(json());
      json &scalar_json = out_json->back();
      status = ParseScalar(pb_tensor, pos + i, &scalar_json);
      if (status != SUCCESS) {
        return status;
      }
    }
  } else {
    for (int i = 0; i < required_shape[depth]; i++) {
      // array:
      out_json->push_back(json());
      json &tensor_json = out_json->back();
      size_t sub_element_cnt =
        std::accumulate(required_shape.begin() + depth + 1, required_shape.end(), 1LL, std::multiplies<size_t>());
      status = RecursiveParseArray(pb_tensor, depth + 1, i * sub_element_cnt + pos, &tensor_json);
      if (status != SUCCESS) {
        return status;
      }
    }
  }
  return status;
}

Status RestfulService::CheckReply(const ProtoTensor &pb_tensor) {
  Status status(SUCCESS);
  DataType data_type = pb_tensor.data_type();
  if (data_type == kMSI_Unknown) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "reply data type is unknown";
  }

  if (data_type == kMSI_String || data_type == kMSI_Bytes) {
    auto shape = pb_tensor.shape();
    if (shape.size() != 1) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "reply string or bytes, shape should be 1, given shape size:" << shape.size();
    }
  }
  return status;
}

void RestfulService::ParseErrorMsg(const proto::ErrorMsg &error, json *const js) {
  std::string str = error.error_msg();
  *js = str;
}

// 5.Parse reply
Status RestfulService::ParseReply(const PredictReply &reply, json *const out_json) {
  Status status(SUCCESS);
  switch (request_type_) {
    case kInstanceType:
      status = ParseInstancesReply(reply, out_json);
      break;
    default:
      return INFER_STATUS_LOG_ERROR(FAILED) << "restful request only support instance mode";
  }

  PrintReply(reply);
  return status;
}

Status RestfulService::ParseInstancesReply(const PredictReply &reply, json *const out_json) {
  Status status(SUCCESS);
  auto error_size = reply.error_msg_size();
  if (error_size != 0 && error_size != 1 && error_size != instances_nums_) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "reply error size:" << error_size << " is not 0,1 or instances size";
  }

  (*out_json)[kInstancesReply] = json();
  json &instances_json = (*out_json)[kInstancesReply];

  int32_t reply_num = instances_nums_;
  if (reply.instances().empty()) {
    reply_num = error_size;
  }
  for (int32_t i = 0; i < reply_num; i++) {
    bool success_flag = true;
    if (i < error_size) {
      auto &cur_error = reply.error_msg().at(i);
      success_flag = (cur_error.error_code() == 0);
    }

    if (success_flag) {
      if (i >= reply.instances_size()) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "index:" << i << " is more than reply instances size:" << reply.instances_size();
      }
      auto &cur_instance = reply.instances(i);
      auto &items = cur_instance.items();
      if (items.empty()) {
        return INFER_STATUS_LOG_ERROR(FAILED) << "reply instance items is empty";
      }
      instances_json.push_back(json());
      auto &instance = instances_json.back();

      for (auto &item : items) {
        instance[item.first] = json();
        auto &value_json = instance[item.first];
        status = ParseReplyDetail(item.second, &value_json);
        if (status != SUCCESS) {
          return status;
        }
      }
    } else {
      instances_json.push_back(json());
      auto &obj = instances_json.back();
      obj[kErrorMsg] = json();
      auto &js = obj[kErrorMsg];
      ParseErrorMsg(reply.error_msg(i), &js);
    }
  }
  return status;
}

// For test, to be deleted
void RestfulService::FadeReply(const proto::PredictRequest &request, proto::PredictReply *reply) {
  MSI_EXCEPTION_IF_NULL(reply);
  MSI_LOG_INFO << "Start";
  if (request_type_ == kInstanceType) {
    size_t instances_size = request.instances_size();
    for (size_t i = 0; i < instances_size; i++) {
      auto cur_instance = request.instances(i);
      auto target_ptr = reply->add_instances();
      for (const auto &item : cur_instance.items()) {
        auto key = item.first;
        auto tensor = item.second;
        auto &map_item = *(target_ptr->mutable_items());
        map_item[key] = tensor;
      }
    }
  } else {
    MSI_LOG_ERROR << "only support two kind type";
  }
  MSI_LOG_INFO << "End";
}  // namespace serving

void RestfulService::PrintRequest(const proto::PredictRequest *const request) {
  MSI_LOG_INFO << "=============start print quest==================";
  if (request_type_ == kInstanceType) {
    size_t size = request->instances_size();
    MSI_LOG_INFO << "instance size:" << request->instances_size();
    for (size_t i = 0; i < size; i++) {
      auto instance = request->instances(i);
      auto &cur_map = instance.items();
      MSI_LOG_INFO << "map size:" << cur_map.size();
      for (const auto &item : cur_map) {
        MSI_LOG_INFO << "======deail instance=======";
        auto key = item.first;
        auto &tensor = item.second;
        const ProtoTensor pb_tensor(const_cast<proto::Tensor *>(&tensor));
        MSI_LOG_INFO << "key:" << key;
        DataType type = pb_tensor.data_type();
        MSI_LOG_INFO << "data type:" << type;
        auto shape = pb_tensor.shape();
        for (const auto &item1 : shape) {
          MSI_LOG_INFO << "shape:" << item1;
        }

        size_t data_size = pb_tensor.data_size();
        MSI_LOG_INFO << "data size:" << data_size;

        auto data = reinterpret_cast<const int32_t *>(pb_tensor.data());
        MSI_LOG_INFO << "data value:" << data;

        size_t bytes_data_size = pb_tensor.bytes_data_size();
        MSI_LOG_INFO << "bytest data size:" << bytes_data_size;
        for (size_t k = 0; k < bytes_data_size; k++) {
          const uint8_t *data1;
          size_t bytes_len;
          pb_tensor.get_bytes_data(k, &data1, &bytes_len);
          MSI_LOG_INFO << "start ptr:" << data1 << "; bytes length:" << bytes_len;
        }
      }
    }
  } else {
    MSI_LOG_ERROR << "=========Print Failed";
  }
  MSI_LOG_INFO << "=============End print request==================";
}

void RestfulService::PrintReply(const proto::PredictReply &reply) {
  MSI_LOG_INFO << "=============start print reply==================";
  if (request_type_ == kInstanceType) {
    size_t size = reply.instances_size();
    MSI_LOG_INFO << "instance size:" << reply.instances_size();
    for (size_t i = 0; i < size; i++) {
      auto instance = reply.instances(i);
      auto &cur_map = instance.items();
      MSI_LOG_INFO << "map size:" << cur_map.size();
      for (const auto &item : cur_map) {
        MSI_LOG_INFO << "======deail instance=======";
        auto key = item.first;
        auto &tensor = item.second;
        const ProtoTensor pb_tensor(const_cast<proto::Tensor *>(&tensor));
        MSI_LOG_INFO << "key:" << key;
        DataType type = pb_tensor.data_type();
        MSI_LOG_INFO << "data type:" << type;
        auto shape = pb_tensor.shape();
        for (const auto &item1 : shape) {
          MSI_LOG_INFO << "shape:" << item1;
        }

        size_t data_size = pb_tensor.data_size();
        MSI_LOG_INFO << "data size:" << data_size;
        size_t bytes_data_size = pb_tensor.bytes_data_size();
        MSI_LOG_INFO << "bytest data size:" << bytes_data_size;
        for (size_t k = 0; k < bytes_data_size; k++) {
          const uint8_t *data;
          size_t bytes_len;
          pb_tensor.get_bytes_data(k, &data, &bytes_len);
          MSI_LOG_INFO << "start ptr:" << data << "; bytes length:" << bytes_len;
        }
      }
    }
  } else {
    MSI_LOG_ERROR << "=========Print Failed";
  }
}

}  // namespace serving
}  // namespace mindspore
