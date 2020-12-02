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
#include "common/tensor_base.h"
#include <functional>
#include <string>
#include "common/log.h"
#include "securec.h"

#define TENSOR_MAX_ELEMENT_COUNT UINT32_MAX

namespace mindspore::serving {

bool TensorBase::set_data(const void *data, size_t data_len) {
  resize_data(data_len);
  if (mutable_data() == nullptr) {
    MSI_LOG_ERROR << "set data failed, data len " << data_len;
    return false;
  }
  if (data_size() != data_len) {
    MSI_LOG_ERROR << "set data failed, tensor current data size " << data_size() << " not match data len " << data_len;
    return false;
  }
  if (data_len == 0) {
    return true;
  }
  memcpy_s(mutable_data(), data_size(), data, data_len);
  return true;
}

size_t TensorBase::itemsize() const { return GetTypeSize(data_type()); }

size_t TensorBase::element_cnt() const {
  size_t element_num = 1;
  for (auto dim : shape()) {
    if (dim <= 0 || TENSOR_MAX_ELEMENT_COUNT / static_cast<size_t>(dim) < element_num) {
      return 0;
    }
    element_num *= dim;
  }
  return element_num;
}

size_t TensorBase::GetTypeSize(DataType type) {
  const std::map<DataType, size_t> type_size_map{
    {kMSI_Bool, sizeof(bool)},       {kMSI_Float64, sizeof(double)},   {kMSI_Int8, sizeof(int8_t)},
    {kMSI_Uint8, sizeof(uint8_t)},   {kMSI_Int16, sizeof(int16_t)},    {kMSI_Uint16, sizeof(uint16_t)},
    {kMSI_Int32, sizeof(int32_t)},   {kMSI_Uint32, sizeof(uint32_t)},  {kMSI_Int64, sizeof(int64_t)},
    {kMSI_Uint64, sizeof(uint64_t)}, {kMSI_Float16, sizeof(uint16_t)}, {kMSI_Float32, sizeof(float)},
  };
  auto it = type_size_map.find(type);
  if (it != type_size_map.end()) {
    return it->second;
  }
  return 0;
}

void TensorBase::assgin(const TensorBase &other) {
  if (is_bytes_val_data()) {
    clear_bytes_data();
  }
  set_shape(other.shape());
  set_data_type(other.data_type());
  if (other.is_bytes_val_data()) {
    for (size_t i = 0; i < other.bytes_data_size(); i++) {
      const uint8_t *data;
      size_t data_len;
      other.get_bytes_data(i, &data, &data_len);
      add_bytes_data(data, data_len);
    }
  } else {
    set_data(other.data(), other.data_size());
  }
}
Status TensorBase::concat(const std::vector<TensorBasePtr> &inputs) {
  if (inputs.empty()) {
    MSI_LOG_ERROR << "inputs is empty";
    return FAILED;
  }
  if (is_bytes_val_data()) {
    clear_bytes_data();
  }
  auto &input0 = inputs[0];
  std::vector<int64_t> new_shape = input0->shape();
  new_shape.insert(new_shape.begin(), static_cast<int64_t>(inputs.size()));
  set_shape(new_shape);
  set_data_type(input0->data_type());
  if (input0->is_bytes_val_data()) {
    if (input0->bytes_data_size() != 1) {
      MSI_LOG_ERROR << "input 0 bytes data batch size " << input0->bytes_data_size() << " is not 1";
      return FAILED;
    }
    const uint8_t *data;
    size_t data_len;
    input0->get_bytes_data(0, &data, &data_len);
    add_bytes_data(data, data_len);
  } else {
    resize_data(input0->data_size() * inputs.size());
    memcpy_s(mutable_data(), data_size(), input0->data(), input0->data_size());
  }
  for (size_t i = 1; i < inputs.size(); i++) {
    auto &other = inputs[i];
    if (input0->data_type() != other->data_type()) {
      MSI_LOG_ERROR << "input " << i << " data type " << other->data_type() << " not match input 0 "
                    << input0->data_type();
      return FAILED;
    }
    if (input0->shape() != other->shape()) {
      MSI_LOG_ERROR << "input " << i << " shape " << other->shape() << " not match input 0 " << input0->shape();
      return FAILED;
    }
    if (input0->is_bytes_val_data()) {
      if (other->bytes_data_size() != 1) {
        MSI_LOG_ERROR << "input " << i << " bytes data batch size " << other->bytes_data_size() << " is not 1";
        return FAILED;
      }
      const uint8_t *data = nullptr;
      size_t data_len = 0;
      other->get_bytes_data(0, &data, &data_len);
      add_bytes_data(data, data_len);
    } else {
      if (input0->data_size() != other->data_size()) {
        MSI_LOG_ERROR << "input " << i << " data size " << other->data_size() << " not match input 0 "
                      << input0->data_size();
        return FAILED;
      }
      memcpy_s(mutable_data() + input0->data_size() * i, data_size() - input0->data_size() * i, other->data(),
               other->data_size());
    }
  }
  return SUCCESS;
}

LogStream &operator<<(LogStream &stream, DataType data_type) {
  const std::map<DataType, std::string> type_name_map{
    {kMSI_Unknown, "kMSI_Unknown"}, {kMSI_Bool, "kMSI_Bool"},       {kMSI_Int8, "kMSI_Int8"},
    {kMSI_Uint8, "kMSI_Uint8"},     {kMSI_Int16, "kMSI_Int16"},     {kMSI_Uint16, "kMSI_Uint16"},
    {kMSI_Int32, "kMSI_Int32"},     {kMSI_Uint32, "kMSI_Uint32"},   {kMSI_Int64, "kMSI_Int64"},
    {kMSI_Uint64, "kMSI_Uint64"},   {kMSI_Float16, "kMSI_Float16"}, {kMSI_Float32, "kMSI_Float32"},
    {kMSI_Float64, "kMSI_Float64"}, {kMSI_Bytes, "kMSI_Bytes"},     {kMSI_String, "kMSI_String"},
  };
  auto it = type_name_map.find(data_type);
  if (it != type_name_map.end()) {
    stream << it->second;
  } else {
    stream << "kMSI_Unknown";
  }
  return stream;
}

}  // namespace mindspore::serving
