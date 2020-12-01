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

#include "common/buffer_tensor.h"
namespace mindspore::serving {

BufferTensor::BufferTensor(DataType type, const std::vector<int64_t> shape, uint8_t *data, size_t data_len,
                           bool data_readonly) {
  set_shape(shape);
  set_data_type(type);
  data_ = data;
  data_len_ = data_len;
  data_readonly_ = data_readonly;
}

std::vector<int64_t> BufferTensor::shape() const { return shape_; }

void BufferTensor::set_shape(const std::vector<int64_t> &shape) { shape_ = shape; }

DataType BufferTensor::data_type() const { return type_; }

void BufferTensor::set_data_type(DataType type) { type_ = type; }

const uint8_t *BufferTensor::data() const { return data_; }

size_t BufferTensor::data_size() const { return data_len_; }

uint8_t *BufferTensor::mutable_data() {
  if (data_readonly_) {
    MSI_LOG_EXCEPTION << "Buffer tensor is create readonly";
  }
  return data_;
}

size_t BufferTensor::bytes_data_size() const {
  if (!is_bytes_val_data()) {
    return 0;
  }
  return 1;
}

void BufferTensor::get_bytes_data(size_t index, const uint8_t **data, size_t *bytes_len) const {
  MSI_EXCEPTION_IF_NULL(data);
  MSI_EXCEPTION_IF_NULL(bytes_len);
  if (!is_bytes_val_data()) {
    MSI_LOG_EXCEPTION << "Buffer tensor data type is not kMSI_Bytes or kMSI_String, cannot get bytes data";
  }
  *data = data_;
  *bytes_len = data_len_;
}

}  // namespace mindspore::serving
