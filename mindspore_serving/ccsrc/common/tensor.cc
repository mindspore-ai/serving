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
#include "common/tensor.h"
#include <securec.h>
#include <functional>
#include <utility>
#include "common/log.h"

namespace mindspore::serving {

Tensor::Tensor() = default;

Tensor::Tensor(DataType type, std::vector<int64_t> shape, const void *data, size_t data_len) {
  set_data_type(type);
  set_shape(shape);
  set_data(data, data_len);
}

const uint8_t *Tensor::data() const {
  if (data_size() == 0) {
    return nullptr;
  }
  return data_.data();
}

size_t Tensor::data_size() const { return data_.size(); }

bool Tensor::resize_data(size_t data_len) {
  data_.resize(data_len);
  return true;
}

uint8_t *Tensor::mutable_data() {
  if (data_size() == 0) {
    return nullptr;
  }
  return data_.data();
}

// For kMSI_String and kMSI_Bytes
void Tensor::clear_bytes_data() { bytes_.clear(); }

void Tensor::add_bytes_data(const uint8_t *data, size_t bytes_len) {
  std::vector<uint8_t> bytes(bytes_len);
  memcpy_s(bytes.data(), bytes.size(), data, bytes_len);
  bytes_.push_back(std::move(bytes));
}

size_t Tensor::bytes_data_size() const { return bytes_.size(); }

void Tensor::get_bytes_data(size_t index, const uint8_t **data, size_t *bytes_len) const {
  MSI_EXCEPTION_IF_NULL(data);
  MSI_EXCEPTION_IF_NULL(bytes_len);
  *bytes_len = bytes_[index].size();
  if (*bytes_len == 0) {
    *data = nullptr;
  } else {
    *data = bytes_[index].data();
  }
}

}  // namespace mindspore::serving
