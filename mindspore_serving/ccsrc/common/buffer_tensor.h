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

#ifndef MINDSPORE_BUFFER_TENSOR_H
#define MINDSPORE_BUFFER_TENSOR_H

#include <vector>
#include "common/serving_common.h"

namespace mindspore::serving {

class BufferTensor : public TensorBase {
 public:
  // the data's lifetime must longer than this object
  BufferTensor(DataType type, std::vector<int64_t> shape, uint8_t *data, size_t data_len, bool data_readonly);
  ~BufferTensor() = default;

  // For all data type
  std::vector<int64_t> shape() const override;
  void set_shape(const std::vector<int64_t> &shape) override;
  DataType data_type() const override;
  void set_data_type(DataType type) override;

  // All the following interfaces are not for kMSI_String and kMSI_Bytes
  const uint8_t *data() const override;
  size_t data_size() const override;
  bool resize_data(size_t data_len) override { MSI_LOG_EXCEPTION << "Buffer tensor cannot resize data"; }
  uint8_t *mutable_data() override;

  // For kMSI_String and kMSI_Bytes
  void clear_bytes_data() override { MSI_LOG_EXCEPTION << "Buffer tensor cannot clear bytes data"; }
  void add_bytes_data(const uint8_t *data, size_t bytes_len) override {
    MSI_LOG_EXCEPTION << "Buffer tensor cannot add bytes data";
  }

  size_t bytes_data_size() const override;
  void get_bytes_data(size_t index, const uint8_t **data, size_t *bytes_len) const override;

 private:
  uint8_t *data_ = nullptr;
  size_t data_len_ = 0;
  std::vector<int64_t> shape_;
  DataType type_;
  bool data_readonly_ = false;
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_BUFFER_TENSOR_H
