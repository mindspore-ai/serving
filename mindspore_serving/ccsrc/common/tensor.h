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

#ifndef MINDSPORE_SERVING_TENSOR_H
#define MINDSPORE_SERVING_TENSOR_H

#include <vector>
#include "common/tensor_base.h"

namespace mindspore::serving {
class MS_API Tensor : public TensorBase {
 public:
  Tensor();
  Tensor(DataType type, const std::vector<int64_t> &shape, const void *data, size_t data_len);
  ~Tensor() = default;

  void set_data_type(DataType type) override { type_ = type; }
  DataType data_type() const override { return type_; }

  void set_shape(const std::vector<int64_t> &shape) override { shape_ = shape; }
  std::vector<int64_t> shape() const override { return shape_; }

  const uint8_t *data() const override;
  size_t data_size() const override;

  bool resize_data(size_t data_len) override;
  uint8_t *mutable_data() override;

  // For kMSI_String and kMSI_Bytes
  void clear_bytes_data() override;
  void add_bytes_data(const uint8_t *data, size_t bytes_len) override;
  size_t bytes_data_size() const override;
  void get_bytes_data(size_t index, const uint8_t **data, size_t *bytes_len) const override;

 private:
  DataType type_ = kMSI_Unknown;
  std::vector<int64_t> shape_;
  std::vector<uint8_t> data_;
  // For kMSI_String and kMSI_Bytes
  std::vector<std::vector<uint8_t>> bytes_;
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_TENSOR_H
