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

#ifndef MINDSPORE_SERVING_TENSOR_BASE_H
#define MINDSPORE_SERVING_TENSOR_BASE_H

#include <utility>
#include <vector>
#include <memory>
#include <numeric>
#include <map>
#include <functional>
#include <cstring>
#include "common/log.h"
#include "common/status.h"

namespace mindspore {
namespace serving {

enum DataType {
  kMSI_Unknown = 0,
  kMSI_Bool = 1,
  kMSI_Int8 = 2,
  kMSI_Int16 = 3,
  kMSI_Int32 = 4,
  kMSI_Int64 = 5,
  kMSI_Uint8 = 6,
  kMSI_Uint16 = 7,
  kMSI_Uint32 = 8,
  kMSI_Uint64 = 9,
  kMSI_Float16 = 10,
  kMSI_Float32 = 11,
  kMSI_Float64 = 12,
  kMSI_String = 13,  // for model STRING input
  kMSI_Bytes = 14,   // for image etc.
};

class TensorBase;
using TensorBasePtr = std::shared_ptr<TensorBase>;

class MS_API TensorBase : public std::enable_shared_from_this<TensorBase> {
 public:
  TensorBase() = default;
  virtual ~TensorBase() = default;

  // For all data type
  virtual std::vector<int64_t> shape() const = 0;
  virtual void set_shape(const std::vector<int64_t> &shape) = 0;
  virtual DataType data_type() const = 0;
  virtual void set_data_type(DataType type) = 0;

  // All the following interfaces are not for  kMSI_String and kMSI_Bytes
  virtual const uint8_t *data() const = 0;
  virtual size_t data_size() const = 0;
  virtual bool resize_data(size_t data_len) = 0;
  virtual uint8_t *mutable_data() = 0;

  // Byte size of a single element.
  size_t itemsize() const;
  // Total number of elements.
  size_t element_cnt() const;
  // resize and copy data
  bool set_data(const void *data, size_t data_len);
  static size_t GetTypeSize(DataType type);

  // For kMSI_String and kMSI_Bytes
  virtual void clear_bytes_data() = 0;
  virtual void add_bytes_data(const uint8_t *data, size_t bytes_len) = 0;
  virtual size_t bytes_data_size() const = 0;
  virtual void get_bytes_data(size_t index, const uint8_t **data, size_t *bytes_len) const = 0;

  // TensorBase(const TensorBase& other) = delete;
  // TensorBase& operator=(const TensorBase& other) = delete;
  void assign(const TensorBase &other);
  bool is_bytes_val_data() const { return data_type() == kMSI_Bytes || data_type() == kMSI_String; }
};

class RequestBase {
 public:
  RequestBase() = default;
  virtual ~RequestBase() = default;
  virtual size_t size() const = 0;
  virtual const TensorBase *operator[](size_t index) const = 0;
};

class ReplyBase {
 public:
  ReplyBase() = default;
  virtual ~ReplyBase() = default;
  virtual size_t size() const = 0;
  virtual TensorBase *operator[](size_t index) = 0;
  virtual const TensorBase *operator[](size_t index) const = 0;
  virtual TensorBase *add() = 0;
  virtual void clear() = 0;
};

extern MS_API LogStream &operator<<(LogStream &stream, DataType data_type);

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_TENSOR_BASE_H
