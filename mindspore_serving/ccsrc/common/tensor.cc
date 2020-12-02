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

VectorTensorWrapReply::VectorTensorWrapReply(std::vector<Tensor> *tensor_list) : tensor_list_(tensor_list) {}

VectorTensorWrapReply::~VectorTensorWrapReply() = default;

size_t VectorTensorWrapReply::size() const {
  MSI_EXCEPTION_IF_NULL(tensor_list_);
  return tensor_list_->size();
}

void VectorTensorWrapReply::clear() {
  MSI_EXCEPTION_IF_NULL(tensor_list_);
  tensor_list_->clear();
}

TensorBase *VectorTensorWrapReply::operator[](size_t index) {
  MSI_EXCEPTION_IF_NULL(tensor_list_);
  if (index >= tensor_list_->size()) {
    MSI_LOG_EXCEPTION << "visit invalid index " << index << " total size " << tensor_list_->size();
  }
  return &((*tensor_list_)[index]);
}

const TensorBase *VectorTensorWrapReply::operator[](size_t index) const {
  MSI_EXCEPTION_IF_NULL(tensor_list_);
  if (index >= tensor_list_->size()) {
    MSI_LOG_EXCEPTION << "visit invalid index " << index << " total size " << tensor_list_->size();
  }
  return &((*tensor_list_)[index]);
}

TensorBase *VectorTensorWrapReply::add() {
  MSI_EXCEPTION_IF_NULL(tensor_list_);
  tensor_list_->push_back(Tensor());
  return &(tensor_list_->back());
}

const TensorBase *VectorTensorWrapRequest::operator[](size_t index) const {
  if (index >= tensor_list_.size()) {
    MSI_LOG_EXCEPTION << "visit invalid index " << index << " total size " << tensor_list_.size();
  }
  return &(tensor_list_[index]);
}

VectorTensorPtrWrapReply::VectorTensorPtrWrapReply(std::vector<TensorBasePtr> *tensor_list,
                                                   std::function<TensorBasePtr()> create_fun)
    : tensor_list_(tensor_list), tensor_create_fun_(create_fun) {}

VectorTensorPtrWrapReply::~VectorTensorPtrWrapReply() = default;

size_t VectorTensorPtrWrapReply::size() const {
  MSI_EXCEPTION_IF_NULL(tensor_list_);
  return tensor_list_->size();
}

void VectorTensorPtrWrapReply::clear() {
  MSI_EXCEPTION_IF_NULL(tensor_list_);
  tensor_list_->clear();
}

TensorBase *VectorTensorPtrWrapReply::operator[](size_t index) {
  MSI_EXCEPTION_IF_NULL(tensor_list_);
  if (index >= tensor_list_->size()) {
    MSI_LOG_EXCEPTION << "visit invalid index " << index << " total size " << tensor_list_->size();
  }
  return (*tensor_list_)[index].get();
}

const TensorBase *VectorTensorPtrWrapReply::operator[](size_t index) const {
  MSI_EXCEPTION_IF_NULL(tensor_list_);
  if (index >= tensor_list_->size()) {
    MSI_LOG_EXCEPTION << "visit invalid index " << index << " total size " << tensor_list_->size();
  }
  return (*tensor_list_)[index].get();
}

TensorBase *VectorTensorPtrWrapReply::add() {
  MSI_EXCEPTION_IF_NULL(tensor_list_);
  MSI_EXCEPTION_IF_NULL(tensor_create_fun_);
  auto tensor = tensor_create_fun_();
  if (tensor == nullptr) {
    MSI_LOG_EXCEPTION << "create tensor failed";
  }
  tensor_list_->push_back(tensor);
  return tensor.get();
}

const TensorBase *VectorTensorPtrWrapRequest::operator[](size_t index) const {
  if (index >= tensor_list_.size()) {
    MSI_LOG_EXCEPTION << "visit invalid index " << index << " total size " << tensor_list_.size();
  }
  return tensor_list_[index].get();
}

}  // namespace mindspore::serving
