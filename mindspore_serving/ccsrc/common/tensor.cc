#include "tensor.h"
#include <functional>
#include "log.h"
#include "securec.h"

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

void Tensor::get_bytes_data(size_t index, const uint8_t *&data, size_t &bytes_len) const {
  bytes_len = bytes_[index].size();
  if (bytes_len == 0) {
    data = nullptr;
  } else {
    data = bytes_[index].data();
  }
}

TensorBase *VectorTensorWrapReply::operator[](size_t index) {
  if (index >= tensor_list_.size()) {
    MSI_LOG_EXCEPTION << "visit invalid index " << index << " total size " << tensor_list_.size();
  }
  return &(tensor_list_[index]);
}

const TensorBase *VectorTensorWrapReply::operator[](size_t index) const {
  if (index >= tensor_list_.size()) {
    MSI_LOG_EXCEPTION << "visit invalid index " << index << " total size " << tensor_list_.size();
  }
  return &(tensor_list_[index]);
}

TensorBase *VectorTensorWrapReply::add() {
  tensor_list_.push_back(Tensor());
  return &(tensor_list_.back());
}

const TensorBase *VectorTensorWrapRequest::operator[](size_t index) const {
  if (index >= tensor_list_.size()) {
    MSI_LOG_EXCEPTION << "visit invalid index " << index << " total size " << tensor_list_.size();
  }
  return &(tensor_list_[index]);
}

TensorBase *VectorTensorPtrWrapReply::operator[](size_t index) {
  if (index >= tensor_list_.size()) {
    MSI_LOG_EXCEPTION << "visit invalid index " << index << " total size " << tensor_list_.size();
  }
  return tensor_list_[index].get();
}

const TensorBase *VectorTensorPtrWrapReply::operator[](size_t index) const {
  if (index >= tensor_list_.size()) {
    MSI_LOG_EXCEPTION << "visit invalid index " << index << " total size " << tensor_list_.size();
  }
  return tensor_list_[index].get();
}

TensorBase *VectorTensorPtrWrapReply::add() {
  auto tensor = tensor_create_fun_();
  if (tensor == nullptr) {
    MSI_LOG_EXCEPTION << "create tensor failed";
  }
  tensor_list_.push_back(tensor);
  return tensor.get();
}

const TensorBase *VectorTensorPtrWrapRequest::operator[](size_t index) const {
  if (index >= tensor_list_.size()) {
    MSI_LOG_EXCEPTION << "visit invalid index " << index << " total size " << tensor_list_.size();
  }
  return tensor_list_[index].get();
}

}  // namespace mindspore::serving
