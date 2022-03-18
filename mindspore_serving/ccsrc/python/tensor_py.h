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

#ifndef MINDSPORE_SERVING_SERVING_PY_H
#define MINDSPORE_SERVING_SERVING_PY_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <utility>
#include <vector>
#include "common/serving_common.h"
#include "common/instance.h"

namespace py = pybind11;

namespace mindspore::serving {
class NumpyTensor : public TensorBase {
 public:
  explicit NumpyTensor(py::buffer_info &&buffer) : buffer_(std::move(buffer)) {}
  ~NumpyTensor() noexcept {
    py::gil_scoped_acquire acquire;
    { buffer_ = py::buffer_info(); }
  }
  /// py::array object.
  py::array py_array() const {
    // Use dummy owner to avoid copy data.
    py::str dummyOwner;
    return py::array(py::dtype(buffer_), buffer_.shape, buffer_.strides, buffer_.ptr, dummyOwner);
  }

  void set_data_type(DataType) override {
    MSI_LOG_EXCEPTION << "NumpyTensor is readyonly, cannot invoke set_data_type";
  }
  DataType data_type() const override { return GetDataType(buffer_); }

  void set_shape(const std::vector<int64_t> &) override {
    MSI_LOG_EXCEPTION << "NumpyTensor is readyonly, cannot invoke set_shape";
  }
  std::vector<int64_t> shape() const override { return buffer_.shape; }

  const uint8_t *data() const override { return static_cast<const uint8_t *>(buffer_.ptr); }
  size_t data_size() const override {
    if (buffer_.size <= 0 || buffer_.itemsize <= 0) {
      return 0;
    }
    return static_cast<size_t>(buffer_.size * buffer_.itemsize);
  }

  bool resize_data(size_t) override { MSI_LOG_EXCEPTION << "NumpyTensor is readonly, cannot invoke resize_data"; }
  uint8_t *mutable_data() override { MSI_LOG_EXCEPTION << "NumpyTensor is readonly, cannot invoke mutable_data"; }

  void clear_bytes_data() override { MSI_LOG_EXCEPTION << "NumpyTensor is readyonly, cannot invoke clear_bytes_data"; }
  void add_bytes_data(const uint8_t *, size_t) override {
    MSI_LOG_EXCEPTION << "NumpyTensor is readyonly, cannot invoke add_bytes_data";
  }

  size_t bytes_data_size() const override { return 0; }
  void get_bytes_data(size_t, const uint8_t **, size_t *) const override {
    MSI_LOG_EXCEPTION << "NumpyTensor is readyonly, cannot invoke get_bytes_data";
  }

  static DataType GetDataType(const py::buffer_info &buf);

 private:
  py::buffer_info buffer_;
};

class PyTensor {
 public:
  // For all type, but for BYTES type, there can only be one item in bytes_val.
  // If the tensor data is destroyed when the numpy array is return to python env, the tensor data need to be copied
  static py::object AsPythonData(const TensorBasePtr &tensor, bool copy = false);
  static TensorBasePtr MakeTensor(const py::array &input);
  static TensorBasePtr MakeTensorNoCopy(const py::array &input);

  static py::tuple AsNumpyTuple(const InstanceData &instance);
  static InstanceData AsInstanceData(const py::tuple &tuple);
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_SERVING_PY_H
