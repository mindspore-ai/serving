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

#include "python/tensor_py.h"
#include <pybind11/pytypes.h>
#include <string>
#include <vector>
#include <memory>
#include <set>
#include "mindspore_serving/ccsrc/common/tensor.h"

namespace mindspore::serving {
static std::vector<ssize_t> GetStrides(const std::vector<ssize_t> &shape, ssize_t item_size) {
  std::vector<ssize_t> strides;
  strides.reserve(shape.size());
  const auto ndim = shape.size();
  for (size_t i = 0; i < ndim; ++i) {
    auto stride = item_size;
    for (size_t j = i + 1; j < ndim; ++j) {
      stride *= shape[j];
    }
    strides.push_back(stride);
  }
  return strides;
}

DataType NumpyTensor::GetDataType(const py::buffer_info &buf) {
  std::set<char> fp_format = {'e', 'f', 'd'};
  std::set<char> int_format = {'b', 'h', 'i', 'l', 'q'};
  std::set<char> uint_format = {'B', 'H', 'I', 'L', 'Q'};
  if (buf.format.size() == 1) {
    char format = buf.format.front();
    if (fp_format.find(format) != fp_format.end()) {
      constexpr int size_of_fp16 = 2;
      constexpr int size_of_fp32 = 4;
      constexpr int size_of_fp64 = 8;
      switch (buf.itemsize) {
        case size_of_fp16:
          return kMSI_Float16;
        case size_of_fp32:
          return kMSI_Float32;
        case size_of_fp64:
          return kMSI_Float64;
      }
    } else if (int_format.find(format) != int_format.end()) {
      switch (buf.itemsize) {
        case sizeof(int8_t):
          return kMSI_Int8;
        case sizeof(int16_t):
          return kMSI_Int16;
        case sizeof(int32_t):
          return kMSI_Int32;
        case sizeof(int64_t):
          return kMSI_Int64;
      }
    } else if (uint_format.find(format) != uint_format.end()) {
      switch (buf.itemsize) {
        case sizeof(uint8_t):
          return kMSI_Uint8;
        case sizeof(uint16_t):
          return kMSI_Uint16;
        case sizeof(uint32_t):
          return kMSI_Uint32;
        case sizeof(uint64_t):
          return kMSI_Uint64;
      }
    } else if (format == '?') {
      return kMSI_Bool;
    }
  }
  MSI_LOG(WARNING) << "Unsupported DataType format " << buf.format << " item size " << buf.itemsize;
  return kMSI_Unknown;
}

static std::string GetPyTypeFormat(DataType data_type) {
  switch (data_type) {
    case kMSI_Float16:
      return "e";
    case kMSI_Float32:
      return py::format_descriptor<float>::format();
    case kMSI_Float64:
      return py::format_descriptor<double>::format();
    case kMSI_Uint8:
      return py::format_descriptor<uint8_t>::format();
    case kMSI_Uint16:
      return py::format_descriptor<uint16_t>::format();
    case kMSI_Uint32:
      return py::format_descriptor<uint32_t>::format();
    case kMSI_Uint64:
      return py::format_descriptor<uint64_t>::format();
    case kMSI_Int8:
      return py::format_descriptor<int8_t>::format();
    case kMSI_Int16:
      return py::format_descriptor<int16_t>::format();
    case kMSI_Int32:
      return py::format_descriptor<int32_t>::format();
    case kMSI_Int64:
      return py::format_descriptor<int64_t>::format();
    case kMSI_Bool:
      return py::format_descriptor<bool>::format();
    default:
      MSI_LOG(WARNING) << "Unsupported DataType " << data_type << ".";
      return "";
  }
}

static bool IsCContiguous(const py::array &input) {
  auto flags = static_cast<unsigned int>(input.flags());
  return (flags & pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_) != 0;
}

TensorBasePtr PyTensor::MakeTensor(const py::array &input) {
  // Get input buffer info.
  py::buffer_info buf = input.request();
  // Check data types.
  auto buf_type = NumpyTensor::GetDataType(buf);
  if (buf_type == kMSI_Unknown) {
    MSI_LOG(EXCEPTION) << "Unsupported tensor type!";
  }
  // Convert input array to C contiguous if need.
  std::unique_ptr<char[]> tmp_buf;
  if (!IsCContiguous(input)) {
    Py_buffer pybuf;
    if (PyObject_GetBuffer(input.ptr(), &pybuf, PyBUF_ANY_CONTIGUOUS) || pybuf.len < 0) {
      MSI_LOG(EXCEPTION) << "Failed to get buffer from the input!";
    }
    tmp_buf = std::make_unique<char[]>(static_cast<size_t>(pybuf.len));
    if (PyBuffer_ToContiguous(tmp_buf.get(), &pybuf, pybuf.len, 'C')) {
      MSI_LOG(EXCEPTION) << "Can't copy numpy.ndarray to a contiguous buffer.";
    }
    PyBuffer_Release(&pybuf);
    buf.ptr = tmp_buf.get();
  }
  // Get tensor shape.
  std::vector<int64_t> shape(buf.shape.begin(), buf.shape.end());
  return std::make_shared<Tensor>(buf_type, shape, buf.ptr, buf.size * buf.itemsize);
}

/// Creates a Tensor from a numpy array without copy
TensorBasePtr PyTensor::MakeTensorNoCopy(const py::array &input) {
  // Check format.
  if (!IsCContiguous(input)) {
    MSI_LOG(EXCEPTION) << "Array should be C contiguous.";
  }
  // Get input buffer info.
  py::buffer_info buf = input.request();
  // Get tensor dtype and check it.
  auto dtype = NumpyTensor::GetDataType(buf);
  if (dtype == kMSI_Unknown) {
    MSI_LOG(EXCEPTION) << "Unsupported data type!";
  }
  // Make a tensor with shared data with numpy array.
  auto tensor_data = std::make_shared<NumpyTensor>(std::move(buf));
  return tensor_data;
}

py::object PyTensor::AsPythonData(const TensorBasePtr &tensor, bool copy) {
  auto data_numpy = std::dynamic_pointer_cast<NumpyTensor>(tensor);
  if (data_numpy) {
    return data_numpy->py_array();
  }
  if (tensor->is_bytes_val_data()) {
    if (tensor->bytes_data_size() != 1) {
      return py::array();
    }
    const uint8_t *data = nullptr;
    size_t bytes_len = 0;
    tensor->get_bytes_data(0, &data, &bytes_len);
    if (tensor->data_type() == kMSI_String) {
      return py::str(reinterpret_cast<const char *>(data), bytes_len);
    }
    std::vector<ssize_t> shape{static_cast<ssize_t>(bytes_len)};
    std::vector<ssize_t> strides = GetStrides(shape, static_cast<ssize_t>(sizeof(uint8_t)));
    py::buffer_info info(reinterpret_cast<void *>(const_cast<uint8_t *>(data)), sizeof(uint8_t),
                         py::format_descriptor<uint8_t>::format(), 1, shape, strides);
    if (!copy) {
      py::object self = py::cast(tensor);
      return py::array(py::dtype(info), info.shape, info.strides, info.ptr, self);
    } else {
      return py::array(py::dtype(info), info.shape, info.strides, info.ptr);
    }
  } else {
    const auto &tensor_shape = tensor->shape();
    std::vector<ssize_t> shape(tensor_shape.begin(), tensor_shape.end());
    std::vector<ssize_t> strides = GetStrides(shape, static_cast<ssize_t>(tensor->itemsize()));
    py::buffer_info info(reinterpret_cast<void *>(const_cast<uint8_t *>(tensor->data())),
                         static_cast<ssize_t>(tensor->itemsize()), GetPyTypeFormat(tensor->data_type()),
                         static_cast<ssize_t>(tensor_shape.size()), shape, strides);

    if (!copy) {
      py::object self = py::cast(tensor);
      return py::array(py::dtype(info), info.shape, info.strides, info.ptr, self);
    } else {
      return py::array(py::dtype(info), info.shape, info.strides, info.ptr);
    }
  }
}

py::tuple PyTensor::AsNumpyTuple(const InstanceData &instance_data) {
  py::tuple numpy_inputs_tuple(instance_data.size());
  for (size_t i = 0; i < instance_data.size(); i++) {  // inputs
    numpy_inputs_tuple[i] = PyTensor::AsPythonData(instance_data[i], false);
  }
  return numpy_inputs_tuple;
}

InstanceData PyTensor::AsInstanceData(const py::tuple &tuple) {
  InstanceData instance_data;
  for (auto &item : tuple) {
    TensorBasePtr tensor = nullptr;
    if (py::isinstance<py::bytes>(item)) {  // bytes can be seen as str, so check bytes first
      tensor = std::make_shared<Tensor>();
      tensor->set_data_type(serving::kMSI_Bytes);
      auto val = std::string(item.cast<py::bytes>());
      tensor->add_bytes_data(reinterpret_cast<const uint8_t *>(val.data()), val.length());
    } else if (py::isinstance<py::str>(item)) {
      tensor = std::make_shared<Tensor>();
      tensor->set_data_type(serving::kMSI_String);
      auto val = item.cast<std::string>();
      tensor->add_bytes_data(reinterpret_cast<const uint8_t *>(val.data()), val.length());
    } else if (py::isinstance<py::bool_>(item)) {
      auto val = item.cast<bool>();
      tensor = std::make_shared<Tensor>(serving::kMSI_Bool, std::vector<int64_t>(), &val, sizeof(val));
    } else if (py::isinstance<py::int_>(item)) {
      auto val = item.cast<int64_t>();
      tensor = std::make_shared<Tensor>(serving::kMSI_Int64, std::vector<int64_t>(), &val, sizeof(val));
    } else if (py::isinstance<py::float_>(item)) {
      auto val = item.cast<double>();
      tensor = std::make_shared<Tensor>(serving::kMSI_Float64, std::vector<int64_t>(), &val, sizeof(val));
    } else {
      try {
        tensor = PyTensor::MakeTensorNoCopy(py::cast<py::array>(item));
      } catch (const std::runtime_error &error) {
        MSI_LOG_EXCEPTION << "Get illegal result data with type " << py::str(item.get_type()).cast<std::string>();
      }
    }
    instance_data.push_back(tensor);
  }
  return instance_data;
}
}  // namespace mindspore::serving
