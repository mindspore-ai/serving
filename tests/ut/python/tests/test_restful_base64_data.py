# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""test Serving RESTful, with master, worker and client"""

import base64

import numpy as np

from common import ServingTestBase, serving_test
from common import servable_config_import, servable_config_declare_servable
from common_restful import compare_float_value, check_number_result, post_restful
from common_restful import start_str_restful_server, start_bytes_restful_server, start_bool_int_float_restful_server
from mindspore_serving import master
from mindspore_serving import worker


def b64_decode_to_str(a):
    return bytes.decode(base64.b64decode(a["b64"]))


def common_test_restful_base64_str_scalar_input_output_success(shape):
    base = start_str_restful_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "DEF", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        if shape is None:
            instance["text1"] = {"b64": base64.b64encode(str.encode(str_a[i])).decode(), "type": "str"}
            instance["text2"] = {"b64": base64.b64encode(str.encode(str_b[i])).decode(), "type": "str"}
        else:
            instance["text1"] = {"b64": base64.b64encode(str.encode(str_a[i])).decode(), "type": "str",
                                 'shape': shape}
            instance["text2"] = {"b64": base64.b64encode(str.encode(str_b[i])).decode(), "type": "str",
                                 'shape': shape}

    result = post_restful("localhost", 5500, base.servable_name, "str_concat", instances)
    result = result["instances"]
    assert result[0]["text"] == str_a[0] + str_b[0]
    assert result[1]["text"] == str_a[1] + str_b[1]
    assert result[2]["text"] == str_a[2] + str_b[2]


@serving_test
def test_restful_base64_str_scalar_input_output_success():
    common_test_restful_base64_str_scalar_input_output_success(shape=None)


@serving_test
def test_restful_base64_str_scalar_shape1_input_output_success():
    common_test_restful_base64_str_scalar_input_output_success(shape=[1])


@serving_test
def test_restful_base64_str_scalar_shape_empty_input_output_success():
    common_test_restful_base64_str_scalar_input_output_success(shape=[])


@serving_test
def test_restful_base64_empty_str_input_output_success():
    base = start_str_restful_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = {"b64": base64.b64encode(str.encode(str_a[i])).decode(), "type": "str"}
        instance["text2"] = {"b64": base64.b64encode(str.encode(str_b[i])).decode(), "type": "str"}

    result = post_restful("localhost", 5500, base.servable_name, "str_empty", instances)
    result = result["instances"]
    assert result[0]["text"] == ""
    assert result[1]["text"] == "456"
    assert result[2]["text"] == ""


@serving_test
def test_restful_base64_str_scalar_invalid_shape0_input_failed():
    base = start_str_restful_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "DEF", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = {"b64": base64.b64encode(str.encode(str_a[i])).decode(), "type": "str", "shape": [0]}
        instance["text2"] = {"b64": base64.b64encode(str.encode(str_b[i])).decode(), "type": "str", "shape": [0]}

    result = post_restful("localhost", 5500, base.servable_name, "str_concat", instances)
    assert "only support scalar when data type is string or bytes, please check 'type' or 'shape'" \
           in str(result["error_msg"])


@serving_test
def test_restful_base64_str_scalar_invalid_shape_input_failed():
    base = start_str_restful_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "DEF", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = {"b64": base64.b64encode(str.encode(str_a[i])).decode(), "type": "str", 'shape': [2]}
        instance["text2"] = {"b64": base64.b64encode(str.encode(str_b[i])).decode(), "type": "str", 'shape': [2]}

    result = post_restful("localhost", 5500, base.servable_name, "str_concat", instances)
    assert "json object, only support scalar when data type is string or bytes, please check 'type' or 'shape'" \
           in str(result["error_msg"])


@serving_test
def test_restful_base64_str_1d_array_failed():
    base = start_str_restful_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "DEF", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = [{"b64": base64.b64encode(str.encode(str_a[i])).decode(), "type": "str"},
                             {"b64": base64.b64encode(str.encode(str_a[i])).decode(), "type": "str"}]
        instance["text2"] = [{"b64": base64.b64encode(str.encode(str_b[i])).decode(), "type": "str"},
                             {"b64": base64.b64encode(str.encode(str_b[i])).decode(), "type": "str"}]

    result = post_restful("localhost", 5500, base.servable_name, "str_concat", instances)
    assert "json array, string or bytes type only support one item" in str(result["error_msg"])


def common_test_restful_bytes_input_output_success(shape):
    base = start_bytes_restful_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "DEF", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        if shape is not None:
            instance["text1"] = {"b64": base64.b64encode(str.encode(str_a[i])).decode(), "shape": shape}
            instance["text2"] = {"b64": base64.b64encode(str.encode(str_b[i])).decode(), "shape": shape}
        else:
            instance["text1"] = {"b64": base64.b64encode(str.encode(str_a[i])).decode()}
            instance["text2"] = {"b64": base64.b64encode(str.encode(str_b[i])).decode()}

    result = post_restful("localhost", 5500, base.servable_name, "bytes_concat", instances)
    result = result["instances"]
    assert b64_decode_to_str(result[0]["text"]) == str_a[0] + str_b[0]
    assert b64_decode_to_str(result[1]["text"]) == str_a[1] + str_b[1]
    assert b64_decode_to_str(result[2]["text"]) == str_a[2] + str_b[2]


@serving_test
def test_restful_bytes_input_output_success():
    common_test_restful_bytes_input_output_success(None)


@serving_test
def test_restful_bytes_empty_shape_success():
    common_test_restful_bytes_input_output_success([])


@serving_test
def test_restful_bytes_shape1_success():
    common_test_restful_bytes_input_output_success([1])


@serving_test
def test_restful_empty_bytes_input_output_success():
    base = start_bytes_restful_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = {"b64": base64.b64encode(str.encode(str_a[i])).decode()}
        instance["text2"] = {"b64": base64.b64encode(str.encode(str_b[i])).decode()}

    result = post_restful("localhost", 5500, base.servable_name, "bytes_empty", instances)
    result = result["instances"]
    assert b64_decode_to_str(result[0]["text"]) == ""
    assert b64_decode_to_str(result[1]["text"]) == "456"
    assert b64_decode_to_str(result[2]["text"]) == ""


@serving_test
def test_restful_bytes_1d_array_failed():
    base = start_bytes_restful_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "DEF", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = [{"b64": base64.b64encode(str.encode(str_a[i])).decode()},
                             {"b64": base64.b64encode(str.encode(str_a[i])).decode()}]
        instance["text2"] = [{"b64": base64.b64encode(str.encode(str_b[i])).decode()},
                             {"b64": base64.b64encode(str.encode(str_b[i])).decode()}]

    result = post_restful("localhost", 5500, base.servable_name, "bytes_concat", instances)
    assert "json array, string or bytes type only support one item" in str(result["error_msg"])


@serving_test
def test_restful_bytes_invalid_shape_input_failed():
    base = start_bytes_restful_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "DEF", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = {"b64": base64.b64encode(str.encode(str_a[i])).decode(), 'shape': [0]}
        instance["text2"] = {"b64": base64.b64encode(str.encode(str_b[i])).decode(), 'shape': [0]}

    result = post_restful("localhost", 5500, base.servable_name, "bytes_concat", instances)
    assert "only support scalar when data type is string or bytes, please check 'type' or 'shape'" \
           in result["error_msg"]


@serving_test
def test_restful_base64_bool_scalar_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = np.int8(i % 2 == 0)
        instance["bool_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "bool"}

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    result = result["instances"]
    assert not result[0]["value"]
    assert result[1]["value"]
    assert not result[2]["value"]


@serving_test
def test_restful_base64_bool_1d_array_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = [(i % 2 == 0)] * (i + 1)
        val = np.array(val)
        instance["bool_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "bool", "shape": [i + 1]}

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    result = result["instances"]
    assert result[0]["value"] == [False]
    assert result[1]["value"] == [True, True]
    assert result[2]["value"] == [False, False, False]


@serving_test
def test_restful_base64_bool_2d_array_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = (i % 2 == 0)
        val = [[val] * (i + 1)] * (i + 1)
        val = np.array(val)
        instance["bool_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "bool",
                                "shape": [i + 1, i + 1]}

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    result = result["instances"]
    assert result[0]["value"] == [[False]]
    assert result[1]["value"] == [[True, True], [True, True]]
    assert result[2]["value"] == [[False, False, False], [False, False, False], [False, False, False]]


@serving_test
def test_restful_base64_int_scalar_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = np.int32(i * 2)
        instance["int_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "int32"}

    result = post_restful("localhost", 5500, base.servable_name, "int_plus_1", instances)
    result = result["instances"]
    assert result[0]["value"] == 1
    assert result[1]["value"] == 3
    assert result[2]["value"] == 5


@serving_test
def test_restful_base64_int_1d_empty_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        if i % 2 == 0:
            val = []
        else:
            val = [i * 2] * (i + 1)
        val = np.array(val).astype(np.int32)
        instance["int_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "int32", "shape": val.shape}

    result = post_restful("localhost", 5500, base.servable_name, "int_plus_1", instances)
    result = result["instances"]
    assert result[0]["value"] == []
    assert result[1]["value"] == [3, 3]
    assert result[2]["value"] == []


@serving_test
def test_restful_base64_int_2d_empty_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        if i % 2 == 0:
            val = [[]]
        else:
            val = [i * 2] * (i + 1)
        val = np.array(val).astype(np.int32)
        instance["int_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "int32", "shape": val.shape}

    result = post_restful("localhost", 5500, base.servable_name, "int_plus_1", instances)
    result = result["instances"]
    assert result[0]["value"] == [[]]
    assert result[1]["value"] == [3, 3]
    assert result[2]["value"] == [[]]


@serving_test
def test_restful_base64_int_2d_empty_invalid_shape_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for _, instance in enumerate(instances):
        val = [[]]
        val = np.array(val).astype(np.int32)
        instance["int_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "int32", "shape": [1, 2, 0, 1]}

    result = post_restful("localhost", 5500, base.servable_name, "int_plus_1", instances)
    assert "json object, key is 'shape', invalid shape value" in result["error_msg"]


@serving_test
def test_restful_base64_int_1d_array_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = i * 2
        val = [val] * (i + 1)
        val = np.array(val).astype(np.int32)
        instance["int_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "int32", "shape": val.shape}

    result = post_restful("localhost", 5500, base.servable_name, "int_plus_1", instances)
    result = result["instances"]
    assert result[0]["value"] == [1]
    assert result[1]["value"] == [3, 3]
    assert result[2]["value"] == [5, 5, 5]


def common_test_restful_base64_int_type_2d_array_input_output_success(dtype):
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    dtype_str_map = {np.int8: "int8", np.int16: "int16", np.int32: "int32", np.int64: "int64"}
    assert dtype in dtype_str_map
    for i, instance in enumerate(instances):
        val = (i + 1) * 2 * (-1 if i % 2 == 0 else 1)  # -2, 4, -6
        val = [[val] * (i + 1)] * (i + 1)
        val = np.array(val).astype(dtype)
        instance["int_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': dtype_str_map[dtype],
                               "shape": val.shape}

    result = post_restful("localhost", 5500, base.servable_name, "int_plus_1", instances)
    result = result["instances"]
    assert result[0]["value"] == [[-1]]
    assert result[1]["value"] == [[5, 5], [5, 5]]
    assert result[2]["value"] == [[-5, -5, -5], [-5, -5, -5], [-5, -5, -5]]


@serving_test
def test_restful_base64_int8_2d_array_input_output_success():
    common_test_restful_base64_int_type_2d_array_input_output_success(np.int8)


@serving_test
def test_restful_base64_int16_2d_array_input_output_success():
    common_test_restful_base64_int_type_2d_array_input_output_success(np.int16)


@serving_test
def test_restful_base64_int32_2d_array_input_output_success():
    common_test_restful_base64_int_type_2d_array_input_output_success(np.int32)


@serving_test
def test_restful_base64_int64_2d_array_input_output_success():
    common_test_restful_base64_int_type_2d_array_input_output_success(np.int64)


def common_test_restful_base64_uint_type_2d_array_input_output_success(dtype):
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    dtype_str_map = {np.uint8: "uint8", np.uint16: "uint16", np.uint32: "uint32", np.uint64: "uint64"}
    assert dtype in dtype_str_map
    for i, instance in enumerate(instances):
        val = i * 2
        val = [[val] * (i + 1)] * (i + 1)
        val = np.array(val).astype(dtype)
        instance["int_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': dtype_str_map[dtype],
                               "shape": val.shape}

    result = post_restful("localhost", 5500, base.servable_name, "int_plus_1", instances)
    result = result["instances"]
    assert result[0]["value"] == [[1]]
    assert result[1]["value"] == [[3, 3], [3, 3]]
    assert result[2]["value"] == [[5, 5, 5], [5, 5, 5], [5, 5, 5]]


@serving_test
def test_restful_base64_uint8_2d_array_input_output_success():
    common_test_restful_base64_uint_type_2d_array_input_output_success(np.uint8)


@serving_test
def test_restful_base64_uint16_2d_array_input_output_success():
    common_test_restful_base64_uint_type_2d_array_input_output_success(np.uint16)


@serving_test
def test_restful_base64_uint32_2d_array_input_output_success():
    common_test_restful_base64_uint_type_2d_array_input_output_success(np.uint32)


@serving_test
def test_restful_base64_uint64_2d_array_input_output_success():
    common_test_restful_base64_uint_type_2d_array_input_output_success(np.uint64)


@serving_test
def test_restful_base64_float_scalar_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = np.float32(i * 2.2)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp32"}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    result = result["instances"]
    assert result[0]["value"] == 1.0
    assert abs(result[1]["value"] - (2.2 + 1)) < 0.001
    assert abs(result[2]["value"] - (4.4 + 1)) < 0.001


@serving_test
def test_restful_base64_float_1d_array_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    y_data_list = []
    for i, instance in enumerate(instances):
        val = [i * 2.2 * (-1 if i % 2 == 0 else 1)] * (i + 1)  # [0], [2.2, 2.2], [-4.4, -4.4, -4.4]
        val = np.array(val).astype(np.float32)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp32", 'shape': [i + 1]}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    check_number_result(result, y_data_list, "value")


def common_test_restful_base64_float_type_2d_array_input_output_success(dtype, dtype_str=None):
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]

    dtype_str_map = {np.float16: "fp16", np.float32: "fp32", np.float64: "fp64"}
    assert dtype in dtype_str_map
    if dtype_str is None:
        dtype_str = dtype_str_map[dtype]

    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 1)] * (i + 1)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': dtype_str,
                                 'shape': [i + 1, i + 1]}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    check_number_result(result, y_data_list, "value")


@serving_test
def test_restful_base64_float16_2d_array_input_output_success():
    common_test_restful_base64_float_type_2d_array_input_output_success(np.float16)


@serving_test
def test_restful_base64_float32_2d_array_input_output_success():
    common_test_restful_base64_float_type_2d_array_input_output_success(np.float32)


@serving_test
def test_restful_base64_float64_2d_array_input_output_success():
    common_test_restful_base64_float_type_2d_array_input_output_success(np.float64)


@serving_test
def test_restful_base64_float16_2_2d_array_input_output_success():
    common_test_restful_base64_float_type_2d_array_input_output_success(np.float16, "float16")


@serving_test
def test_restful_base64_float32_2_2d_array_input_output_success():
    common_test_restful_base64_float_type_2d_array_input_output_success(np.float32, "float32")


@serving_test
def test_restful_base64_float64_2_2d_array_input_output_success():
    common_test_restful_base64_float_type_2d_array_input_output_success(np.float64, "float64")


@serving_test
def test_restful_base64_mix_all_type_success():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def preprocess(float_val):
    return np.ones([2,2], np.float32), np.ones([2,2], np.float32)  
    
def postprocess(bool_val, int_val, float_val, str_val, bytes_val):
    return ~bool_val, int_val+1, float_val+1, str_val+"123", str.encode(bytes.decode(bytes_val.tobytes()) + "456") 

@register.register_method(output_names=['bool_val', 'int_val', 'float_val', 'str_val', 'bytes_val'])
def mix_all_type(bool_val, int_val, float_val, str_val, bytes_val):
    x1, x2 = register.call_preprocess(preprocess, float_val)
    y = register.call_servable(x1, x2)    
    bool_val, int_val, float_val, str_val, bytes_val = \
        register.call_postprocess(postprocess, bool_val, int_val, float_val, str_val, bytes_val)
    return bool_val, int_val, float_val, str_val, bytes_val
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        float_val = np.array([2.2, 3.3]).astype(np.float32)
        instance["float_val"] = {"b64": base64.b64encode(float_val.tobytes()).decode(), 'type': "fp32", 'shape': [2]}

        int_val = np.array([2, 3]).astype(np.int32)
        instance["int_val"] = {"b64": base64.b64encode(int_val.tobytes()).decode(), 'type': "int32", 'shape': [2]}

        bool_val = np.array([True, False])
        instance["bool_val"] = {"b64": base64.b64encode(bool_val.tobytes()).decode(), 'type': "bool", 'shape': [2]}

        str_val = "ABC"
        instance["str_val"] = {"b64": base64.b64encode(str.encode(str_val)).decode(), 'type': "str", 'shape': []}

        bytes_val = "DEF"
        instance["bytes_val"] = {"b64": base64.b64encode(str.encode(bytes_val)).decode(), 'type': "bytes", 'shape': []}

    result = post_restful("localhost", 5500, base.servable_name, "mix_all_type", instances)
    result = result["instances"]

    for i in range(3):
        compare_float_value(result[i]["float_val"], [3.2, 4.3])
        assert result[i]["int_val"] == [3, 4]
        assert result[i]["bool_val"] == [False, True]
        assert result[i]["str_val"] == "ABC123"
        assert b64_decode_to_str(result[i]["bytes_val"]) == "DEF456"


@serving_test
def test_restful_base64_without_b64_key_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]

    dtype = np.float16
    dtype_str_map = {np.float16: "fp16", np.float32: "fp32", np.float64: "fp64"}
    assert dtype in dtype_str_map

    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 1)] * (i + 1)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {'type': dtype_str_map[dtype], 'shape': [i + 1, i + 1]}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    assert "'b64' should be specified only one time" in result["error_msg"]


@serving_test
def test_restful_base64_b64_invalid_type_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]

    dtype = np.float16
    dtype_str_map = {np.float16: "fp16", np.float32: "fp32", np.float64: "fp64"}
    assert dtype in dtype_str_map

    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 1)] * (i + 1)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {'b64': 123, 'type': dtype_str_map[dtype], 'shape': [i + 1, i + 1]}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    assert "get scalar data failed, type is string, but json is not string type" in result["error_msg"]


@serving_test
def test_restful_base64_b64_invalid_value_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]

    dtype = np.float16
    dtype_str_map = {np.float16: "fp16", np.float32: "fp32", np.float64: "fp64"}
    assert dtype in dtype_str_map

    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 1)] * (i + 1)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        b64_val = base64.b64encode(val.tobytes()).decode()
        b64_val = '+==+==' + b64_val[:len('+==+==')]
        instance["float_val"] = {'b64': b64_val, 'type': dtype_str_map[dtype], 'shape': [i + 1, i + 1]}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    assert "is illegal b64 encode string" in result["error_msg"]


@serving_test
def test_restful_base64_b64_value_empty_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]

    dtype = np.float16
    dtype_str_map = {np.float16: "fp16", np.float32: "fp32", np.float64: "fp64"}
    assert dtype in dtype_str_map

    for i, instance in enumerate(instances):
        instance["float_val"] = {'b64': "", 'type': dtype_str_map[dtype], 'shape': [i + 1, i + 1]}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    assert "decode base64 size:0; Given info: type:float16; type size:2; element nums:1" in result["error_msg"]


@serving_test
def test_restful_base64_dtype_unknow_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]

    dtype = np.float16
    dtype_str_map = {np.float16: "fp16", np.float32: "fp32", np.float64: "fp64"}
    assert dtype in dtype_str_map

    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 1)] * (i + 1)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "dtype_unknow",
                                 'shape': [i + 1, i + 1]}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    assert "Parser request failed, json object, specified type:'dtype_unknow' is illegal" in result["error_msg"]


@serving_test
def test_restful_base64_dtype_empty_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]

    dtype = np.float16
    dtype_str_map = {np.float16: "fp16", np.float32: "fp32", np.float64: "fp64"}
    assert dtype in dtype_str_map

    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 1)] * (i + 1)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "",
                                 'shape': [i + 1, i + 1]}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    assert "Parser request failed, json object, specified type:'' is illegal" in result["error_msg"]


@serving_test
def test_restful_base64_dtype_invalid_type_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]

    dtype = np.float16
    dtype_str_map = {np.float16: "fp16", np.float32: "fp32", np.float64: "fp64"}
    assert dtype in dtype_str_map

    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 1)] * (i + 1)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': 1,
                                 'shape': [i + 1, i + 1]}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    assert "json object, key is 'type', value should be string type" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_dtype_not_match_empty_data_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]

    dtype = np.float16
    y_data_list = []
    for i, instance in enumerate(instances):
        val = [[]]
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp16",
                                 'shape': [i + 1, i + 1]}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    assert "Parser request failed, size is not matched" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_dtype_not_match_size_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]

    dtype = np.float16
    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 2)] * (i + 2)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp32",
                                 'shape': [i + 2, i + 2]}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    assert "Parser request failed, size is not matched" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_shape_large_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]

    dtype = np.float16
    dtype_str_map = {np.float16: "fp16", np.float32: "fp32", np.float64: "fp64"}
    assert dtype in dtype_str_map

    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 1)] * (i + 1)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': dtype_str_map[dtype],
                                 'shape': [i + 2, i + 2]}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    assert "Parser request failed, size is not matched" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_shape_small_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]

    dtype = np.float16
    dtype_str_map = {np.float16: "fp16", np.float32: "fp32", np.float64: "fp64"}
    assert dtype in dtype_str_map

    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 2)] * (i + 2)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': dtype_str_map[dtype],
                                 'shape': [i + 1, i + 1]}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    assert "Parser request failed, size is not matched" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_shape_small2_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]

    dtype = np.float16
    dtype_str_map = {np.float16: "fp16", np.float32: "fp32", np.float64: "fp64"}
    assert dtype in dtype_str_map

    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 2)] * (i + 2)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': dtype_str_map[dtype],
                                 'shape': [i + 2, i]}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    assert "Parser request failed, size is not matched" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_empty_shape_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]

    dtype = np.float16
    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 2)] * (i + 2)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp16",
                                 'shape': []}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    assert "Parser request failed, size is not matched" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_none_shape_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]

    dtype = np.float16
    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 2)] * (i + 2)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp16"}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    assert "Parser request failed, size is not matched" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_invalid_2d_shape_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]

    dtype = np.float16
    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 2)] * (i + 2)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp16", "shape": [[]]}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    assert "json object, key is 'shape', array value should be unsigned integer" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_invalid_shape_str_shape_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]

    dtype = np.float16
    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 2)] * (i + 2)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp16", "shape": ["abc"]}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    assert "json object, key is 'shape', array value should be unsigned integer" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_float_shape_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]

    dtype = np.float16
    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 2)] * (i + 2)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp16", "shape": [1.1]}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    assert "json object, key is 'shape', array value should be unsigned integer" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_negative_shape_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]

    dtype = np.float16
    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 2)] * (i + 2)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp16", "shape": [-1]}

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    assert "json object, key is 'shape', array value should be unsigned integer" in result["error_msg"]
