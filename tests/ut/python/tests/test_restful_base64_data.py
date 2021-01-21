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
from common_restful import create_multi_instances_fp32, check_result, post_restful
from mindspore_serving import master
from mindspore_serving import worker


def start_str_restful_server():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
index = 0
list_str = ["123", "456", "789"]
def postprocess(y, label):
    global index
    text = list_str[index]
    index = (index + 1) if index + 1 < len(list_str) else 0
    return y.astype(np.int32), label + text

@register.register_method(output_names=["y", "text"])
def add_common(x1, x2, label):
    y = register.call_servable(x1, x2)    
    y, text = register.call_postprocess(postprocess, y, label)
    return y, text
    
def empty_postprocess(y, label):
    global index
    if len(label) == 0:
        text = list_str[index]
    else:
        text = ""
    index = (index + 1) if index + 1 < len(list_str) else 0
    return y.astype(np.int32), text

@register.register_method(output_names=["y", "text"])
def add_empty(x1, x2, label):
    y = register.call_servable(x1, x2)    
    y, text = register.call_postprocess(empty_postprocess, y, label)
    return y, text
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    return base


def start_bytes_restful_server():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
index = 0
list_str = ["123", "456", "789"]
def postprocess(y, label):
    global index
    label = bytes.decode(label.tobytes()) # bytes decode to str
    text = list_str[index]
    index = (index + 1) if index + 1 < len(list_str) else 0
    return y.astype(np.int32), str.encode(label + text) # str encode to bytes

@register.register_method(output_names=["y", "text"])
def add_common(x1, x2, label):
    y = register.call_servable(x1, x2)    
    y, text = register.call_postprocess(postprocess, y, label)
    return y, text

def empty_postprocess(y, label):
    global index
    label = bytes.decode(label.tobytes()) # bytes decode to str
    if len(label) == 0:
        text = list_str[index]
    else:
        text = ""
    index = (index + 1) if index + 1 < len(list_str) else 0
    return y.astype(np.int32), str.encode(text) # str encode to bytes

@register.register_method(output_names=["y", "text"])
def add_empty(x1, x2, label):
    y = register.call_servable(x1, x2)    
    y, text = register.call_postprocess(empty_postprocess, y, label)
    return y, text
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    return base


def start_bool_int_float_restful_server():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def bool_postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_bool(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(bool_postprocess, y, bool_val)
    return y, value

def int_postprocess(y, int_val):
    return y.astype(np.int32), int_val + 1

@register.register_method(output_names=["y", "value"])
def add_int(x1, x2, int_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(int_postprocess, y, int_val)
    return y, value
    
def float_postprocess(y, float_val):
    value = float_val + 1
    if value.dtype == np.float16:
        value = value.astype(np.float32)
    return y, value   
    
@register.register_method(output_names=["y", "value"])
def add_float(x1, x2, float_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(float_postprocess, y, float_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    return base


def common_test_restful_base64_str_scalar_input_output_success(shape):
    base = start_str_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    list_str = ["ABC", "DEF", "HIJ"]
    for i, instance in enumerate(instances):
        if shape is None:
            instance["label"] = {"b64": base64.b64encode(str.encode(list_str[i])).decode(), "type": "str"}
        else:
            instance["label"] = {"b64": base64.b64encode(str.encode(list_str[i])).decode(), "type": "str",
                                 'shape': shape}

    result = post_restful("localhost", 5500, base.servable_name, "add_common", instances)
    result = result["instances"]
    assert result[0]["text"] == "ABC123"
    assert result[1]["text"] == "DEF456"
    assert result[2]["text"] == "HIJ789"


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
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    list_str = ["ABC", "", "HIJ"]
    for i, instance in enumerate(instances):
        instance["label"] = {"b64": base64.b64encode(str.encode(list_str[i])).decode(), "type": "str"}

    result = post_restful("localhost", 5500, base.servable_name, "add_empty", instances)
    result = result["instances"]
    assert result[0]["text"] == ""
    assert result[1]["text"] == "456"
    assert result[2]["text"] == ""


@serving_test
def test_restful_base64_str_scalar_invalid_shape0_input_failed():
    base = start_str_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    list_str = ["ABC", "DEF", "HIJ"]
    for i, instance in enumerate(instances):
        instance["label"] = {"b64": base64.b64encode(str.encode(list_str[i])).decode(), "type": "str", "shape": [0]}

    result = post_restful("localhost", 5500, base.servable_name, "add_common", instances)
    assert "only support scalar when data type is string or bytes, please check 'type' or 'shape'" \
           in str(result["error_msg"])


@serving_test
def test_restful_base64_str_scalar_invalid_shape_input_failed():
    base = start_str_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    list_str = ["ABC", "DEF", "HIJ"]
    for i, instance in enumerate(instances):
        instance["label"] = {"b64": base64.b64encode(str.encode(list_str[i])).decode(), "type": "str", 'shape': [2]}

    result = post_restful("localhost", 5500, base.servable_name, "add_common", instances)
    assert "json object, only support scalar when data type is string or bytes, please check 'type' or 'shape'" \
           in str(result["error_msg"])


@serving_test
def test_restful_base64_str_1d_array_failed():
    base = start_str_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    list_str = ["ABC", "DEF", "HIJ"]
    for i, instance in enumerate(instances):
        instance["label"] = [{"b64": base64.b64encode(str.encode(list_str[i])).decode(), "type": "str"},
                             {"b64": base64.b64encode(str.encode(list_str[i])).decode(), "type": "str"}]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "json array, string or bytes type only support one item" in str(result["error_msg"])


def common_test_restful_bytes_input_output_success(shape):
    base = start_bytes_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    list_str = ["ABC", "DEF", "HIJ"]
    for i, instance in enumerate(instances):
        if shape is not None:
            instance["label"] = {"b64": base64.b64encode(str.encode(list_str[i])).decode(), "shape": shape}
        else:
            instance["label"] = {"b64": base64.b64encode(str.encode(list_str[i])).decode()}

    result = post_restful("localhost", 5500, base.servable_name, "add_common", instances)
    result = result["instances"]
    b64_decode_to_str = lambda a: bytes.decode(base64.b64decode(a["b64"]))
    assert b64_decode_to_str(result[0]["text"]) == "ABC123"
    assert b64_decode_to_str(result[1]["text"]) == "DEF456"
    assert b64_decode_to_str(result[2]["text"]) == "HIJ789"


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
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    list_str = ["ABC", "", "HIJ"]
    for i, instance in enumerate(instances):
        instance["label"] = {"b64": base64.b64encode(str.encode(list_str[i])).decode()}

    result = post_restful("localhost", 5500, base.servable_name, "add_empty", instances)
    result = result["instances"]
    b64_decode_to_str = lambda a: bytes.decode(base64.b64decode(a["b64"]))
    assert b64_decode_to_str(result[0]["text"]) == ""
    assert b64_decode_to_str(result[1]["text"]) == "456"
    assert b64_decode_to_str(result[2]["text"]) == ""


@serving_test
def test_restful_bytes_1d_array_failed():
    base = start_bytes_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    list_str = ["ABC", "DEF", "HIJ"]
    for i, instance in enumerate(instances):
        instance["label"] = [{"b64": base64.b64encode(str.encode(list_str[i])).decode()},
                             {"b64": base64.b64encode(str.encode(list_str[i])).decode()}]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "json array, string or bytes type only support one item" in str(result["error_msg"])


@serving_test
def test_restful_bytes_invalid_shape_input_failed():
    base = start_bytes_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    list_str = ["ABC", "DEF", "HIJ"]
    for i, instance in enumerate(instances):
        instance["label"] = {"b64": base64.b64encode(str.encode(list_str[i])).decode(), 'shape': [0]}

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "only support scalar when data type is string or bytes, please check 'type' or 'shape'" \
           in result["error_msg"]


@serving_test
def test_restful_base64_bool_scalar_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for i, instance in enumerate(instances):
        val = np.int8(i % 2 == 0)
        instance["bool_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "bool"}

    result = post_restful("localhost", 5500, base.servable_name, "add_bool", instances)
    result = result["instances"]
    assert not result[0]["value"]
    assert result[1]["value"]
    assert not result[2]["value"]


@serving_test
def test_restful_base64_bool_1d_array_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for i, instance in enumerate(instances):
        val = [(i % 2 == 0)] * (i + 1)
        val = np.array(val)
        instance["bool_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "bool", "shape": [i + 1]}

    result = post_restful("localhost", 5500, base.servable_name, "add_bool", instances)
    result = result["instances"]
    assert result[0]["value"] == [False]
    assert result[1]["value"] == [True, True]
    assert result[2]["value"] == [False, False, False]


@serving_test
def test_restful_base64_bool_2d_array_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for i, instance in enumerate(instances):
        val = (i % 2 == 0)
        val = [[val] * (i + 1)] * (i + 1)
        val = np.array(val)
        instance["bool_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "bool",
                                "shape": [i + 1, i + 1]}

    result = post_restful("localhost", 5500, base.servable_name, "add_bool", instances)
    result = result["instances"]
    assert result[0]["value"] == [[False]]
    assert result[1]["value"] == [[True, True], [True, True]]
    assert result[2]["value"] == [[False, False, False], [False, False, False], [False, False, False]]


@serving_test
def test_restful_base64_int_scalar_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for i, instance in enumerate(instances):
        val = np.int32(i * 2)
        instance["int_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "int32"}

    result = post_restful("localhost", 5500, base.servable_name, "add_int", instances)
    result = result["instances"]
    assert result[0]["value"] == 1
    assert result[1]["value"] == 3
    assert result[2]["value"] == 5


@serving_test
def test_restful_base64_int_1d_empty_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for i, instance in enumerate(instances):
        if i % 2 == 0:
            val = []
        else:
            val = [i * 2] * (i + 1)
        val = np.array(val).astype(np.int32)
        instance["int_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "int32", "shape": val.shape}

    result = post_restful("localhost", 5500, base.servable_name, "add_int", instances)
    result = result["instances"]
    assert result[0]["value"] == []
    assert result[1]["value"] == [3, 3]
    assert result[2]["value"] == []


@serving_test
def test_restful_base64_int_2d_empty_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for i, instance in enumerate(instances):
        if i % 2 == 0:
            val = [[]]
        else:
            val = [i * 2] * (i + 1)
        val = np.array(val).astype(np.int32)
        instance["int_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "int32", "shape": val.shape}

    result = post_restful("localhost", 5500, base.servable_name, "add_int", instances)
    result = result["instances"]
    assert result[0]["value"] == [[]]
    assert result[1]["value"] == [3, 3]
    assert result[2]["value"] == [[]]


@serving_test
def test_restful_base64_int_2d_empty_invalid_shape_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for _, instance in enumerate(instances):
        val = [[]]
        val = np.array(val).astype(np.int32)
        instance["int_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "int32", "shape": [1, 2, 0, 1]}

    result = post_restful("localhost", 5500, base.servable_name, "add_int", instances)
    assert "json object, key is 'shape', invalid shape value" in result["error_msg"]


@serving_test
def test_restful_base64_int_1d_array_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for i, instance in enumerate(instances):
        val = i * 2
        val = [val] * (i + 1)
        val = np.array(val).astype(np.int32)
        instance["int_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "int32", "shape": val.shape}

    result = post_restful("localhost", 5500, base.servable_name, "add_int", instances)
    result = result["instances"]
    assert result[0]["value"] == [1]
    assert result[1]["value"] == [3, 3]
    assert result[2]["value"] == [5, 5, 5]


def common_test_restful_base64_int_type_2d_array_input_output_success(dtype):
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    dtype_str_map = {np.int8: "int8", np.int16: "int16", np.int32: "int32", np.int64: "int64"}
    assert dtype in dtype_str_map
    for i, instance in enumerate(instances):
        val = (i + 1) * 2 * (-1 if i % 2 == 0 else 1)  # -2, 4, -6
        val = [[val] * (i + 1)] * (i + 1)
        val = np.array(val).astype(dtype)
        instance["int_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': dtype_str_map[dtype],
                               "shape": val.shape}

    result = post_restful("localhost", 5500, base.servable_name, "add_int", instances)
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
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    dtype_str_map = {np.uint8: "uint8", np.uint16: "uint16", np.uint32: "uint32", np.uint64: "uint64"}
    assert dtype in dtype_str_map
    for i, instance in enumerate(instances):
        val = i * 2
        val = [[val] * (i + 1)] * (i + 1)
        val = np.array(val).astype(dtype)
        instance["int_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': dtype_str_map[dtype],
                               "shape": val.shape}

    result = post_restful("localhost", 5500, base.servable_name, "add_int", instances)
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
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for i, instance in enumerate(instances):
        val = np.float32(i * 2.2)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp32"}

    result = post_restful("localhost", 5500, base.servable_name, "add_float", instances)
    result = result["instances"]
    assert result[0]["value"] == 1.0
    assert abs(result[1]["value"] - (2.2 + 1)) < 0.001
    assert abs(result[2]["value"] - (4.4 + 1)) < 0.001


@serving_test
def test_restful_base64_float_1d_array_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    y_data_list = []
    for i, instance in enumerate(instances):
        val = [i * 2.2 * (-1 if i % 2 == 0 else 1)] * (i + 1)  # [0], [2.2, 2.2], [-4.4, -4.4, -4.4]
        val = np.array(val).astype(np.float32)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp32", 'shape': [i + 1]}

    result = post_restful("localhost", 5500, base.servable_name, "add_float", instances)
    check_result(result, y_data_list, "value")


def common_test_restful_base64_float_type_2d_array_input_output_success(dtype):
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)

    dtype_str_map = {np.float16: "fp16", np.float32: "fp32", np.float64: "fp64"}
    assert dtype in dtype_str_map

    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 1)] * (i + 1)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': dtype_str_map[dtype],
                                 'shape': [i + 1, i + 1]}

    result = post_restful("localhost", 5500, base.servable_name, "add_float", instances)
    check_result(result, y_data_list, "value")


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
def test_restful_base64_float16_2d_array_not_support_fp16_output_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, float_val):
    return y, float_val + 1    

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, float_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, float_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)

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
                                 'shape': [i + 1, i + 1]}

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "fp16 reply is not supported" in result["error_msg"]


@serving_test
def test_restful_base64_dtype_unknow_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)

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

    result = post_restful("localhost", 5500, base.servable_name, "add_float", instances)
    assert "Parser request failed, json object, specified type:'dtype_unknow' is illegal" in result["error_msg"]


@serving_test
def test_restful_base64_dtype_empty_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)

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

    result = post_restful("localhost", 5500, base.servable_name, "add_float", instances)
    assert "Parser request failed, json object, specified type:'' is illegal" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_shape_not_match1_large_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)

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

    result = post_restful("localhost", 5500, base.servable_name, "add_float", instances)
    assert "Parser request failed, size is not matched" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_shape_not_match2_small_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)

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

    result = post_restful("localhost", 5500, base.servable_name, "add_float", instances)
    assert "Parser request failed, size is not matched" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_shape_not_match3_small_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)

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

    result = post_restful("localhost", 5500, base.servable_name, "add_float", instances)
    assert "Parser request failed, size is not matched" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_dtype_not_match4_empty_data_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)

    dtype = np.float16
    y_data_list = []
    for i, instance in enumerate(instances):
        val = [[]]
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp16",
                                 'shape': [i + 1, i + 1]}

    result = post_restful("localhost", 5500, base.servable_name, "add_float", instances)
    assert "Parser request failed, size is not matched" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_dtype_not_match5_empty_shape_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)

    dtype = np.float16
    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 2)] * (i + 2)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp16",
                                 'shape': []}

    result = post_restful("localhost", 5500, base.servable_name, "add_float", instances)
    assert "Parser request failed, size is not matched" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_dtype_not_match6_empty_shape3_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)

    dtype = np.float16
    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 2)] * (i + 2)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp16"}

    result = post_restful("localhost", 5500, base.servable_name, "add_float", instances)
    assert "Parser request failed, size is not matched" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_dtype_not_match_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)

    dtype = np.float16
    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 2)] * (i + 2)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp32",
                                 'shape': [i + 2, i + 2]}

    result = post_restful("localhost", 5500, base.servable_name, "add_float", instances)
    assert "Parser request failed, size is not matched" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_invalid_shape_2d_shape_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)

    dtype = np.float16
    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 2)] * (i + 2)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp16", "shape": [[]]}

    result = post_restful("localhost", 5500, base.servable_name, "add_float", instances)
    assert "json object, key is 'shape', array value should be unsigned integer" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_invalid_shape2_str_shape_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)

    dtype = np.float16
    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 2)] * (i + 2)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp16", "shape": ["abc"]}

    result = post_restful("localhost", 5500, base.servable_name, "add_float", instances)
    assert "json object, key is 'shape', array value should be unsigned integer" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_invalid_shape3_float_shape_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)

    dtype = np.float16
    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 2)] * (i + 2)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp16", "shape": [1.1]}

    result = post_restful("localhost", 5500, base.servable_name, "add_float", instances)
    assert "json object, key is 'shape', array value should be unsigned integer" in result["error_msg"]


@serving_test
def test_restful_base64_float16_2d_array_invalid_shape4_negative_shape_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)

    dtype = np.float16
    y_data_list = []
    for i, instance in enumerate(instances):
        val = i * 2.2 * (-1 if i % 2 == 0 else 1)  # 0, 2.2 ,-4.4
        val = [[val] * (i + 2)] * (i + 2)
        val = np.array(val).astype(dtype)
        y_data_list.append(val + 1)
        instance["float_val"] = {"b64": base64.b64encode(val.tobytes()).decode(), 'type': "fp16", "shape": [-1]}

    result = post_restful("localhost", 5500, base.servable_name, "add_float", instances)
    assert "json object, key is 'shape', array value should be unsigned integer" in result["error_msg"]
