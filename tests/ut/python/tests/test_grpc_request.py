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
"""test Serving with master, worker and client"""

import numpy as np

from common import init_str_servable, init_bytes_servable, init_bool_int_float_servable
from common import serving_test, create_client
from mindspore_serving import server


def check_result(result, y_data_list):
    assert len(result) == len(y_data_list)
    for result_item, y_data in zip(result, y_data_list):
        assert (result_item["y"] == y_data).all()


def start_str_grpc_server():
    base = init_str_servable()
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    return base


def start_bytes_grpc_server():
    base = init_bytes_servable()
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    return base


def start_bool_int_float_grpc_server():
    base = init_bool_int_float_servable()
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    return base


@serving_test
def test_grpc_request_str_input_output_success():
    base = start_str_grpc_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "DEF", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = str_a[i]
        instance["text2"] = str_b[i]

    client = create_client("localhost:5500", base.servable_name, "str_concat")
    result = client.infer(instances)
    print("result", result)
    assert result[0]["text"] == str_a[0] + str_b[0]
    assert result[1]["text"] == str_a[1] + str_b[1]
    assert result[2]["text"] == str_a[2] + str_b[2]


@serving_test
def test_grpc_request_empty_str_input_output_success():
    base = start_str_grpc_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = str_a[i]
        instance["text2"] = str_b[i]

    client = create_client("localhost:5500", base.servable_name, "str_empty")
    result = client.infer(instances)
    assert result[0]["text"] == ""
    assert result[1]["text"] == "456"
    assert result[2]["text"] == ""


@serving_test
def test_grpc_request_str_shape1_list_input_failed():
    base = start_str_grpc_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "DEF", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = [str_a[i]]
        instance["text2"] = [str_b[i]]

    client = create_client("localhost:5500", base.servable_name, "str_concat")
    try:
        client.infer(instances)
        assert False
    except RuntimeError as e:
        assert "Not support value type <class 'list'>" in str(e)


@serving_test
def test_grpc_request_str_np_1d_array_input_failed():
    base = start_str_grpc_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "DEF", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = np.array([str_a[i], str_a[i]])
        instance["text2"] = np.array([str_b[i], str_b[i]])
        print(instance)

    client = create_client("localhost:5500", base.servable_name, "str_concat")
    try:
        client.infer(instances)
        assert False
    except RuntimeError as e:
        assert "Unknown data type" in str(e)


@serving_test
def test_grpc_request_bytes_input_output_success():
    base = start_bytes_grpc_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "DEF", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = str.encode(str_a[i])
        instance["text2"] = str.encode(str_b[i])

    client = create_client("localhost:5500", base.servable_name, "bytes_concat")
    result = client.infer(instances)
    assert bytes.decode(result[0]["text"]) == str_a[0] + str_b[0]
    assert bytes.decode(result[1]["text"]) == str_a[1] + str_b[1]
    assert bytes.decode(result[2]["text"]) == str_a[2] + str_b[2]


@serving_test
def test_grpc_request_empty_bytes_input_output_success():
    base = start_bytes_grpc_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = str.encode(str_a[i])
        instance["text2"] = str.encode(str_b[i])

    client = create_client("localhost:5500", base.servable_name, "bytes_empty")
    result = client.infer(instances)
    assert bytes.decode(result[0]["text"]) == ""
    assert bytes.decode(result[1]["text"]) == str_b[1]
    assert bytes.decode(result[2]["text"]) == ""


@serving_test
def test_grpc_request_bytes_1d_array_input_failed():
    base = start_bytes_grpc_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "DEF", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = [str.encode(str_a[i])]
        instance["text2"] = [str.encode(str_b[i])]

    client = create_client("localhost:5500", base.servable_name, "bytes_concat")
    try:
        client.infer(instances)
        assert False
    except RuntimeError as e:
        assert "Not support value type <class 'list'>" in str(e)


@serving_test
def test_grpc_request_bool_scalar_input_output_success():
    base = start_bool_int_float_grpc_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        instance["bool_val"] = (i % 2 == 0)

    client = create_client("localhost:5500", base.servable_name, "bool_not")
    result = client.infer(instances)
    assert not result[0]["value"]
    assert result[1]["value"]
    assert not result[2]["value"]


@serving_test
def test_grpc_request_bool_1d_array_input_output_success():
    base = start_bool_int_float_grpc_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = (i % 2 == 0)
        val = [val] * i
        instance["bool_val"] = np.array(val).astype(np.bool)

    client = create_client("localhost:5500", base.servable_name, "bool_not")
    result = client.infer(instances)
    assert result[0]["value"].tolist() == []
    assert result[1]["value"].tolist() == [True]
    assert result[2]["value"].tolist() == [False, False]


@serving_test
def test_grpc_request_bool_2d_array_input_output_success():
    base = start_bool_int_float_grpc_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = (i % 2 == 0)
        val = [[val] * i] * i
        if i == 0:
            val = [[]]
        instance["bool_val"] = np.array(val).astype(np.bool)

    client = create_client("localhost:5500", base.servable_name, "bool_not")
    result = client.infer(instances)
    assert result[0]["value"].tolist() == [[]]
    assert result[1]["value"].tolist() == [[True]]
    assert result[2]["value"].tolist() == [[False, False], [False, False]]


@serving_test
def test_grpc_request_bool_invalid_2d_array_input_failed():
    base = start_bool_int_float_grpc_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = (i % 2 == 0)
        val = [[val, val], [val]]
        instance["bool_val"] = np.array(val)

    client = create_client("localhost:5500", base.servable_name, "bool_not")
    try:
        client.infer(instances)
        assert False
    except RuntimeError as e:
        assert "Unknown data type object" in str(e)


@serving_test
def test_grpc_request_int_scalar_input_output_success():
    base = start_bool_int_float_grpc_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = (i * 2) * (-1 if i % 2 == 0 else 1)  # 0, 2, -4
        instance["int_val"] = val

    client = create_client("localhost:5500", base.servable_name, "int_plus_1")
    result = client.infer(instances)
    assert result[0]["value"] == 1
    assert result[1]["value"] == 3
    assert result[2]["value"] == -3


def common_test_grpc_request_np_int_type_scalar_input_output_success(dtype):
    base = start_bool_int_float_grpc_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = (i * 2) * (-1 if i % 2 == 0 else 1)  # 0, 2, -4
        instance["int_val"] = dtype(val)

    client = create_client("localhost:5500", base.servable_name, "int_plus_1")
    result = client.infer(instances)
    assert result[0]["value"] == 1
    assert result[1]["value"] == 3
    assert result[2]["value"] == -3


@serving_test
def test_grpc_request_np_int8_type_scalar_input_output_success():
    common_test_grpc_request_np_int_type_scalar_input_output_success(np.int8)


@serving_test
def test_grpc_request_np_int16_type_scalar_input_output_success():
    common_test_grpc_request_np_int_type_scalar_input_output_success(np.int16)


@serving_test
def test_grpc_request_np_int32_type_scalar_input_output_success():
    common_test_grpc_request_np_int_type_scalar_input_output_success(np.int32)


@serving_test
def test_grpc_request_np_int64_type_scalar_input_output_success():
    common_test_grpc_request_np_int_type_scalar_input_output_success(np.int64)


def common_test_grpc_request_np_uint_type_scalar_input_output_success(dtype):
    base = start_bool_int_float_grpc_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = (i * 2)  # 0, 2, 4
        instance["int_val"] = dtype(val)

    client = create_client("localhost:5500", base.servable_name, "int_plus_1")
    result = client.infer(instances)
    assert result[0]["value"] == 1
    assert result[1]["value"] == 3
    assert result[2]["value"] == 5


@serving_test
def test_grpc_request_np_uint8_type_scalar_input_output_success():
    common_test_grpc_request_np_uint_type_scalar_input_output_success(np.uint8)


@serving_test
def test_grpc_request_np_uint16_type_scalar_input_output_success():
    common_test_grpc_request_np_uint_type_scalar_input_output_success(np.uint16)


@serving_test
def test_grpc_request_np_uint32_type_scalar_input_output_success():
    common_test_grpc_request_np_uint_type_scalar_input_output_success(np.uint32)


@serving_test
def test_grpc_request_np_uint64_type_scalar_input_output_success():
    common_test_grpc_request_np_uint_type_scalar_input_output_success(np.uint64)


def common_test_grpc_request_np_int_type_1d_array_input_output_success(dtype):
    base = start_bool_int_float_grpc_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = (i * 2) * (-1 if i % 2 == 0 else 1)  # 0, 2, -4
        val = [val] * i
        instance["int_val"] = np.array(val).astype(dtype)

    client = create_client("localhost:5500", base.servable_name, "int_plus_1")
    result = client.infer(instances)
    assert result[0]["value"].tolist() == []
    assert result[1]["value"].tolist() == [3]
    assert result[2]["value"].tolist() == [-3, -3]


@serving_test
def test_grpc_request_np_int8_type_1d_array_input_output_success():
    common_test_grpc_request_np_int_type_1d_array_input_output_success(np.int8)


@serving_test
def test_grpc_request_np_int16_type_1d_array_input_output_success():
    common_test_grpc_request_np_int_type_1d_array_input_output_success(np.int16)


@serving_test
def test_grpc_request_np_int32_type_1d_array_input_output_success():
    common_test_grpc_request_np_int_type_1d_array_input_output_success(np.int32)


@serving_test
def test_grpc_request_np_int64_type_1d_array_input_output_success():
    common_test_grpc_request_np_int_type_1d_array_input_output_success(np.int64)


def common_test_grpc_request_np_uint_type_1d_array_input_output_success(dtype):
    base = start_bool_int_float_grpc_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = (i * 2)  # 0, 2, 4
        val = [val] * i
        instance["int_val"] = np.array(val).astype(dtype)

    client = create_client("localhost:5500", base.servable_name, "int_plus_1")
    result = client.infer(instances)
    assert result[0]["value"].tolist() == []
    assert result[1]["value"].tolist() == [3]
    assert result[2]["value"].tolist() == [5, 5]


@serving_test
def test_grpc_request_np_uint8_type_1d_array_input_output_success():
    common_test_grpc_request_np_uint_type_1d_array_input_output_success(np.uint8)


@serving_test
def test_grpc_request_np_uint16_type_1d_array_input_output_success():
    common_test_grpc_request_np_uint_type_1d_array_input_output_success(np.uint16)


@serving_test
def test_grpc_request_np_uint32_type_1d_array_input_output_success():
    common_test_grpc_request_np_uint_type_1d_array_input_output_success(np.uint32)


@serving_test
def test_grpc_request_np_uint64_type_1d_array_input_output_success():
    common_test_grpc_request_np_uint_type_1d_array_input_output_success(np.uint64)


def common_test_grpc_request_np_int_type_2d_array_input_output_success(dtype):
    base = start_bool_int_float_grpc_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = (i * 2) * (-1 if i % 2 == 0 else 1)  # 0, 2, -4
        val = [[val] * i] * i
        if i == 0:
            val = [[]]
        instance["int_val"] = np.array(val).astype(dtype)

    client = create_client("localhost:5500", base.servable_name, "int_plus_1")
    result = client.infer(instances)
    assert result[0]["value"].tolist() == [[]]
    assert result[1]["value"].tolist() == [[3]]
    assert result[2]["value"].tolist() == [[-3, -3], [-3, -3]]


@serving_test
def test_grpc_request_np_int8_type_2d_array_input_output_success():
    common_test_grpc_request_np_int_type_2d_array_input_output_success(np.int8)


@serving_test
def test_grpc_request_np_int16_type_2d_array_input_output_success():
    common_test_grpc_request_np_int_type_2d_array_input_output_success(np.int16)


@serving_test
def test_grpc_request_np_int32_type_2d_array_input_output_success():
    common_test_grpc_request_np_int_type_2d_array_input_output_success(np.int32)


@serving_test
def test_grpc_request_np_int64_type_2d_array_input_output_success():
    common_test_grpc_request_np_int_type_2d_array_input_output_success(np.int64)


def common_test_grpc_request_np_uint_type_2d_array_input_output_success(dtype):
    base = start_bool_int_float_grpc_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = (i * 2)  # 0, 2, 4
        val = [[val] * i] * i
        if i == 0:
            val = [[]]
        instance["int_val"] = np.array(val).astype(dtype)

    client = create_client("localhost:5500", base.servable_name, "int_plus_1")
    result = client.infer(instances)
    assert result[0]["value"].tolist() == [[]]
    assert result[1]["value"].tolist() == [[3]]
    assert result[2]["value"].tolist() == [[5, 5], [5, 5]]


@serving_test
def test_grpc_request_np_uint8_type_2d_array_input_output_success():
    common_test_grpc_request_np_uint_type_2d_array_input_output_success(np.uint8)


@serving_test
def test_grpc_request_np_uint16_type_2d_array_input_output_success():
    common_test_grpc_request_np_uint_type_2d_array_input_output_success(np.uint16)


@serving_test
def test_grpc_request_np_uint32_type_2d_array_input_output_success():
    common_test_grpc_request_np_uint_type_2d_array_input_output_success(np.uint32)


@serving_test
def test_grpc_request_np_uint64_type_2d_array_input_output_success():
    common_test_grpc_request_np_uint_type_2d_array_input_output_success(np.uint64)


@serving_test
def test_grpc_request_float_scalar_input_output_success():
    base = start_bool_int_float_grpc_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        instance["float_val"] = i * 2.2

    client = create_client("localhost:5500", base.servable_name, "float_plus_1")
    result = client.infer(instances)
    assert result[0]["value"] == 1
    assert result[1]["value"] == (2.2 + 1)
    assert result[2]["value"] == (4.4 + 1)


def common_test_grpc_request_np_float_type_scalar_input_output_success(dtype):
    base = start_bool_int_float_grpc_server()
    # Client
    instances = [{}, {}, {}]
    y_data_list = []
    for i, instance in enumerate(instances):
        val = (i * 2.2) * (-1 if i % 2 == 0 else 1)  # 0, 2.2, -4.4
        val = np.array(val).astype(dtype)
        y_data_list.append((val + 1).tolist())
        instance["float_val"] = val

    client = create_client("localhost:5500", base.servable_name, "float_plus_1")
    result = client.infer(instances)
    assert result[0]["value"].dtype == dtype
    assert result[1]["value"].dtype == dtype
    assert result[2]["value"].dtype == dtype
    assert result[0]["value"].tolist() == y_data_list[0]
    assert result[1]["value"].tolist() == y_data_list[1]
    assert result[2]["value"].tolist() == y_data_list[2]


@serving_test
def test_grpc_request_np_float16_scalar_input_output_success():
    common_test_grpc_request_np_float_type_scalar_input_output_success(np.float16)


@serving_test
def test_grpc_request_np_float32_scalar_input_output_success():
    common_test_grpc_request_np_float_type_scalar_input_output_success(np.float32)


@serving_test
def test_grpc_request_np_float64_scalar_input_output_success():
    common_test_grpc_request_np_float_type_scalar_input_output_success(np.float64)


def common_test_grpc_request_np_float_type_1d_array_input_output_success(dtype):
    base = start_bool_int_float_grpc_server()
    # Client
    instances = [{}, {}, {}]
    y_data_list = []
    for i, instance in enumerate(instances):
        val = (i * 2.2) * (-1 if i % 2 == 0 else 1)  # 0, 2.2, -4.4
        val = [val] * i
        val = np.array(val).astype(dtype)
        y_data_list.append((val + 1).tolist())
        instance["float_val"] = val

    client = create_client("localhost:5500", base.servable_name, "float_plus_1")
    result = client.infer(instances)
    assert result[0]["value"].dtype == dtype
    assert result[1]["value"].dtype == dtype
    assert result[2]["value"].dtype == dtype
    assert result[0]["value"].tolist() == y_data_list[0]
    assert result[1]["value"].tolist() == y_data_list[1]
    assert result[2]["value"].tolist() == y_data_list[2]


@serving_test
def test_grpc_request_np_float16_1d_array_input_output_success():
    common_test_grpc_request_np_float_type_1d_array_input_output_success(np.float16)


@serving_test
def test_grpc_request_np_float32_1d_array_input_output_success():
    common_test_grpc_request_np_float_type_1d_array_input_output_success(np.float32)


@serving_test
def test_grpc_request_np_float64_1d_array_input_output_success():
    common_test_grpc_request_np_float_type_1d_array_input_output_success(np.float64)


def common_test_grpc_request_np_float_type_2d_array_input_output_success(dtype):
    base = start_bool_int_float_grpc_server()
    # Client
    instances = [{}, {}, {}]
    y_data_list = []
    for i, instance in enumerate(instances):
        val = (i * 2.2) * (-1 if i % 2 == 0 else 1)  # 0, 2.2, -4.4
        val = [[val] * i] * i
        if i == 0:
            val = [[]]
        val = np.array(val).astype(dtype)
        y_data_list.append((val + 1).tolist())
        instance["float_val"] = val

    client = create_client("localhost:5500", base.servable_name, "float_plus_1")
    result = client.infer(instances)
    assert result[0]["value"].dtype == dtype
    assert result[1]["value"].dtype == dtype
    assert result[2]["value"].dtype == dtype
    assert result[0]["value"].tolist() == y_data_list[0]
    assert result[1]["value"].tolist() == y_data_list[1]
    assert result[2]["value"].tolist() == y_data_list[2]


@serving_test
def test_grpc_request_np_float16_2d_array_input_output_success():
    common_test_grpc_request_np_float_type_2d_array_input_output_success(np.float16)


@serving_test
def test_grpc_request_np_float32_2d_array_input_output_success():
    common_test_grpc_request_np_float_type_2d_array_input_output_success(np.float32)


@serving_test
def test_grpc_request_np_float64_2d_array_input_output_success():
    common_test_grpc_request_np_float_type_2d_array_input_output_success(np.float64)


@serving_test
def test_grpc_request_unix_domain_socket_success():
    base = init_str_servable()
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server_address = "unix:unix_socket_files/test_grpc_request_unix_domain_socket_success"
    server.start_grpc_server(server_address)
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "DEF", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = str_a[i]
        instance["text2"] = str_b[i]

    client = create_client(server_address, base.servable_name, "str_concat")
    result = client.infer(instances)
    print("result", result)
    assert result[0]["text"] == str_a[0] + str_b[0]
    assert result[1]["text"] == str_a[1] + str_b[1]
    assert result[2]["text"] == str_a[2] + str_b[2]
