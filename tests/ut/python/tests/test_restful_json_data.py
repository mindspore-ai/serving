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

from common import serving_test
from common_restful import compare_float_value, post_restful
from common_restful import start_str_restful_server, start_bool_int_float_restful_server


@serving_test
def test_restful_str_scalar_input_output_success():
    base = start_str_restful_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "DEF", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = str_a[i]
        instance["text2"] = str_b[i]

    result = post_restful("localhost", 5500, base.servable_name, "str_concat", instances)
    result = result["instances"]
    assert result[0]["text"] == str_a[0] + str_b[0]
    assert result[1]["text"] == str_a[1] + str_b[1]
    assert result[2]["text"] == str_a[2] + str_b[2]


@serving_test
def test_restful_str_scalar_shape1_input_output_success():
    base = start_str_restful_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "DEF", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = [str_a[i]]
        instance["text2"] = [str_b[i]]

    result = post_restful("localhost", 5500, base.servable_name, "str_concat", instances)
    result = result["instances"]
    assert result[0]["text"] == str_a[0] + str_b[0]
    assert result[1]["text"] == str_a[1] + str_b[1]
    assert result[2]["text"] == str_a[2] + str_b[2]


@serving_test
def test_restful_empty_str_input_output_success():
    base = start_str_restful_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = str_a[i]
        instance["text2"] = str_b[i]

    result = post_restful("localhost", 5500, base.servable_name, "str_empty", instances)
    result = result["instances"]
    assert result[0]["text"] == ""
    assert result[1]["text"] == "456"
    assert result[2]["text"] == ""


@serving_test
def test_restful_str_2d_array_one_item_input_output_failed():
    base = start_str_restful_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "DEF", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = [[str_a[i]]]
        instance["text2"] = [[str_b[i]]]

    result = post_restful("localhost", 5500, base.servable_name, "str_concat", instances)
    assert "bytes or string type input  shape can only be (1,) or empty, but given shape is [1, 1]" \
           in result["error_msg"]


@serving_test
def test_restful_str_1d_array_input_failed():
    base = start_str_restful_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "DEF", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = [str_a[i], str_a[i]]
        instance["text2"] = [str_b[i], str_b[i]]

    result = post_restful("localhost", 5500, base.servable_name, "str_concat", instances)
    assert "json array, string or bytes type only support one item" in str(result["error_msg"])


@serving_test
def test_restful_str_invalid_array_input_failed():
    base = start_str_restful_server()
    # Client
    instances = [{}, {}, {}]
    str_a = ["ABC", "DEF", "HIJ"]
    str_b = ["123", "456", "789"]
    for i, instance in enumerate(instances):
        instance["text1"] = [str_a[i], [str_a[i]]]
        instance["text2"] = [str_b[i], [str_b[i]]]

    result = post_restful("localhost", 5500, base.servable_name, "str_concat", instances)
    assert "json array, string or bytes type only support one item" in str(result["error_msg"])


@serving_test
def test_restful_bool_scalar_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        instance["bool_val"] = (i % 2 == 0)

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    result = result["instances"]
    assert not result[0]["value"]
    assert result[1]["value"]
    assert not result[2]["value"]


@serving_test
def test_restful_bool_1d_array_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        instance["bool_val"] = [(i % 2 == 0)] * (i + 1)

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    result = result["instances"]
    assert result[0]["value"] == [False]
    assert result[1]["value"] == [True, True]
    assert result[2]["value"] == [False, False, False]


@serving_test
def test_restful_bool_2d_array_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = (i % 2 == 0)
        val = [[val] * (i + 1)] * (i + 1)
        instance["bool_val"] = val

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    result = result["instances"]
    assert result[0]["value"] == [[False]]
    assert result[1]["value"] == [[True, True], [True, True]]
    assert result[2]["value"] == [[False, False, False], [False, False, False], [False, False, False]]


@serving_test
def test_restful_bool_invalid_array_array_scalar_mix_input_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for instance in instances:
        instance["bool_val"] = [[False], True]

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    assert "invalid json array: json type is not array" in result['error_msg']


@serving_test
def test_restful_bool_invalid_array2_scalar_array_mix_input_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for instance in instances:
        instance["bool_val"] = [False, [True]]

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    assert "json array, data should be number, bool, string or bytes" in result['error_msg']


@serving_test
def test_restful_bool_invalid_array3_array_dim_not_match_input_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for instance in instances:
        instance["bool_val"] = [[False, True], [True]]

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    assert "invalid json array: json size is 1, the dim 1 expected to be 2" in result['error_msg']


@serving_test
def test_restful_bool_invalid_array4_array_dim_not_match_input_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for instance in instances:
        instance["bool_val"] = [[[False, True]], [[True]]]

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    assert "invalid json array: json size is 1, the dim 2 expected to be 2" in result['error_msg']


@serving_test
def test_restful_int_scalar_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = i * 2
        instance["int_val"] = val

    result = post_restful("localhost", 5500, base.servable_name, "int_plus_1", instances)
    result = result["instances"]
    assert result[0]["value"] == 1
    assert result[1]["value"] == 3
    assert result[2]["value"] == 5


@serving_test
def test_restful_int_empty_input_output_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        if i % 2 == 0:
            val = []
        else:
            val = [i * 2] * (i + 1)
        instance["int_val"] = val

    result = post_restful("localhost", 5500, base.servable_name, "int_plus_1", instances)
    assert "json array, shape is empty" in result["error_msg"]


@serving_test
def test_restful_int_1d_array_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = i * 2
        val = [val] * (i + 1)
        instance["int_val"] = val

    result = post_restful("localhost", 5500, base.servable_name, "int_plus_1", instances)
    result = result["instances"]
    assert result[0]["value"] == [1]
    assert result[1]["value"] == [3, 3]
    assert result[2]["value"] == [5, 5, 5]


@serving_test
def test_restful_int_2d_array_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = i * 2
        val = [[val] * (i + 1)] * (i + 1)
        instance["int_val"] = val

    result = post_restful("localhost", 5500, base.servable_name, "int_plus_1", instances)
    result = result["instances"]
    assert result[0]["value"] == [[1]]
    assert result[1]["value"] == [[3, 3], [3, 3]]
    assert result[2]["value"] == [[5, 5, 5], [5, 5, 5], [5, 5, 5]]


@serving_test
def test_restful_float_scalar_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = i * 2.2
        instance["float_val"] = val

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    result = result["instances"]
    compare_float_value(result[0]["value"], 1.0)
    compare_float_value(result[1]["value"], 2.2 + 1)
    compare_float_value(result[2]["value"], 4.4 + 1)


@serving_test
def test_restful_float_1d_array_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = [i * 2.2] * (i + 1)
        instance["float_val"] = val

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    result = result["instances"]
    compare_float_value(result[0]["value"], [1.0])
    compare_float_value(result[1]["value"], [3.2, 3.2])
    compare_float_value(result[2]["value"], [5.4, 5.4, 5.4])


@serving_test
def test_restful_float_2d_array_input_output_success():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for i, instance in enumerate(instances):
        val = i * 2.2
        val = [[val] * (i + 1)] * (i + 1)
        instance["float_val"] = val

    result = post_restful("localhost", 5500, base.servable_name, "float_plus_1", instances)
    result = result["instances"]
    compare_float_value(result[0]["value"], [[1.0]])
    compare_float_value(result[1]["value"], [[3.2, 3.2], [3.2, 3.2]])
    compare_float_value(result[2]["value"], [[5.4, 5.4, 5.4], [5.4, 5.4, 5.4], [5.4, 5.4, 5.4]])


@serving_test
def test_restful_mix_bool_int_input_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for instance in instances:
        instance["bool_val"] = [[False, True], [1, 1]]

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    assert "json array, elements type is not equal" in result['error_msg']


@serving_test
def test_restful_mix_bool_int2_input_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for instance in instances:
        instance["bool_val"] = [[False, 1]]

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    assert "json array, elements type is not equal" in result['error_msg']


@serving_test
def test_restful_mix_float_int_input_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for instance in instances:
        instance["bool_val"] = [[1.1, 1.2], [1, 1]]

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    assert "json array, elements type is not equal" in result['error_msg']


@serving_test
def test_restful_mix_float_int2_input_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for instance in instances:
        instance["bool_val"] = [[1.1, 1]]

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    assert "json array, elements type is not equal" in result['error_msg']


@serving_test
def test_restful_mix_int_float_input_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for instance in instances:
        instance["bool_val"] = [[1, 1], [1.1, 1.2]]

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    assert "json array, elements type is not equal" in result['error_msg']


@serving_test
def test_restful_mix_int_float2_input_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for instance in instances:
        instance["bool_val"] = [[1, 1.2]]

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    assert "json array, elements type is not equal" in result['error_msg']


@serving_test
def test_restful_mix_str_float_input_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for instance in instances:
        instance["bool_val"] = [["a", "b"], [1.1, 1.2]]

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    assert "string or bytes type only support one item" in result['error_msg']


@serving_test
def test_restful_mix_str_float2_input_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for instance in instances:
        instance["bool_val"] = [["a", 1.2]]

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    assert "string or bytes type only support one item" in result['error_msg']


@serving_test
def test_restful_mix_float_str_input_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for instance in instances:
        instance["bool_val"] = [[1.1, 1.2], ["a", "b"]]

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    assert "json array, elements type is not equal" in result['error_msg']


@serving_test
def test_restful_mix_float_str2_input_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for instance in instances:
        instance["bool_val"] = [[1.1, "b"]]

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    assert "json array, elements type is not equal" in result['error_msg']


@serving_test
def test_restful_mix_bytes_str_input_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for instance in instances:
        instance["bool_val"] = [[{"b64": ""}, {"b64": ""}], ["a", "b"]]

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    assert "string or bytes type only support one item" in result['error_msg']


@serving_test
def test_restful_mix_bytes_bool_input_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for instance in instances:
        instance["bool_val"] = [[{"b64": ""}, {"b64": ""}], [True, False]]

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    assert "string or bytes type only support one item" in result['error_msg']


@serving_test
def test_restful_mix_bool_bytes_input_failed():
    base = start_bool_int_float_restful_server()
    # Client
    instances = [{}, {}, {}]
    for instance in instances:
        instance["bool_val"] = [[True, False], [{"b64": ""}, {"b64": ""}]]

    result = post_restful("localhost", 5500, base.servable_name, "bool_not", instances)
    assert "json array, data should be number, bool, string or bytes" in result['error_msg']
