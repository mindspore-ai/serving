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

from mindspore_serving import master
from mindspore_serving import worker
from common import ServingTestBase, serving_test
from common import servable_config_import, servable_config_declare_servable
from common_restful import compare_float_value, create_multi_instances_fp32, post_restful


@serving_test
def test_restful_str_scalar_input_output_success():
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
def add_cast(x1, x2, label):
    y = register.call_servable(x1, x2)    
    y, text = register.call_postprocess(postprocess, y, label)
    return y, text
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    list_str = ["ABC", "DEF", "HIJ"]
    for i, instance in enumerate(instances):
        instance["label"] = list_str[i]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    result = result["instances"]
    assert result[0]["text"] == "ABC123"
    assert result[1]["text"] == "DEF456"
    assert result[2]["text"] == "HIJ789"


@serving_test
def test_restful_str_scalar_shape1_input_output_success():
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
def add_cast(x1, x2, label):
    y = register.call_servable(x1, x2)    
    y, text = register.call_postprocess(postprocess, y, label)
    return y, text
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    list_str = ["ABC", "DEF", "HIJ"]
    for i, instance in enumerate(instances):
        instance["label"] = [list_str[i]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    result = result["instances"]
    assert result[0]["text"] == "ABC123"
    assert result[1]["text"] == "DEF456"
    assert result[2]["text"] == "HIJ789"


@serving_test
def test_restful_empty_str_input_output_success():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
index = 0
list_str = ["123", "456", "789"]
def postprocess(y, label):
    global index
    if len(label) == 0:
        text = list_str[index]
    else:
        text = ""
    index = (index + 1) if index + 1 < len(list_str) else 0
    return y.astype(np.int32), text

@register.register_method(output_names=["y", "text"])
def add_cast(x1, x2, label):
    y = register.call_servable(x1, x2)    
    y, text = register.call_postprocess(postprocess, y, label)
    return y, text
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    list_str = ["ABC", "", "HIJ"]
    for i, instance in enumerate(instances):
        instance["label"] = list_str[i]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    result = result["instances"]
    assert result[0]["text"] == ""
    assert result[1]["text"] == "456"
    assert result[2]["text"] == ""


@serving_test
def test_restful_str_2d_array_one_item_input_output_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
index = 0
list_str = ["123", "456", "789"]
def postprocess(y, label):
    global index
    if len(label) == 0:
        text = list_str[index]
    else:
        text = ""
    index = (index + 1) if index + 1 < len(list_str) else 0
    return y.astype(np.int32), text

@register.register_method(output_names=["y", "text"])
def add_cast(x1, x2, label):
    y = register.call_servable(x1, x2)    
    y, text = register.call_postprocess(postprocess, y, label)
    return y, text
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    list_str = ["ABC", "", "HIJ"]
    for i, instance in enumerate(instances):
        instance["label"] = [[list_str[i]]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "bytes or string type input  shape can only be (1,) or empty, but given shape is [1, 1]" \
           in result["error_msg"]


@serving_test
def test_restful_str_1d_array_input_failed():
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
def add_cast(x1, x2, label):
    y = register.call_servable(x1, x2)    
    y, text = register.call_postprocess(postprocess, y, label)
    return y, text
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    list_str = ["ABC", "DEF", "HIJ"]
    for i, instance in enumerate(instances):
        instance["label"] = [list_str[i], list_str[i]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "json array, string or bytes type only support one item" in str(result["error_msg"])


@serving_test
def test_restful_str_invalid_array_input_failed():
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
def add_cast(x1, x2, label):
    y = register.call_servable(x1, x2)    
    y, text = register.call_postprocess(postprocess, y, label)
    return y, text
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    list_str = ["ABC", "DEF", "HIJ"]
    for i, instance in enumerate(instances):
        instance["label"] = [list_str[i], [list_str[i]]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "json array, string or bytes type only support one item" in str(result["error_msg"])


@serving_test
def test_restful_bool_scalar_input_output_success():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), not bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for i, instance in enumerate(instances):
        instance["bool_val"] = (i % 2 == 0)

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    result = result["instances"]
    assert not result[0]["value"]
    assert result[1]["value"]
    assert not result[2]["value"]


@serving_test
def test_restful_bool_1d_array_input_output_success():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for i, instance in enumerate(instances):
        instance["bool_val"] = [(i % 2 == 0)] * (i + 1)

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    result = result["instances"]
    assert result[0]["value"] == [False]
    assert result[1]["value"] == [True, True]
    assert result[2]["value"] == [False, False, False]


@serving_test
def test_restful_bool_2d_array_input_output_success():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for i, instance in enumerate(instances):
        val = (i % 2 == 0)
        val = [[val] * (i + 1)] * (i + 1)
        instance["bool_val"] = val

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    result = result["instances"]
    assert result[0]["value"] == [[False]]
    assert result[1]["value"] == [[True, True], [True, True]]
    assert result[2]["value"] == [[False, False, False], [False, False, False], [False, False, False]]


@serving_test
def test_restful_bool_invalid_array_array_scalar_mix_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for instance in instances:
        instance["bool_val"] = [[False], True]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "invalid json array: json type is not array" in result['error_msg']


@serving_test
def test_restful_bool_invalid_array2_scalar_array_mix_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for instance in instances:
        instance["bool_val"] = [False, [True]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "json array, data should be number, bool, string or bytes" in result['error_msg']


@serving_test
def test_restful_bool_invalid_array3_array_dim_not_match_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for instance in instances:
        instance["bool_val"] = [[False, True], [True]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "invalid json array: json size is 1, the dim 1 expected to be 2" in result['error_msg']


@serving_test
def test_restful_bool_invalid_array4_array_dim_not_match_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for instance in instances:
        instance["bool_val"] = [[[False, True]], [[True]]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "invalid json array: json size is 1, the dim 2 expected to be 2" in result['error_msg']


@serving_test
def test_restful_int_scalar_input_output_success():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, int_val):
    return y.astype(np.int32), int_val + 1

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, int_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, int_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for i, instance in enumerate(instances):
        val = i * 2
        instance["int_val"] = val

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    result = result["instances"]
    assert result[0]["value"] == 1
    assert result[1]["value"] == 3
    assert result[2]["value"] == 5


@serving_test
def test_restful_int_empty_input_output_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, int_val):
    return y.astype(np.int32), int_val + 1

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, int_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, int_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for i, instance in enumerate(instances):
        if i % 2 == 0:
            val = []
        else:
            val = [i * 2] * (i + 1)
        instance["int_val"] = val

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "json array, shape is empty" in result["error_msg"]


@serving_test
def test_restful_int_1d_array_input_output_success():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, int_val):
    return y.astype(np.int32), int_val + 1

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, int_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, int_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for i, instance in enumerate(instances):
        val = i * 2
        val = [val] * (i + 1)
        instance["int_val"] = val

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    result = result["instances"]
    assert result[0]["value"] == [1]
    assert result[1]["value"] == [3, 3]
    assert result[2]["value"] == [5, 5, 5]


@serving_test
def test_restful_int_2d_array_input_output_success():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, int_val):
    print("-----------------------int val ", int_val, int_val + 1)
    return y.astype(np.int32), int_val + 1

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, int_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, int_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for i, instance in enumerate(instances):
        val = i * 2
        val = [[val] * (i + 1)] * (i + 1)
        instance["int_val"] = val

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    result = result["instances"]
    assert result[0]["value"] == [[1]]
    assert result[1]["value"] == [[3, 3], [3, 3]]
    assert result[2]["value"] == [[5, 5, 5], [5, 5, 5], [5, 5, 5]]


@serving_test
def test_restful_float_scalar_input_output_success():
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
    for i, instance in enumerate(instances):
        val = i * 2.2
        instance["float_val"] = val

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    result = result["instances"]
    assert result[0]["value"] == 1
    assert abs(result[1]["value"] - (2.2 + 1)) < 0.001
    assert abs(result[2]["value"] - (4.4 + 1)) < 0.001


@serving_test
def test_restful_float_1d_array_input_output_success():
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
    for i, instance in enumerate(instances):
        val = [i * 2.2] * (i + 1)
        instance["float_val"] = val

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    result = result["instances"]
    compare_float_value(result[0]["value"], [1])
    compare_float_value(result[1]["value"], [3.2, 3.2])
    compare_float_value(result[2]["value"], [5.4, 5.4, 5.4])


@serving_test
def test_restful_float_2d_array_input_output_success():
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
    for i, instance in enumerate(instances):
        val = i * 2.2
        val = [[val] * (i + 1)] * (i + 1)
        instance["float_val"] = val

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    result = result["instances"]
    compare_float_value(result[0]["value"], [[1]])
    compare_float_value(result[1]["value"], [[3.2, 3.2], [3.2, 3.2]])
    compare_float_value(result[2]["value"], [[5.4, 5.4, 5.4], [5.4, 5.4, 5.4], [5.4, 5.4, 5.4]])


@serving_test
def test_restful_mix_bool_int_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for instance in instances:
        instance["bool_val"] = [[False, True], [1, 1]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "json array, elements type is not equal" in result['error_msg']


@serving_test
def test_restful_mix_bool_int2_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for instance in instances:
        instance["bool_val"] = [[False, 1]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "json array, elements type is not equal" in result['error_msg']


@serving_test
def test_restful_mix_float_int_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for instance in instances:
        instance["bool_val"] = [[1.1, 1.2], [1, 1]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "json array, elements type is not equal" in result['error_msg']


@serving_test
def test_restful_mix_float_int2_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for instance in instances:
        instance["bool_val"] = [[1.1, 1]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "json array, elements type is not equal" in result['error_msg']


@serving_test
def test_restful_mix_int_float_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for instance in instances:
        instance["bool_val"] = [[1, 1], [1.1, 1.2]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "json array, elements type is not equal" in result['error_msg']


@serving_test
def test_restful_mix_int_float2_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for instance in instances:
        instance["bool_val"] = [[1, 1.2]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "json array, elements type is not equal" in result['error_msg']


@serving_test
def test_restful_mix_str_float_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for instance in instances:
        instance["bool_val"] = [["a", "b"], [1.1, 1.2]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "string or bytes type only support one item" in result['error_msg']


@serving_test
def test_restful_mix_str_float2_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for instance in instances:
        instance["bool_val"] = [["a", 1.2]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "string or bytes type only support one item" in result['error_msg']


@serving_test
def test_restful_mix_float_str_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for instance in instances:
        instance["bool_val"] = [[1.1, 1.2], ["a", "b"]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "json array, elements type is not equal" in result['error_msg']


@serving_test
def test_restful_mix_float_str2_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for instance in instances:
        instance["bool_val"] = [[1.1, "b"]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "json array, elements type is not equal" in result['error_msg']


@serving_test
def test_restful_mix_bytes_str_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for instance in instances:
        instance["bool_val"] = [[{"b64": ""}, {"b64": ""}], ["a", "b"]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "string or bytes type only support one item" in result['error_msg']


@serving_test
def test_restful_mix_bytes_bool_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for instance in instances:
        instance["bool_val"] = [[{"b64": ""}, {"b64": ""}], [True, False]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "string or bytes type only support one item" in result['error_msg']


@serving_test
def test_restful_mix_bool_bytes_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, bool_val):
    return y.astype(np.int32), ~bool_val

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, bool_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, bool_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_restful_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for instance in instances:
        instance["bool_val"] = [[True, False], [{"b64": ""}, {"b64": ""}]]

    result = post_restful("localhost", 5500, base.servable_name, "add_cast", instances)
    assert "json array, data should be number, bool, string or bytes" in result['error_msg']
