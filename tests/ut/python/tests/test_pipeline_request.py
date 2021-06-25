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
"""test Serving pipeline with client"""

import numpy as np

from common import ServingTestBase
from common import serving_test, create_client
from mindspore_serving import server


def check_result(result, y_data_list):
    assert len(result) == len(y_data_list)
    for result_item, y_data in zip(result, y_data_list):
        assert (result_item["y"] == y_data).all()


def start_serving_server(extra=None):
    base = ServingTestBase()
    servable_content = f"""
import numpy as np
from mindspore_serving.server import register
from mindspore_serving.server.register import PipelineServable

register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
def preprocess(other):
    return np.ones([2,2], np.float32), np.ones([2,2], np.float32)
    
def postprocess_one(x1):
    return x1

def postprocess_two(x1, x2):
    return x1, x2

@register.register_method(output_names=["x1"])
def method_one(x1):
    a1, a2 = register.call_preprocess(preprocess, x1)
    y = register.call_servable(a1, a2)    
    x1 = register.call_postprocess(postprocess_one, x1)
    return x1

@register.register_method(output_names=["x1", "x2"])
def method_two(x1, x2):
    a1, a2 = register.call_preprocess(preprocess, x1)
    y = register.call_servable(a1, a2)    
    x1, x2 = register.call_postprocess(postprocess_two, x1, x2)
    return x1, x2

servable_one = PipelineServable('{base.servable_name}', 'method_one')
servable_two = PipelineServable('{base.servable_name}', 'method_two')

@register.register_pipeline(output_names="x1")
def pipeline_one(x1):
    x1 = servable_one.run(x1)
    return x1

@register.register_pipeline(output_names=["x1", "x2"])
def pipeline_two(x1, x2):
    x1, x2 = servable_two.run(x1, x2)
    return x1, x2
"""
    if extra:
        servable_content += extra
    base.init_servable_with_servable_config(1, servable_content)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    return base


def start_serving_server_batch(extra=None):
    base = ServingTestBase()
    servable_content = f"""
import numpy as np
from mindspore_serving.server import register
from mindspore_serving.server.register import PipelineServable

register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)
def preprocess(other):
    return np.ones([2], np.float32), np.ones([2], np.float32)
    
def postprocess_one(x1):
    return x1

def postprocess_two(x1, x2):
    return x1, x2

@register.register_method(output_names=["x1"])
def method_one(x1):
    a1, a2 = register.call_preprocess(preprocess, x1)
    y = register.call_servable(a1, a2)    
    x1 = register.call_postprocess(postprocess_one, x1)
    return x1

@register.register_method(output_names=["x1", "x2"])
def method_two(x1, x2):
    a1, a2 = register.call_preprocess(preprocess, x1)
    y = register.call_servable(a1, a2)    
    x1, x2 = register.call_postprocess(postprocess_two, x1, x2)
    return x1, x2

servable_one = PipelineServable('{base.servable_name}', 'method_one')
servable_two = PipelineServable('{base.servable_name}', 'method_two')

@register.register_pipeline(output_names="x1")
def pipeline_one(x1):
    x1 = servable_one.run(x1)
    return x1

@register.register_pipeline(output_names=["x1", "x2"])
def pipeline_two(x1, x2):
    x1, x2 = servable_two.run(x1, x2)
    return x1, x2
"""
    if extra:
        servable_content += extra
    base.init_servable_with_servable_config(1, servable_content)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    return base


@serving_test
def test_pipeline_request_str_one_success():
    base = start_serving_server()
    # Client
    instances = [{}, {}, {}]
    input_x1 = ["ABC", "DEF", "HIJ"]
    for i, instance in enumerate(instances):
        instance["x1"] = input_x1[i]

    client = create_client("localhost:5500", base.servable_name, "pipeline_one")
    result = client.infer(instances)
    print("result", result)
    assert result[0]["x1"] == input_x1[0]
    assert result[1]["x1"] == input_x1[1]
    assert result[2]["x1"] == input_x1[2]


@serving_test
def test_pipeline_request_int_one_success():
    base = start_serving_server()
    # Client
    instances = [{}, {}, {}]
    input_x1 = [11, 12, 13]
    for i, instance in enumerate(instances):
        instance["x1"] = input_x1[i]

    client = create_client("localhost:5500", base.servable_name, "pipeline_one")
    result = client.infer(instances)
    print("result", result)
    assert result[0]["x1"] == input_x1[0]
    assert result[1]["x1"] == input_x1[1]
    assert result[2]["x1"] == input_x1[2]


@serving_test
def test_pipeline_request_bool_one_success():
    base = start_serving_server()
    # Client
    instances = [{}, {}, {}]
    input_x1 = [True, False, True]
    for i, instance in enumerate(instances):
        instance["x1"] = input_x1[i]

    client = create_client("localhost:5500", base.servable_name, "pipeline_one")
    result = client.infer(instances)
    print("result", result)
    assert result[0]["x1"] == input_x1[0]
    assert result[1]["x1"] == input_x1[1]
    assert result[2]["x1"] == input_x1[2]


@serving_test
def test_pipeline_request_float_one_success():
    base = start_serving_server()
    # Client
    instances = [{}, {}, {}]
    input_x1 = [1.1, 2.2, 3.3]
    for i, instance in enumerate(instances):
        instance["x1"] = input_x1[i]

    client = create_client("localhost:5500", base.servable_name, "pipeline_one")
    result = client.infer(instances)
    print("result", result)
    assert result[0]["x1"] == input_x1[0]
    assert result[1]["x1"] == input_x1[1]
    assert result[2]["x1"] == input_x1[2]


@serving_test
def test_pipeline_request_np_array_one_success():
    base = start_serving_server()
    # Client
    instances = [{}, {}, {}]
    data = np.array([1.1, 2.2])
    input_x1 = [data, data * 2, data * 3]
    for i, instance in enumerate(instances):
        instance["x1"] = input_x1[i]

    client = create_client("localhost:5500", base.servable_name, "pipeline_one")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["x1"] == input_x1[0]).all()
    assert (result[1]["x1"] == input_x1[1]).all()
    assert (result[2]["x1"] == input_x1[2]).all()


@serving_test
def test_pipeline_request_str_int_two_success():
    base = start_serving_server()
    # Client
    instances = [{}, {}, {}]
    input_x1 = ["ABC", "DEF", "GHI"]
    input_x2 = [1, 2, 3]
    for i, instance in enumerate(instances):
        instance["x1"] = input_x1[i]
        instance["x2"] = input_x2[i]

    client = create_client("localhost:5500", base.servable_name, "pipeline_two")
    result = client.infer(instances)
    print("result", result)
    assert result[0]["x1"] == input_x1[0]
    assert result[1]["x1"] == input_x1[1]
    assert result[2]["x1"] == input_x1[2]
    assert result[0]["x2"] == input_x2[0]
    assert result[1]["x2"] == input_x2[1]
    assert result[2]["x2"] == input_x2[2]


@serving_test
def test_pipeline_request_bool_np_array_two_success():
    base = start_serving_server()
    # Client
    instances = [{}, {}, {}]
    input_x1 = [True, False, True]
    data = np.array([1.1, 2.2])
    input_x2 = [data, data * 2, data * 3]
    for i, instance in enumerate(instances):
        instance["x1"] = input_x1[i]
        instance["x2"] = input_x2[i]

    client = create_client("localhost:5500", base.servable_name, "pipeline_two")
    result = client.infer(instances)
    print("result", result)
    assert result[0]["x1"] == input_x1[0]
    assert result[1]["x1"] == input_x1[1]
    assert result[2]["x1"] == input_x1[2]
    assert (result[0]["x2"] == input_x2[0]).all()
    assert (result[1]["x2"] == input_x2[1]).all()
    assert (result[2]["x2"] == input_x2[2]).all()


@serving_test
def test_pipeline_request_input_not_match_failed():
    base = start_serving_server()
    # Client
    instances = [{}, {}, {}]
    input_x1 = [True, False, True]
    data = np.array([1.1, 2.2])
    input_x2 = [data, data * 2, data * 3]
    for i, instance in enumerate(instances):
        instance["x1"] = input_x1[i]
        instance["x3"] = input_x2[i]

    client = create_client("localhost:5500", base.servable_name, "pipeline_two")
    result = client.infer(instances)
    print("result", result)
    assert "Cannot find input x2 in instance input" in result["error"]


@serving_test
def test_pipeline_request_input_not_match2_failed():
    base = start_serving_server()
    # Client
    instances = [{}, {}, {}]
    input_x1 = [True, False, True]
    data = np.array([1.1, 2.2])
    input_x2 = [data, data * 2, data * 3]
    for i, instance in enumerate(instances):
        instance["x1"] = input_x1[i]
        if i == 1:
            instance["x3"] = input_x2[i]
        else:
            instance["x2"] = input_x2[i]

    client = create_client("localhost:5500", base.servable_name, "pipeline_two")
    result = client.infer(instances)
    print("result", result)
    assert "Cannot find input x2 in instance input" in result["error"]


@serving_test
def test_pipeline_request_run_input_count_not_match_failed():
    extra = r"""
@register.register_pipeline(output_names=["x1", "x2"])
def pipeline_two_run_input_count_not_match(x1, x2):
    if not x1:
        x3 = 123
        x1, x2 = servable_two.run(x1, x2, x3)
    else:
        x1, x2 = servable_two.run(x1, x2)
    return x1, x2
    """
    base = start_serving_server(extra)
    # Client
    instances = [{}, {}, {}]
    input_x1 = [True, False, True]
    data = np.array([1.1, 2.2])
    input_x2 = [data, data * 2, data * 3]
    for i, instance in enumerate(instances):
        instance["x1"] = input_x1[i]
        instance["x2"] = input_x2[i]

    client = create_client("localhost:5500", base.servable_name, "pipeline_two_run_input_count_not_match")
    result = client.infer(instances)
    print("result", result)
    assert "Pipeline Failed" in result[1]["error"]
    assert result[0]["x1"] == input_x1[0]
    assert result[2]["x1"] == input_x1[2]
    assert (result[0]["x2"] == input_x2[0]).all()
    assert (result[2]["x2"] == input_x2[2]).all()


@serving_test
def test_pipeline_request_run_input_count_not_match2_failed():
    extra = r"""
@register.register_pipeline(output_names=["x1", "x2"])
def pipeline_two_run_input_count_not_match(x1, x2):
    if not x1:
        x1, x2 = servable_two.run(x1)
    else:
        x1, x2 = servable_two.run(x1, x2)
    return x1, x2
    """
    base = start_serving_server(extra)
    # Client
    instances = [{}, {}, {}]
    input_x1 = [True, False, True]
    data = np.array([1.1, 2.2])
    input_x2 = [data, data * 2, data * 3]
    for i, instance in enumerate(instances):
        instance["x1"] = input_x1[i]
        instance["x2"] = input_x2[i]

    client = create_client("localhost:5500", base.servable_name, "pipeline_two_run_input_count_not_match")
    result = client.infer(instances)
    print("result", result)
    assert "Pipeline Failed" in result[1]["error"]
    assert result[0]["x1"] == input_x1[0]
    assert result[2]["x1"] == input_x1[2]
    assert (result[0]["x2"] == input_x2[0]).all()
    assert (result[2]["x2"] == input_x2[2]).all()


@serving_test
def test_pipeline_request_run_output_count_not_match_failed():
    extra = r"""
@register.register_pipeline(output_names=["x1", "x2"])
def pipeline_two_run_output_count_not_match(x1, x2):
    if not x1:
        x1, x2, x3 = servable_two.run(x1, x2)
    else:
        x1, x2 = servable_two.run(x1, x2)
    return x1, x2
    """
    base = start_serving_server(extra)
    # Client
    instances = [{}, {}, {}]
    input_x1 = [True, False, True]
    data = np.array([1.1, 2.2])
    input_x2 = [data, data * 2, data * 3]
    for i, instance in enumerate(instances):
        instance["x1"] = input_x1[i]
        instance["x2"] = input_x2[i]

    client = create_client("localhost:5500", base.servable_name, "pipeline_two_run_output_count_not_match")
    result = client.infer(instances)
    print("result", result)
    assert "Pipeline Failed" in result[1]["error"]
    assert result[0]["x1"] == input_x1[0]
    assert result[2]["x1"] == input_x1[2]
    assert (result[0]["x2"] == input_x2[0]).all()
    assert (result[2]["x2"] == input_x2[2]).all()


@serving_test
def test_pipeline_request_pipeline_output_count_not_match_failed():
    extra = r"""
@register.register_pipeline(output_names=["x1", "x2"])
def pipeline_two_pipeline_output_count_not_match(x1, x2):
    x1, x2 = servable_two.run(x1, x2)
    if not x1:
        return x1, x2, 123
    return x1, x2
    """
    base = start_serving_server(extra)
    # Client
    instances = [{}, {}, {}]
    input_x1 = [True, False, True]
    data = np.array([1.1, 2.2])
    input_x2 = [data, data * 2, data * 3]
    for i, instance in enumerate(instances):
        instance["x1"] = input_x1[i]
        instance["x2"] = input_x2[i]

    client = create_client("localhost:5500", base.servable_name, "pipeline_two_pipeline_output_count_not_match")
    result = client.infer(instances)
    print("result", result)
    assert "Pipeline Failed" in result[1]["error"]
    assert result[0]["x1"] == input_x1[0]
    assert result[2]["x1"] == input_x1[2]
    assert (result[0]["x2"] == input_x2[0]).all()
    assert (result[2]["x2"] == input_x2[2]).all()


@serving_test
def test_pipeline_request_invalid_pipeline_output_type_failed():
    extra = r"""
@register.register_pipeline(output_names=["x1", "x2"])
def pipeline_two_invalid(x1, x2):
    x1, x2 = servable_two.run(x1, x2)
    if not x1:
        return x1, [123, [456]]
    return x1, x2
    """
    base = start_serving_server(extra)
    # Client
    instances = [{}, {}, {}]
    input_x1 = [True, False, True]
    data = np.array([1.1, 2.2])
    input_x2 = [data, data * 2, data * 3]
    for i, instance in enumerate(instances):
        instance["x1"] = input_x1[i]
        instance["x2"] = input_x2[i]

    client = create_client("localhost:5500", base.servable_name, "pipeline_two_invalid")
    result = client.infer(instances)
    print("result", result)
    assert "Pipeline Failed" in result[1]["error"]
    assert result[0]["x1"] == input_x1[0]
    assert result[2]["x1"] == input_x1[2]
    assert (result[0]["x2"] == input_x2[0]).all()
    assert (result[2]["x2"] == input_x2[2]).all()


@serving_test
def test_pipeline_request_invalid_run_input_type_failed():
    extra = r"""
@register.register_pipeline(output_names=["x1", "x2"])
def pipeline_two_invalid(x1, x2):
    if not x1:
        x1, x2 = servable_two.run(x1, [123, [456]])
    else:
        x1, x2 = servable_two.run(x1, x2)
    return x1, x2
    """
    base = start_serving_server(extra)
    # Client
    instances = [{}, {}, {}]
    input_x1 = [True, False, True]
    data = np.array([1.1, 2.2])
    input_x2 = [data, data * 2, data * 3]
    for i, instance in enumerate(instances):
        instance["x1"] = input_x1[i]
        instance["x2"] = input_x2[i]

    client = create_client("localhost:5500", base.servable_name, "pipeline_two_invalid")
    result = client.infer(instances)
    print("result", result)
    assert "Pipeline Failed" in result[1]["error"]
    assert result[0]["x1"] == input_x1[0]
    assert result[2]["x1"] == input_x1[2]
    assert (result[0]["x2"] == input_x2[0]).all()
    assert (result[2]["x2"] == input_x2[2]).all()


@serving_test
def test_pipeline_request_pipeline_exception_failed():
    extra = r"""
@register.register_pipeline(output_names=["x1", "x2"])
def pipeline_two_invalid(x1, x2):
    if not x1:
        raise RuntimeError("error test")
    else:
        x1, x2 = servable_two.run(x1, x2)
    return x1, x2
    """
    base = start_serving_server(extra)
    # Client
    instances = [{}, {}, {}]
    input_x1 = [True, False, True]
    data = np.array([1.1, 2.2])
    input_x2 = [data, data * 2, data * 3]
    for i, instance in enumerate(instances):
        instance["x1"] = input_x1[i]
        instance["x2"] = input_x2[i]

    client = create_client("localhost:5500", base.servable_name, "pipeline_two_invalid")
    result = client.infer(instances)
    print("result", result)
    assert "Pipeline Failed" in result[1]["error"]
    assert result[0]["x1"] == input_x1[0]
    assert result[2]["x1"] == input_x1[2]
    assert (result[0]["x2"] == input_x2[0]).all()
    assert (result[2]["x2"] == input_x2[2]).all()


@serving_test
def test_pipeline_request_batch_str_one_success():
    base = start_serving_server_batch()
    # Client
    instances = [{}, {}, {}]
    input_x1 = ["ABC", "DEF", "HIJ"]
    for i, instance in enumerate(instances):
        instance["x1"] = input_x1[i]

    client = create_client("localhost:5500", base.servable_name, "pipeline_one")
    result = client.infer(instances)
    print("result", result)
    assert result[0]["x1"] == input_x1[0]
    assert result[1]["x1"] == input_x1[1]
    assert result[2]["x1"] == input_x1[2]


@serving_test
def test_pipeline_request_batch_bool_np_array_two_success():
    base = start_serving_server_batch()
    # Client
    instances = [{}, {}, {}]
    input_x1 = [True, False, True]
    data = np.array([1.1])
    input_x2 = [data, data * 2, data * 3]
    for i, instance in enumerate(instances):
        instance["x1"] = input_x1[i]
        instance["x2"] = input_x2[i]

    client = create_client("localhost:5500", base.servable_name, "pipeline_two")
    result = client.infer(instances)
    print("result", result)
    assert result[0]["x1"] == input_x1[0]
    assert result[1]["x1"] == input_x1[1]
    assert result[2]["x1"] == input_x1[2]
    assert (result[0]["x2"] == input_x2[0]).all()
    assert (result[1]["x2"] == input_x2[1]).all()
    assert (result[2]["x2"] == input_x2[2]).all()
