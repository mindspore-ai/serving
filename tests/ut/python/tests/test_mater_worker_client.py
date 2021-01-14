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
from mindspore_serving import master
from mindspore_serving import worker
from mindspore_serving.client import Client
from common import ServingTestBase, serving_test


def create_multi_instances_fp32(instance_count):
    instances = []
    # instance 1
    y_data_list = []
    for i in range(instance_count):
        x1 = np.asarray([[1.1, 2.2], [3.3, 4.4]]).astype(np.float32) * (i + 1)
        x2 = np.asarray([[5.5, 6.6], [7.7, 8.8]]).astype(np.float32) * (i + 1)
        y_data_list.append(x1 + x2)
        instances.append({"x1": x1, "x2": x2})
    return instances, y_data_list


def check_result(result, y_data_list):
    assert len(result) == len(y_data_list)
    for result_item, y_data in zip(result, y_data_list):
        assert (result_item["y"] == y_data).all()


@serving_test
def test_master_worker_client_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    worker.start_servable_in_master(base.servable_dir, base.servable_name, 0)
    master.start_grpc_server("0.0.0.0", 5500)
    # Client
    client = Client("localhost", 5500, base.servable_name, "add_common")
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)
    result = client.infer(instances)
    client.close()  # avoid affecting the next use case

    print(result)
    check_result(result, y_data_list)


@serving_test
def test_master_worker_client_multi_times_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    worker.start_servable_in_master(base.servable_dir, base.servable_name, 0)
    master.start_grpc_server("0.0.0.0", 5500)
    # Client, use with avoid affecting the next use case
    with Client("localhost", 5500, base.servable_name, "add_common") as client:
        for instance_count in range(1, 5):
            instances, y_data_list = create_multi_instances_fp32(instance_count)
            result = client.infer(instances)
            check_result(result, y_data_list)


@serving_test
def test_master_worker_client_async_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    worker.start_servable_in_master(base.servable_dir, base.servable_name, 0)
    master.start_grpc_server("0.0.0.0", 5500)
    # Client
    client = Client("localhost", 5500, base.servable_name, "add_common")
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)
    result_future = client.infer_async(instances)
    result = result_future.result()
    client.close()  # avoid affecting the next use case

    print(result)
    check_result(result, y_data_list)


@serving_test
def test_master_worker_client_async_multi_times_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    worker.start_servable_in_master(base.servable_dir, base.servable_name, 0)
    master.start_grpc_server("0.0.0.0", 5500)
    # Client, use with avoid affecting the next use case
    with Client("localhost", 5500, base.servable_name, "add_common") as client:
        for instance_count in range(1, 5):
            instances, y_data_list = create_multi_instances_fp32(instance_count)
            result_future = client.infer_async(instances)
            result = result_future.result()
            check_result(result, y_data_list)


@serving_test
def test_master_worker_client_start_grpc_twice_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    worker.start_servable_in_master(base.servable_dir, base.servable_name, 0)
    master.start_grpc_server("0.0.0.0", 5500)
    try:
        master.start_grpc_server("0.0.0.0", 4500)
        assert False
    except RuntimeError as e:
        assert "Serving Error: Serving gRPC server is already running" in str(e)


@serving_test
def test_master_worker_client_start_master_grpc_twice_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    worker.start_servable_in_master(base.servable_dir, base.servable_name, 0)
    master.start_master_server("0.0.0.0", 5500)
    try:
        master.start_master_server("0.0.0.0", 4500)
        assert False
    except RuntimeError as e:
        assert "Serving Error: Master server is already running" in str(e)


@serving_test
def test_master_worker_client_start_restful_server_twice_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    worker.start_servable_in_master(base.servable_dir, base.servable_name, 0)
    master.start_restful_server("0.0.0.0", 5500)
    try:
        master.start_restful_server("0.0.0.0", 4500)
        assert False
    except RuntimeError as e:
        assert "Serving Error: RESTful server is already running" in str(e)


# test servable_config.py with client
servable_config_import = r"""
import numpy as np
from mindspore_serving.worker import register
"""

servable_config_declare_servable = r"""
register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
"""

servable_config_preprocess_cast = r"""
def add_trans_datatype(x1, x2):
    return x1.astype(np.float32), x2.astype(np.float32)
"""

servable_config_method_add_common = r"""
@register.register_method(output_names=["y"])
def add_common(x1, x2):  # only support float32 inputs
    y = register.call_servable(x1, x2)
    return y
"""

servable_config_method_add_cast = r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.call_preprocess(add_trans_datatype, x1, x2)  # cast input to float32
    y = register.call_servable(x1, x2)
    return y
"""


@serving_test
def test_master_worker_client_servable_content_success():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += servable_config_method_add_common
    servable_content += servable_config_method_add_cast

    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_grpc_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)
    with Client("localhost", 5500, base.servable_name, "add_common") as client:
        result = client.infer(instances)

    print(result)
    check_result(result, y_data_list)


@serving_test
def test_master_worker_client_preprocess_outputs_count_not_match_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def add_trans_datatype(x1, x2):
    return x1.astype(np.float32)

@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.call_preprocess(add_trans_datatype, x1, x2)  # cast input to float32
    y = register.call_servable(x1, x2)    
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_grpc_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    # Client, use with avoid affecting the next use case
    with Client("localhost", 5500, base.servable_name, "add_cast") as client:
        result = client.infer(instances)

    print(result)
    assert "Preprocess Failed" in str(result[0]["error"])


@serving_test
def test_master_worker_client_postprocess_outputs_count_not_match_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def add_trans_datatype(x1, x2):
    return x1.astype(np.float32)

@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.call_servable(x1, x2)    
    y, y2 = register.call_postprocess(add_trans_datatype, y, x2)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_grpc_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    # Client, use with avoid affecting the next use case
    with Client("localhost", 5500, base.servable_name, "add_cast") as client:
        result = client.infer(instances)

    print(result)
    assert "Postprocess Failed" in str(result[0]["error"])


@serving_test
def test_master_worker_client_str_input_output_success():
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
    master.start_grpc_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    list_str = ["ABC", "DEF", "HIJ"]
    for i, instance in enumerate(instances):
        instance["label"] = list_str[i]

    # Client, use with avoid affecting the next use case
    with Client("localhost", 5500, base.servable_name, "add_cast") as client:
        result = client.infer(instances)
    assert result[0]["text"] == "ABC123"
    assert result[1]["text"] == "DEF456"
    assert result[2]["text"] == "HIJ789"


@serving_test
def test_master_worker_client_bool_input_output_success():
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
    master.start_grpc_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for i, instance in enumerate(instances):
        instance["bool_val"] = (i % 2 == 0)

    # Client, use with avoid affecting the next use case
    with Client("localhost", 5500, base.servable_name, "add_cast") as client:
        result = client.infer(instances)
    assert not result[0]["value"]
    assert result[1]["value"]
    assert not result[2]["value"]


@serving_test
def test_master_worker_client_int_input_output_success():
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
    master.start_grpc_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for i, instance in enumerate(instances):
        instance["int_val"] = i * 2

    # Client, use with avoid affecting the next use case
    with Client("localhost", 5500, base.servable_name, "add_cast") as client:
        result = client.infer(instances)
    assert result[0]["value"] == 1
    assert result[1]["value"] == 3
    assert result[2]["value"] == 5


@serving_test
def test_master_worker_client_float_input_output_success():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def postprocess(y, float_val):
    return y.astype(np.int32), float_val + 1

@register.register_method(output_names=["y", "value"])
def add_cast(x1, x2, float_val):
    y = register.call_servable(x1, x2)    
    y, value = register.call_postprocess(postprocess, y, float_val)
    return y, value
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_grpc_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for i, instance in enumerate(instances):
        instance["float_val"] = i * 2.2

    # Client, use with avoid affecting the next use case
    with Client("localhost", 5500, base.servable_name, "add_cast") as client:
        result = client.infer(instances)
    assert result[0]["value"] == 1
    assert result[1]["value"] == (2.2 + 1)
    assert result[2]["value"] == (4.4 + 1)
