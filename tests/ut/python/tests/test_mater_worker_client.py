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
from common import ServingTestBase, serving_test, release_client
from common import servable_config_import, servable_config_declare_servable, servable_config_preprocess_cast
from common import servable_config_method_add_common, servable_config_method_add_cast


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
    release_client(client)

    print(result)
    check_result(result, y_data_list)


@serving_test
def test_master_worker_client_multi_times_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    worker.start_servable_in_master(base.servable_dir, base.servable_name, 0)
    master.start_grpc_server("0.0.0.0", 5500)
    # Client, use with avoid affecting the next use case
    client = Client("localhost", 5500, base.servable_name, "add_common")
    for instance_count in range(1, 5):
        instances, y_data_list = create_multi_instances_fp32(instance_count)
        result = client.infer(instances)
        check_result(result, y_data_list)
    release_client(client)


@serving_test
def test_master_worker_client_alone_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    master.start_master_server(master_port=7600)
    master.start_grpc_server("0.0.0.0", 5500)
    worker.start_servable(base.servable_dir, base.servable_name, master_port=7600, worker_port=6600)
    # Client
    client = Client("localhost", 5500, base.servable_name, "add_common")
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)
    result = client.infer(instances)
    release_client(client)
    check_result(result, y_data_list)


@serving_test
def test_master_worker_client_alone_multi_times_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    master.start_master_server(master_port=7600)
    master.start_grpc_server("0.0.0.0", 5500)
    worker.start_servable(base.servable_dir, base.servable_name, master_port=7600, worker_port=6600)
    # Client, use with avoid affecting the next use case
    client = Client("localhost", 5500, base.servable_name, "add_common")
    for instance_count in range(1, 5):
        instances, y_data_list = create_multi_instances_fp32(instance_count)
        result = client.infer(instances)
        check_result(result, y_data_list)
    release_client(client)


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
    release_client(client)  # avoid affecting the next use case

    print(result)
    check_result(result, y_data_list)


@serving_test
def test_master_worker_client_async_multi_times_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    worker.start_servable_in_master(base.servable_dir, base.servable_name, 0)
    master.start_grpc_server("0.0.0.0", 5500)
    # Client, use with avoid affecting the next use case
    client = Client("localhost", 5500, base.servable_name, "add_common")
    for instance_count in range(1, 5):
        instances, y_data_list = create_multi_instances_fp32(instance_count)
        result_future = client.infer_async(instances)
        result = result_future.result()
        check_result(result, y_data_list)
    release_client(client)


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


@serving_test
def test_master_worker_client_alone_repeat_master_and_woker_port_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    master.start_master_server(master_port=7600)
    master.start_grpc_server("0.0.0.0", 5500)
    try:
        worker.start_servable(base.servable_dir, base.servable_name, master_port=7600, worker_port=7600)
        assert False
    except RuntimeError as e:
        assert "Serving Error: Worker gRPC server start failed, create server failed, address" in str(e)


@serving_test
def test_master_worker_client_alone_repeat_grpc_and_worker_port_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    master.start_master_server(master_port=7600)
    master.start_grpc_server("0.0.0.0", 5500)
    try:
        worker.start_servable(base.servable_dir, base.servable_name, master_port=7600, worker_port=5500)
        assert False
    except RuntimeError as e:
        assert "Serving Error: Worker gRPC server start failed, create server failed, address" in str(e)


@serving_test
def test_master_worker_client_alone_repeat_grpc_and_master_port_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    master.start_master_server(master_port=7600)
    try:
        master.start_grpc_server("0.0.0.0", 7600)
        assert False
    except RuntimeError as e:
        assert "Serving Error: Serving gRPC server start failed, create server failed, address" in str(e)


@serving_test
def test_master_worker_client_alone_repeat_grpc_and_master_port2_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    master.start_grpc_server("0.0.0.0", 7600)
    try:
        master.start_master_server(master_port=7600)
        assert False
    except RuntimeError as e:
        assert "Serving Error: Master server start failed, create server failed, address" in str(e)


@serving_test
def test_master_worker_client_alone_repeat_grpc_and_restful_port_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    master.start_grpc_server("0.0.0.0", 7600)
    try:
        master.start_restful_server("0.0.0.0", 7600)
        assert False
    except RuntimeError as e:
        assert "Serving Error: RESTful server start failed, create http listener failed, port" in str(e)


@serving_test
def test_master_worker_client_alone_repeat_grpc_and_restful_port2_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    master.start_restful_server("0.0.0.0", 7600)
    try:
        master.start_grpc_server("0.0.0.0", 7600)
        assert False
    except RuntimeError as e:
        assert "Serving Error: Serving gRPC server start failed, create server failed, address" in str(e)


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
    client = Client("localhost", 5500, base.servable_name, "add_common")
    result = client.infer(instances)
    release_client(client)

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
    client = Client("localhost", 5500, base.servable_name, "add_cast")
    result = client.infer(instances)
    release_client(client)

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
    client = Client("localhost", 5500, base.servable_name, "add_cast")
    result = client.infer(instances)
    release_client(client)

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

    client = Client("localhost", 5500, base.servable_name, "add_cast")
    result = client.infer(instances)
    release_client(client)
    assert result[0]["text"] == "ABC123"
    assert result[1]["text"] == "DEF456"
    assert result[2]["text"] == "HIJ789"


@serving_test
def test_master_worker_client_bytes_input_output_success():
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
        instance["label"] = str.encode(list_str[i])

    client = Client("localhost", 5500, base.servable_name, "add_cast")
    result = client.infer(instances)
    release_client(client)
    assert bytes.decode(result[0]["text"]) == "ABC123"
    assert bytes.decode(result[1]["text"]) == "DEF456"
    assert bytes.decode(result[2]["text"]) == "HIJ789"


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

    client = Client("localhost", 5500, base.servable_name, "add_cast")
    result = client.infer(instances)
    release_client(client)
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

    client = Client("localhost", 5500, base.servable_name, "add_cast")
    result = client.infer(instances)
    release_client(client)
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

    client = Client("localhost", 5500, base.servable_name, "add_cast")
    result = client.infer(instances)
    release_client(client)
    assert result[0]["value"] == 1
    assert result[1]["value"] == (2.2 + 1)
    assert result[2]["value"] == (4.4 + 1)


@serving_test
def test_master_worker_client_preprocess_update_numpy_success():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def preprocess(x3):
    x3[0] = 123
    return x3    
    
def postprocess(x3, x4):
    return x3 + 1, x4 + 2

@register.register_method(output_names=["x3", "x4"])
def add_cast(x1, x2, x3):
    x4 = register.call_preprocess(preprocess, x3) # [123, 1, 1], expect x3 is x4, same as python function call
    y = register.call_servable(x1, x2)    
    x3, x4 = register.call_postprocess(postprocess, x3, x4)
    return x3, x4 
"""
    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_grpc_server("0.0.0.0", 5500)
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    for instance in instances:
        instance["x3"] = np.ones([3])

    # Client, use with avoid affecting the next use case
    client = Client("localhost", 5500, base.servable_name, "add_cast")
    result = client.infer(instances)
    release_client(client)
    x3 = np.array([123, 1, 1]) + 1
    x4 = np.array([123, 1, 1]) + 2

    assert (result[0]["x3"] == x3).all()
    assert (result[1]["x3"] == x3).all()
    assert (result[2]["x3"] == x3).all()
    assert (result[0]["x4"] == x4).all()
    assert (result[1]["x4"] == x4).all()
    assert (result[2]["x4"] == x4).all()
