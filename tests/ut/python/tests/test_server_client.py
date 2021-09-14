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

import os
import time
import signal
import psutil

import numpy as np

from common import ServingTestBase, serving_test, create_client, generate_cert
from common import servable_config_import, servable_config_declare_servable, servable_config_preprocess_cast
from common import servable_config_method_add_common, servable_config_method_add_cast
from common import start_serving_server
from mindspore_serving import server
from mindspore_serving.client import SSLConfig


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


def is_float_equal(left, right):
    return (np.abs(left - right) < 0.00001).all()


@serving_test
def test_grpc_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    # Client
    client = create_client("localhost:5500", base.servable_name, "add_common")
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)
    result = client.infer(instances)

    print(result)
    check_result(result, y_data_list)


@serving_test
def test_grpc_multi_times_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    # Client
    client = create_client("localhost:5500", base.servable_name, "add_common")
    for instance_count in range(1, 5):
        instances, y_data_list = create_multi_instances_fp32(instance_count)
        result = client.infer(instances)
        check_result(result, y_data_list)


@serving_test
def test_grpc_async_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    # Client
    client = create_client("localhost:5500", base.servable_name, "add_common")
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)
    result_future = client.infer_async(instances)
    result = result_future.result()

    print(result)
    check_result(result, y_data_list)


@serving_test
def test_grpc_async_multi_times_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    # Client, use with avoid affecting the next use case
    client = create_client("localhost:5500", base.servable_name, "add_common")
    for instance_count in range(1, 5):
        instances, y_data_list = create_multi_instances_fp32(instance_count)
        result_future = client.infer_async(instances)
        result = result_future.result()
        check_result(result, y_data_list)


@serving_test
def test_grpc_start_grpc_twice_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    try:
        server.start_grpc_server("0.0.0.0:4500")
        assert False
    except RuntimeError as e:
        assert "Serving Error: Serving gRPC server is already running" in str(e)


@serving_test
def test_grpc_start_restful_server_twice_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_restful_server("0.0.0.0:5500")
    try:
        server.start_restful_server("0.0.0.0:4500")
        assert False
    except RuntimeError as e:
        assert "Serving Error: RESTful server is already running" in str(e)


@serving_test
def test_grpc_alone_repeat_grpc_and_restful_port_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_grpc_server("0.0.0.0:7600")
    try:
        server.start_restful_server("0.0.0.0:7600")
        assert False
    except RuntimeError as e:
        assert "Serving Error: RESTful server start failed, bind to the socket address 0.0.0.0:7600 failed" in str(e)


@serving_test
def test_grpc_alone_repeat_grpc_and_restful_port2_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_restful_server("0.0.0.0:7600")
    try:
        server.start_grpc_server("0.0.0.0:7600")
        assert False
    except RuntimeError as e:
        assert "Serving Error: Serving gRPC server start failed, create server failed, address" in str(e)


@serving_test
def test_grpc_servable_content_success():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += servable_config_method_add_common
    servable_content += servable_config_method_add_cast

    base.init_servable_with_servable_config(1, servable_content)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    # Client
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)
    client = create_client("localhost:5500", base.servable_name, "add_common")
    result = client.infer(instances)

    print(result)
    check_result(result, y_data_list)


@serving_test
def test_grpc_one_way_auth_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    generate_cert()
    ssl_config = server.SSLConfig(certificate="server.crt", private_key="server.key", custom_ca="ca.crt",
                                  verify_client=False)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500", ssl_config=ssl_config)

    ssl_config = SSLConfig(custom_ca="ca.crt")
    client = create_client("0.0.0.0:5500", base.servable_name, "add_common", ssl_config=ssl_config)
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)
    result = client.infer(instances)

    print(result)
    check_result(result, y_data_list)


@serving_test
def test_grpc_mutual_auth_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    generate_cert(server_ip="127.0.0.1")
    ssl_config = server.SSLConfig(certificate="server.crt", private_key="server.key", custom_ca="ca.crt",
                                  verify_client=True)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("127.0.0.1:5500", ssl_config=ssl_config)

    ssl_config = SSLConfig(certificate="client.crt", private_key="client.key", custom_ca="ca.crt")
    client = create_client("127.0.0.1:5500", base.servable_name, "add_common", ssl_config=ssl_config)
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)
    result = client.infer(instances)

    print(result)
    check_result(result, y_data_list)


@serving_test
def test_grpc_client_auth_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    generate_cert(server_ip="127.0.0.1")
    ssl_config = server.SSLConfig(certificate="server.crt", private_key="server.key", custom_ca="ca.crt",
                                  verify_client=False)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("127.0.0.1:5500", ssl_config=ssl_config)

    ssl_config = SSLConfig(custom_ca="client.crt")
    client = create_client("127.0.0.1:5500", base.servable_name, "add_common", ssl_config=ssl_config)
    instance_count = 3
    data = create_multi_instances_fp32(instance_count)
    result = client.infer(data[0])

    print(result)
    assert "unavailable" in result["error"]


@serving_test
def test_grpc_missing_cert_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    generate_cert(server_ip="127.0.0.1")
    ssl_config = server.SSLConfig(certificate="server.crt", private_key="server.key", custom_ca="ca.crt",
                                  verify_client=True)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("127.0.0.1:5500", ssl_config=ssl_config)

    ssl_config = SSLConfig(custom_ca="ca.crt")
    client = create_client("127.0.0.1:5500", base.servable_name, "add_common", ssl_config=ssl_config)
    instance_count = 3
    data = create_multi_instances_fp32(instance_count)
    result = client.infer(data[0])

    print(result)
    assert "unavailable" in result["error"]


@serving_test
def test_grpc_unmatched_cert_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    generate_cert(server_ip="127.0.0.1")
    ssl_config = server.SSLConfig(certificate="server.crt", private_key="server.crt", custom_ca="ca.crt",
                                  verify_client=True)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    try:
        server.start_grpc_server("127.0.0.1:5500", ssl_config=ssl_config)
        assert False
    except RuntimeError as e:
        assert "Serving Error: Serving gRPC server start failed, create server failed, address" in str(e)


@serving_test
def test_grpc_preprocess_outputs_count_not_match_failed():
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
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    client = create_client("localhost:5500", base.servable_name, "add_cast")
    result = client.infer(instances)

    print(result)
    assert "Preprocess Failed" in str(result["error"]) or "servable is not available" in str(result["error"])


@serving_test
def test_grpc_postprocess_outputs_count_not_match_failed():
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
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    client = create_client("localhost:5500", base.servable_name, "add_cast")
    result = client.infer(instances)

    print(result)
    assert "Postprocess Failed" in str(result["error"]) or "servable is not available" in str(result["error"])


@serving_test
def test_grpc_preprocess_update_numpy_success():
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
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    # Client
    instances = [{}, {}, {}]
    for instance in instances:
        instance["x1"] = np.ones([2, 2]).astype(np.float32)
        instance["x2"] = np.ones([2, 2]).astype(np.float32)
        instance["x3"] = np.ones([3]).astype(np.int32)

    # Client, use with avoid affecting the next use case
    client = create_client("localhost:5500", base.servable_name, "add_cast")
    result = client.infer(instances)
    print(result)

    x3 = (np.array([123, 1, 1]) + 1).tolist()
    x4 = (np.array([123, 1, 1]) + 2).tolist()

    assert result[0]["x3"].tolist() == x3
    assert result[0]["x4"].tolist() == x4
    assert result[1]["x3"].tolist() == x3
    assert result[1]["x4"].tolist() == x4
    assert result[2]["x3"].tolist() == x3
    assert result[2]["x4"].tolist() == x4


@serving_test
def test_grpc_larger_than_server_receive_max_size():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500", max_msg_mb_size=1)  # 1MB
    # Client
    client = create_client("localhost:5500", base.servable_name, "add_common")
    instances = []
    # instance 1
    y_data_list = []
    x1 = np.ones([1024, 1024], np.float32)
    x2 = np.ones([1024, 1024], np.float32)
    y_data_list.append(x1 + x2)
    instances.append({"x1": x1, "x2": x2})
    result = client.infer(instances)  # more than 1MB msg

    print(result)
    assert "Grpc Error, (8, 'resource exhausted')" in str(result["error"])


@serving_test
def test_server_client_input_param_less():
    # fail returned from Worker::RunAsync
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_method_add_common
    base.init_servable_with_servable_config(1, servable_content)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    # Client
    instance_count = 3

    instances = []
    y_data_list = []
    for i in range(instance_count):
        x1 = np.asarray([[1.1], [3.3]]).astype(np.float32) * (i + 1)
        x2 = np.asarray([[5.5], [7.7]]).astype(np.float32) * (i + 1)
        y_data_list.append(x1 + x2)
        instances.append({"x3": x1, "x2": x2})

    client = create_client("localhost:5500", base.servable_name, "add_common")
    result = client.infer(instances)
    print(result)
    assert "Cannot find input x1 in instance input" in result["error"]


@serving_test
def test_server_client_servable_not_available():
    # fail returned from Worker::RunAsync
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_method_add_common
    base.init_servable_with_servable_config(1, servable_content)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    # Client
    instance_count = 3

    instances = []
    y_data_list = []
    for i in range(instance_count):
        x1 = np.asarray([[1.1], [3.3]]).astype(np.float32) * (i + 1)
        x2 = np.asarray([[5.5], [7.7]]).astype(np.float32) * (i + 1)
        y_data_list.append(x1 + x2)
        instances.append({"x3": x1, "x2": x2})

    client = create_client("localhost:5500", base.servable_name + "error", "add_common")
    result = client.infer(instances)
    print(result)
    assert "servable is not available" in result["error"]


@serving_test
def test_server_client_max_request_count():
    # fail returned from Worker::RunAsync
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
import time
def preprocess(x1, x2):
    time.sleep(1)    
    return x1, x2
    
@register.register_method(output_names=["y"])
def add_common(x1, x2):
    x1, x2 = register.call_preprocess(preprocess, x1, x2)
    y = register.call_servable(x1, x2)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    server.master.context.set_max_enqueued_requests(1)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    # Client
    x1 = np.asarray([[1.1, 2.2], [3.3, 4.4]]).astype(np.float32)
    x2 = np.asarray([[5.5, 6.6], [7.7, 8.8]]).astype(np.float32)
    instance = {"x1": x1, "x2": x2}

    client = create_client("localhost:5500", base.servable_name, "add_common")
    result_list = []
    for _ in range(2):
        result = client.infer_async(instance)
        result_list.append(result)

    result0 = result_list[0].result()
    result1 = result_list[1].result()
    print(result0)
    print(result1)
    assert "error" in result0 or "error" in result1
    if "error" in result0:
        assert "error" not in result1
        assert "Serving Error: enqueued requests count exceeds the limit 1" in result0["error"]
    else:
        assert "error" not in result0
        assert "Serving Error: enqueued requests count exceeds the limit 1" in result1["error"]


@serving_test
def test_server_client_one_model_stage_with_batch_dim_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    x1 = np.array([[3.3, 4.4]], np.float32)
    x2 = np.array([[7.7, 8.8]], np.float32)
    y = x1 + x2
    instances = [{"x1": x1, "x2": x2}] * 3

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()
    assert (result[1]["y"] == y).all()
    assert (result[2]["y"] == y).all()


@serving_test
def test_server_client_one_model_stage_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    y = x1 + x2
    instances = [{"x1": x1, "x2": x2}] * 3

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()
    assert (result[1]["y"] == y).all()
    assert (result[2]["y"] == y).all()


@serving_test
def test_server_client_with_batch_dim_data_size_invalid_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    x1 = np.array([[3.3, 4.4]], np.float32)
    x2 = np.array([[7.7, 8.8]], np.float32)
    y = x1 + x2
    instances = [{"x1": x1, "x2": x2}, {"x1": x1, "x2": x2}, {"x1": x1, "x2": x2}]
    instances[1]["x2"] = np.array([[7.7, 8.8, 9.9]], np.float32)
    print(instances)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()
    assert "Given model input 1 size 12 not match the size 8 defined in model" in result[1]["error"]
    assert (result[2]["y"] == y).all()


@serving_test
def test_server_client_with_batch_dim_data_type_invalid_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    x1 = np.array([[3.3, 4.4]], np.float32)
    x2 = np.array([[7.7, 8.8]], np.float32)
    y = x1 + x2
    instances = [{"x1": x1, "x2": x2}, {"x1": x1, "x2": x2}, {"x1": x1, "x2": x2}]
    instances[1]["x2"] = np.array([[7.7, 9.9]], np.int32)
    print(instances)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()
    assert "Given model input 1 data type kMSI_Int32 not match the data type kMSI_Float32 defined in model" in \
           result[1]["error"]
    assert (result[2]["y"] == y).all()


@serving_test
def test_server_client_data_size_invalid_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    y = x1 + x2
    instances = [{"x1": x1, "x2": x2}, {"x1": x1, "x2": x2}, {"x1": x1, "x2": x2}]
    instances[1]["x2"] = np.array([[5.5, 6.6, 8.8], [7.7, 8.8, 9.9]], np.float32)
    print(instances)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()
    assert "Given model input 1 size 24 not match the size 16 defined in model" in result[1]["error"]
    assert (result[2]["y"] == y).all()


@serving_test
def test_server_client_data_type_invalid_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    y = x1 + x2
    instances = [{"x1": x1, "x2": x2}, {"x1": x1, "x2": x2}, {"x1": x1, "x2": x2}]
    instances[1]["x2"] = np.array([[5.5, 6.8], [7.7, 9.9]], np.int32)
    print(instances)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()
    assert "Given model input 1 data type kMSI_Int32 not match the data type kMSI_Float32 defined in model" in \
           result[1]["error"]
    assert (result[2]["y"] == y).all()


@serving_test
def test_server_client_two_model_stage_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    y = register.add_stage(tensor_add, y, x3, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x3 = np.array([[8.5, 7.3], [6.2, 5.4]], np.float32) * 1.1 * (i + 1)
        y = x1 + x2 + x3
        instances.append({"x1": x1, "x2": x2, "x3": x3})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y"], ys[0])
    assert is_float_equal(result[1]["y"], ys[1])
    assert is_float_equal(result[2]["y"], ys[2])


@serving_test
def test_server_client_one_model_stage_with_function_front_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def add_test(x1, x2):
    return x1 + x2 + 1

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(add_test, x1, x2, outputs_count=1)
    y = register.add_stage(tensor_add, y, x3, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x3 = np.array([[8.5, 7.3], [6.2, 5.4]], np.float32) * 1.1 * (i + 1)
        y = x1 + x2 + x3 + 1
        instances.append({"x1": x1, "x2": x2, "x3": x3})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y"], ys[0])
    assert is_float_equal(result[1]["y"], ys[1])
    assert is_float_equal(result[2]["y"], ys[2])


@serving_test
def test_server_client_one_model_stage_with_function_tail_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def add_test(x1, x2):
    return x1 + x2 + 1

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    y = register.add_stage(add_test, y, x3, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x3 = np.array([[8.5, 7.3], [6.2, 5.4]], np.float32) * 1.1 * (i + 1)
        y = x1 + x2 + x3 + 1
        instances.append({"x1": x1, "x2": x2, "x3": x3})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y"], ys[0])
    assert is_float_equal(result[1]["y"], ys[1])
    assert is_float_equal(result[2]["y"], ys[2])


@serving_test
def test_server_client_one_model_stage_with_function_front_and_tail_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def add_test(x1, x2):
    return x1 + x2 + 1

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4):
    y = register.add_stage(add_test, x1, x2, outputs_count=1)
    y = register.add_stage(tensor_add, y, x3, outputs_count=1)
    y = register.add_stage(add_test, y, x4, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x3 = np.array([[8.5, 7.3], [6.2, 5.4]], np.float32) * 1.1 * (i + 1)
        x4 = np.array([[3.5, 4.3], [5.2, 6.4]], np.float32) * 1.1 * (i + 1)
        y = x1 + x2 + x3 + x4 + 2
        instances.append({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y"], ys[0])
    assert is_float_equal(result[1]["y"], ys[1])
    assert is_float_equal(result[2]["y"], ys[2])


@serving_test
def test_server_client_one_model_stage_with_function_front_and_tail_double_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def add_test(x1, x2):
    return x1 + x2 + 1

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4, x5, x6):
    y = register.add_stage(add_test, x1, x2, outputs_count=1)
    y = register.add_stage(add_test, y, x3, outputs_count=1)
    y = register.add_stage(tensor_add, y, x4, outputs_count=1)
    y = register.add_stage(add_test, y, x5, outputs_count=1)
    y = register.add_stage(add_test, y, x6, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x3 = np.array([[8.5, 7.3], [6.2, 5.4]], np.float32) * 1.1 * (i + 1)
        x4 = np.array([[3.5, 4.3], [5.2, 6.4]], np.float32) * 1.1 * (i + 1)
        x5 = np.array([[1.5, 2.3], [3.2, 4.4]], np.float32) * 1.1 * (i + 1)
        x6 = np.array([[5.5, 6.3], [7.2, 8.4]], np.float32) * 1.1 * (i + 1)
        y = x1 + x2 + x3 + x4 + x5 + x6 + 4
        instances.append({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "x6": x6})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y"], ys[0])
    assert is_float_equal(result[1]["y"], ys[1])
    assert is_float_equal(result[2]["y"], ys[2])


@serving_test
def test_server_client_two_model_stage_with_function_front_and_tail_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def add_test(x1, x2):
    return x1 + x2 + 1

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4, x5, x6):
    y = register.add_stage(add_test, x1, x2, outputs_count=1)
    y = register.add_stage(tensor_add, y, x3, outputs_count=1)
    y = register.add_stage(add_test, y, x4, outputs_count=1)
    y = register.add_stage(tensor_add, y, x5, outputs_count=1)
    y = register.add_stage(add_test, y, x6, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x3 = np.array([[8.5, 7.3], [6.2, 5.4]], np.float32) * 1.1 * (i + 1)
        x4 = np.array([[3.5, 4.3], [5.2, 6.4]], np.float32) * 1.1 * (i + 1)
        x5 = np.array([[1.5, 2.3], [3.2, 4.4]], np.float32) * 1.1 * (i + 1)
        x6 = np.array([[5.5, 6.3], [7.2, 8.4]], np.float32) * 1.1 * (i + 1)
        y = x1 + x2 + x3 + x4 + x5 + x6 + 3
        instances.append({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "x6": x6})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y"], ys[0])
    assert is_float_equal(result[1]["y"], ys[1])
    assert is_float_equal(result[2]["y"], ys[2])


@serving_test
def test_server_client_two_model_stage_with_function_front_and_tail_with_batch_dim_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)

def add_test(x1, x2):
    return x1 + x2 + 1

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4, x5, x6):
    y = register.add_stage(add_test, x1, x2, outputs_count=1)
    y = register.add_stage(tensor_add, y, x3, outputs_count=1)
    y = register.add_stage(add_test, y, x4, outputs_count=1)
    y = register.add_stage(tensor_add, y, x5, outputs_count=1)
    y = register.add_stage(add_test, y, x6, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x3 = np.array([[6.2, 5.4]], np.float32) * 1.1 * (i + 1)
        x4 = np.array([[5.2, 6.4]], np.float32) * 1.1 * (i + 1)
        x5 = np.array([[3.2, 4.4]], np.float32) * 1.1 * (i + 1)
        x6 = np.array([[7.2, 8.4]], np.float32) * 1.1 * (i + 1)
        y = x1 + x2 + x3 + x4 + x5 + x6 + 3
        instances.append({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "x6": x6})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y"], ys[0])
    assert is_float_equal(result[1]["y"], ys[1])
    assert is_float_equal(result[2]["y"], ys[2])


@serving_test
def test_server_client_worker_exit_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    # Client
    client = create_client("localhost:5500", base.servable_name, "add_common")
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)
    result = client.infer(instances)

    print(result)
    check_result(result, y_data_list)

    cur_process = psutil.Process(os.getpid())
    children = cur_process.children(recursive=False)
    for item in children:
        os.kill(item.pid, signal.SIGINT)
    time.sleep(2)
    result = client.infer(instances)
    print(result)
    assert "Grpc Error, (14, 'unavailable')" in result["error"]


@serving_test
def test_server_client_worker_kill_restart_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    # Client
    client = create_client("localhost:5500", base.servable_name, "add_common")
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)
    result = client.infer(instances)

    print(result)
    check_result(result, y_data_list)

    cur_process = psutil.Process(os.getpid())
    children = cur_process.children(recursive=False)
    for item in children:
        os.kill(item.pid, signal.SIGKILL)
    time.sleep(3)
    result = client.infer(instances)
    print(result)
    check_result(result, y_data_list)


@serving_test
def test_server_client_worker_kill_no_restart_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")

    cur_process = psutil.Process(os.getpid())
    children = cur_process.children(recursive=False)
    for item in children:
        os.kill(item.pid, signal.SIGKILL)
    time.sleep(3)

    # Client
    client = create_client("localhost:5500", base.servable_name, "add_common")
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    result = client.infer(instances)
    print(result)
    assert "Grpc Error, (14, 'unavailable')" in result["error"]
