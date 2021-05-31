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

from common import ServingTestBase, serving_test, create_client
from common import servable_config_import, servable_config_declare_servable, servable_config_preprocess_cast
from common import servable_config_method_add_common, servable_config_method_add_cast
from mindspore_serving import server


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
        assert "Serving Error: RESTful server start failed, create http listener failed, address 0.0.0.0:7600" in str(e)


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
    if "error" in result:
        assert "Preprocess Failed" in str(result["error"])
    else:
        assert len(result) == 3
        assert "Preprocess Failed" in str(result[0]["error"])


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
    if "error" in result:
        assert "Postprocess Failed" in str(result["error"])
    else:
        assert len(result) == 3
        assert "Postprocess Failed" in str(result[0]["error"])


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
def test_servable_postprocess_result_count_less():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += r"""
register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)
"""
    servable_content += r"""
def postprocess(instances):
    count = len(instances)
    for i in range(count -1):
        yield i
    
@register.register_method(output_names=["y"])
def add_common(x1, x2):
    y = register.call_servable(x1, x2)
    y = register.call_postprocess_pipeline(postprocess, y)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    # Client
    instance_count = 2

    instances = []
    y_data_list = []
    for i in range(instance_count):
        x1 = np.asarray([[1.1], [3.3]]).astype(np.float32) * (i + 1)
        x2 = np.asarray([[5.5], [7.7]]).astype(np.float32) * (i + 1)
        y_data_list.append(x1 + x2)
        instances.append({"x1": x1, "x2": x2})

    client = create_client("localhost:5500", base.servable_name, "add_common")
    result = client.infer(instances)
    print(result)
    assert len(result) == instance_count
    assert "Postprocess Failed" in str(result[1]["error"]) or "Servable stopped" in str(result[1]["error"])


@serving_test
def test_servable_postprocess_result_count_more():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += r"""
register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)
"""
    servable_content += r"""
def postprocess(instances):
    count = len(instances)
    for i in range(count + 1):
        yield i
    
@register.register_method(output_names=["y"])
def add_common(x1, x2):
    y = register.call_servable(x1, x2)
    y = register.call_postprocess_pipeline(postprocess, y)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    # Client
    instance_count = 2

    instances = []
    y_data_list = []
    for i in range(instance_count):
        x1 = np.asarray([[1.1], [3.3]]).astype(np.float32) * (i + 1)
        x2 = np.asarray([[5.5], [7.7]]).astype(np.float32) * (i + 1)
        y_data_list.append(x1 + x2)
        instances.append({"x1": x1, "x2": x2})

    client = create_client("localhost:5500", base.servable_name, "add_common")
    result = client.infer(instances)
    print(result)
    assert len(result) == instance_count
    assert result[0]["y"] == 0
    assert result[1]["y"] == 1


@serving_test
def test_servable_postprocess_result_type_invalid():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += r"""
register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)
"""
    servable_content += r"""
def postprocess(instances):
    count = len(instances)
    for i in range(count):
        yield np.int8
    
@register.register_method(output_names=["y"])
def add_common(x1, x2):
    y = register.call_servable(x1, x2)
    y = register.call_postprocess_pipeline(postprocess, y)
    return y
"""
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
        instances.append({"x1": x1, "x2": x2})

    client = create_client("localhost:5500", base.servable_name, "add_common")
    result = client.infer(instances)
    print(result)
    if "error" in result:
        assert "Postprocess Failed" in str(result["error"])
    else:
        assert len(result) == instance_count
        assert "Postprocess Failed" in str(result[0]["error"]) or "Servable stopped" in str(result[0]["error"])
        assert "Postprocess Failed" in str(result[1]["error"]) or "Servable stopped" in str(result[1]["error"])


@serving_test
def test_servable_postprocess_get_result_exception():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += r"""
register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)
"""
    servable_content += r"""
def postprocess(instances):
    count = len(instances)
    for i in range(count):
        if i == 0:
           yield i
        raise RuntimeError("RuntimeError")
    
@register.register_method(output_names=["y"])
def add_common(x1, x2):
    y = register.call_servable(x1, x2)
    y = register.call_postprocess_pipeline(postprocess, y)
    return y
"""
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
        instances.append({"x1": x1, "x2": x2})

    client = create_client("localhost:5500", base.servable_name, "add_common")
    result = client.infer(instances)
    print(result)
    assert len(result) == instance_count

    assert result[0]["y"] == 0
    assert "Postprocess Failed" in str(result[1]["error"])


@serving_test
def test_servable_preprocess_result_count_less():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += r"""
register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)
"""
    servable_content += r"""
def preprocess(instances):
    count = len(instances)
    for i in range(count-1):
        yield i
    
@register.register_method(output_names=["y"])
def add_common(x1, x2):
    x3 = register.call_preprocess_pipeline(preprocess, x1)
    y = register.call_servable(x1, x2)
    return x3
"""
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
        instances.append({"x1": x1, "x2": x2})

    client = create_client("localhost:5500", base.servable_name, "add_common")
    result = client.infer(instances)
    print(result)
    assert len(result) == instance_count

    assert "Preprocess Failed" in str(result[2]["error"]) or "Servable stopped" in str(result[2]["error"])


@serving_test
def test_servable_preprocess_result_count_more():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += r"""
register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)
"""
    servable_content += r"""
def preprocess(instances):
    count = len(instances)
    for i in range(count+1):
        yield i
    
@register.register_method(output_names=["y"])
def add_common(x1, x2):
    x3 = register.call_preprocess_pipeline(preprocess, x1)
    y = register.call_servable(x1, x2)
    return x3
"""
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
        instances.append({"x1": x1, "x2": x2})

    client = create_client("localhost:5500", base.servable_name, "add_common")
    result = client.infer(instances)
    print(result)
    assert len(result) == instance_count


@serving_test
def test_servable_preprocess_result_type_invalid():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += r"""
register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)
"""
    servable_content += r"""
def preprocess(instances):
    count = len(instances)
    for i in range(count):
        if i == 0:
            yield i
            continue
        yield np.int8
    
@register.register_method(output_names=["y"])
def add_common(x1, x2):
    x3 = register.call_preprocess_pipeline(preprocess, x1)
    y = register.call_servable(x1, x2)
    return x3
"""
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
        instances.append({"x1": x1, "x2": x2})

    client = create_client("localhost:5500", base.servable_name, "add_common")
    result = client.infer(instances)
    print(result)
    if "error" in result:
        assert "Preprocess Failed" in str(result["error"])
    else:
        assert len(result) == instance_count
        assert "Preprocess Failed" in str(result[0]["error"]) or "Servable stopped" in str(result[0]["error"])
        assert "Preprocess Failed" in str(result[1]["error"]) or "Servable stopped" in str(result[1]["error"])
        assert "Preprocess Failed" in str(result[2]["error"]) or "Servable stopped" in str(result[2]["error"])


@serving_test
def test_servable_preprocess_get_result_exception():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += r"""
register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)
"""
    servable_content += r"""
def preprocess(instances):
    count = len(instances)
    for i in range(count):
        if i == 0:
           yield i
        raise RuntimeError("RuntimeError")
    
@register.register_method(output_names=["y"])
def add_common(x1, x2):
    x3 = register.call_preprocess_pipeline(preprocess, x1)
    y = register.call_servable(x1, x2)
    return x3
"""
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
        instances.append({"x1": x1, "x2": x2})

    client = create_client("localhost:5500", base.servable_name, "add_common")
    result = client.infer(instances)
    print(result)
    assert len(result) == instance_count

    assert result[0]["y"] == 0
    assert "Preprocess Failed" in str(result[1]["error"]) or "Servable stopped" in str(result[1]["error"])
    assert result[2]["y"] == 0


@serving_test
def test_servable_worker_with_master_preprocess_runtime_error():
    # fail returned from Preprocess
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
index = 0
def preprocess(instances):
    count = len(instances)
    global index
    for i in range(count):
        ret = index
        index += 1
        if ret == 0:
            raise RuntimeError("runtime error")
        yield ret
    
@register.register_method(output_names=["y"])
def add_common(x1, x2):
    x3 = register.call_preprocess_pipeline(preprocess, x1)
    y = register.call_servable(x1, x2)
    return x3
"""
    base.init_servable_with_servable_config(1, servable_content)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    # Client
    instance_count = 3

    instances = []
    y_data_list = []
    for i in range(instance_count):
        x1 = np.asarray([[1.1, 2.2], [3.3, 4.4]]).astype(np.float32) * (i + 1)
        x2 = np.asarray([[5.5, 6.6], [7.7, 8.8]]).astype(np.float32) * (i + 1)
        y_data_list.append(x1 + x2)
        instances.append({"x1": x1, "x2": x2})

    client = create_client("localhost:5500", base.servable_name, "add_common")
    result = client.infer(instances)
    print(result)
    assert "Preprocess Failed" in result[0]["error"]
    assert result[1]["y"] == 1
    assert result[2]["y"] == 2


@serving_test
def test_servable_worker_with_master_predict_check_failed():
    # fail returned from Predict
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_common(x1, x2):
    y = register.call_servable(x1, x2)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    # Client
    instance_count = 3

    instances = []
    y_data_list = []
    for i in range(instance_count):
        if i == 0:
            x1 = np.asarray([[1.1], [3.3]]).astype(np.float32) * (i + 1)
        else:
            x1 = np.asarray([[1.1, 2.2], [3.3, 4.4]]).astype(np.float32) * (i + 1)
        x2 = np.asarray([[5.5, 6.6], [7.7, 8.8]]).astype(np.float32) * (i + 1)
        y_data_list.append(x1 + x2)
        instances.append({"x1": x1, "x2": x2})

    client = create_client("localhost:5500", base.servable_name, "add_common")
    result = client.infer(instances)
    print(result)
    assert "Given model input 0 size 8 not match the size 16 defined in model" in result[0]["error"]
    assert "y" in result[1]
    assert "y" in result[2]


@serving_test
def test_servable_worker_with_master_postprocess_runtime_error():
    # fail returned from Preprocess
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
index = 0
def postprocess(y):
    global index
    ret = index
    index += 1
    if ret == 0:
        raise RuntimeError("runtime error")
    return ret
    
@register.register_method(output_names=["y"])
def add_common(x1, x2):
    y = register.call_servable(x1, x2)
    y = register.call_postprocess(postprocess, y)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_grpc_server("0.0.0.0:5500")
    # Client
    instance_count = 3

    instances = []
    y_data_list = []
    for i in range(instance_count):
        x1 = np.asarray([[1.1, 2.2], [3.3, 4.4]]).astype(np.float32) * (i + 1)
        x2 = np.asarray([[5.5, 6.6], [7.7, 8.8]]).astype(np.float32) * (i + 1)
        y_data_list.append(x1 + x2)
        instances.append({"x1": x1, "x2": x2})

    client = create_client("localhost:5500", base.servable_name, "add_common")
    result = client.infer(instances)
    print(result)
    assert "Postprocess Failed" in result[0]["error"]
    assert result[1]["y"] == 1
    assert result[2]["y"] == 2


@serving_test
def test_servable_worker_with_master_input_param_less():
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
def test_servable_worker_with_master_servable_not_available():
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
def test_servable_worker_with_master_max_request_count():
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
