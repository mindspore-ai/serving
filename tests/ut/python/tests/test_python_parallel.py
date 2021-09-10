# Copyright 2021 Huawei Technologies Co., Ltd
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

import os

import numpy as np

from common import ServingTestBase
from common import serving_test, create_client
from mindspore_serving import server


def start_serving_server(servable_content, model_file="tensor_add.mindir", parallel_number=0, device_ids=0):
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content, model_file=model_file)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=device_ids,
                                                      num_parallel_workers=parallel_number, version_number=1))
    server.start_grpc_server("0.0.0.0:5500")
    return base


def is_float_equal(left, right):
    return (np.abs(left - right) < 0.00001).all()


def check_infer_log(servable_name, version, device_id, extra_id):
    if device_id is not None:
        log_file = f"serving_logs/log_{servable_name}_device{device_id}_version{version}.log"
    else:
        log_file = f"serving_logs/log_{servable_name}_extra{extra_id}_version{version}.log"
    if not os.path.isfile(log_file):
        print(f"Not found log file {log_file}", flush=True)
        return False
    with open(log_file) as fp:
        text = fp.read()
    if "WorkerRequestHandle Time Cost" not in text:
        print(f"Not found log 'WorkerRequestHandle Time Cost' in log file {log_file}", flush=True)
        return False
    print(f"Found log 'WorkerRequestHandle Time Cost' in log file {log_file}", flush=True)
    return True


@serving_test
def test_python_parallel_without_model_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def function_test(x1, x2):
    y = x1+x2
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(function_test, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, parallel_number=2, device_ids=0)
    # Client
    ys = []
    instances = []
    for i in range(20):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * (i + 1)
        instances.append({"x1": x1, "x2": x2})
        ys.append(x1 + x2)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    for i in range(len(instances)):
        assert is_float_equal(result[i]["y"], ys[i])
    assert check_infer_log(base.servable_name, base.version_number, device_id=0, extra_id=None)
    assert check_infer_log(base.servable_name, base.version_number, device_id=None, extra_id=0)


@serving_test
def test_python_parallel_with_model_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def function_test(x1, x2):
    return x1+1, x2+1

def function_test2(y):
    return y + 1

@register.register_method(output_names="y")
def predict(x1, x2):
    x1, x2 = register.add_stage(function_test, x1, x2, outputs_count=2)
    y = register.add_stage(model, x1, x2, outputs_count=1)
    y = register.add_stage(function_test2, y, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, parallel_number=2, device_ids=0)
    # Client
    ys = []
    instances = []
    for i in range(20):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * (i + 1)
        instances.append({"x1": x1, "x2": x2})
        ys.append(x1 + x2 + 3)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    for i in range(len(instances)):
        assert is_float_equal(result[i]["y"], ys[i])
    assert check_infer_log(base.servable_name, base.version_number, device_id=0, extra_id=None)
    assert check_infer_log(base.servable_name, base.version_number, device_id=None, extra_id=0)


@serving_test
def test_python_parallel_with_call_model_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def function_call_model(x1, x2):
    return model.call(x1, x2)

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4):
    y1 = register.add_stage(function_call_model, x1, x2, outputs_count=1)
    y2 = register.add_stage(model, x3, x4, outputs_count=1)
    y = register.add_stage(function_call_model, y1, y2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, parallel_number=2, device_ids=0)
    # Client
    ys = []
    instances = []
    for i in range(20):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * (i + 1)
        x3 = np.array([[3.1, 4.2], [5.3, 6.4]], np.float32) * (i + 1)
        x4 = np.array([[0.5, 9.6], [8.7, 7.8]], np.float32) * (i + 1)
        instances.append({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
        y = (x1 + x2) + (x3 + x4)
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    for i in range(len(instances)):
        assert is_float_equal(result[i]["y"], ys[i])
    assert check_infer_log(base.servable_name, base.version_number, device_id=0, extra_id=None)
    assert check_infer_log(base.servable_name, base.version_number, device_id=None, extra_id=0)


@serving_test
def test_python_parallel_with_call_model_multi_process_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def function_call_model(x1, x2):
    return model.call(x1, x2)

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4):
    y1 = register.add_stage(function_call_model, x1, x2, outputs_count=1)
    y2 = register.add_stage(model, x3, x4, outputs_count=1)
    y = register.add_stage(function_call_model, y1, y2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, parallel_number=4, device_ids=(0, 1))
    # Client
    ys = []
    instances = []
    for i in range(20):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * (i + 1)
        x3 = np.array([[3.1, 4.2], [5.3, 6.4]], np.float32) * (i + 1)
        x4 = np.array([[0.5, 9.6], [8.7, 7.8]], np.float32) * (i + 1)
        instances.append({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
        y = (x1 + x2) + (x3 + x4)
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    for i in range(len(instances)):
        assert is_float_equal(result[i]["y"], ys[i])
    assert check_infer_log(base.servable_name, base.version_number, device_id=0, extra_id=None)
    assert check_infer_log(base.servable_name, base.version_number, device_id=1, extra_id=None)
    assert check_infer_log(base.servable_name, base.version_number, device_id=None, extra_id=0)
    assert check_infer_log(base.servable_name, base.version_number, device_id=None, extra_id=1)


@serving_test
def test_python_parallel_with_call_model_with_batch_size_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)

def function_call_model(x1, x2):
    return model.call(x1, x2)

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4):
    y1 = register.add_stage(function_call_model, x1, x2, outputs_count=1)
    y2 = register.add_stage(model, x3, x4, outputs_count=1)
    y = register.add_stage(function_call_model, y1, y2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, parallel_number=2, device_ids=0)
    # Client
    ys = []
    instances = []
    for i in range(20):
        x1 = np.array([[3.3, 4.4]], np.float32) * (i + 1)
        x2 = np.array([[7.7, 8.8]], np.float32) * (i + 1)
        x3 = np.array([[5.3, 6.4]], np.float32) * (i + 1)
        x4 = np.array([[8.7, 7.8]], np.float32) * (i + 1)
        instances.append({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
        y = (x1 + x2) + (x3 + x4)
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    for i in range(len(instances)):
        assert is_float_equal(result[i]["y"], ys[i])
    assert check_infer_log(base.servable_name, base.version_number, device_id=0, extra_id=None)
    assert check_infer_log(base.servable_name, base.version_number, device_id=None, extra_id=0)


@serving_test
def test_python_parallel_multi_models_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

add_model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
sub_model = register.declare_model(model_file="tensor_sub.mindir", model_format="MindIR", with_batch_dim=False)

def function_call_model(x1, x2):
    return add_model.call(x1, x2)

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4):
    y1 = register.add_stage(add_model, x1, x2, outputs_count=1)
    y2 = register.add_stage(sub_model, x3, x4, outputs_count=1)
    y = register.add_stage(function_call_model, y1, y2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, parallel_number=2, device_ids=0,
                                model_file=["tensor_add.mindir", "tensor_sub.mindir"])
    # Client
    ys = []
    instances = []
    for i in range(20):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * (i + 1)
        x3 = np.array([[3.1, 4.2], [5.3, 6.4]], np.float32) * (i + 1)
        x4 = np.array([[0.5, 9.6], [8.7, 7.8]], np.float32) * (i + 1)
        instances.append({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
        y = (x1 + x2) + (x3 - x4)
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    for i in range(len(instances)):
        assert is_float_equal(result[i]["y"], ys[i])
    assert check_infer_log(base.servable_name, base.version_number, device_id=0, extra_id=None)
    assert check_infer_log(base.servable_name, base.version_number, device_id=None, extra_id=0)


@serving_test
def test_python_parallel_multi_models_diff_input_output_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

add_model = register.declare_model(model_file="tensor_add_2_3.mindir", model_format="MindIR", with_batch_dim=False)
sub_model = register.declare_model(model_file="tensor_sub_3_2.mindir", model_format="MindIR", with_batch_dim=False)

def function_call_model(x1, x2):
    return x1 + x2

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4, x5):
    _,y1,_ = register.add_stage(add_model, x1, x2, outputs_count=3) # 2 input, 3 output
    _, y2 = register.add_stage(sub_model, x3, x4, x5, outputs_count=2) # 3 input, 2 output
    y = register.add_stage(function_call_model, y1, y2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, parallel_number=2, device_ids=0,
                                model_file=["tensor_add_2_3.mindir", "tensor_sub_3_2.mindir"])
    # Client
    ys = []
    instances = []
    for i in range(20):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * (i + 1)
        x3 = np.array([[3.1, 4.2], [5.3, 6.4]], np.float32) * (i + 1)
        x4 = np.array([[0.5, 9.6], [8.7, 7.8]], np.float32) * (i + 1)
        x5 = np.array([[0.2, 9.5], [8.2, 7.1]], np.float32) * (i + 1)
        instances.append({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5})
        y = (x1 + x2 + 1) + (x3 - x4 - x5 + 1)
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    for i in range(len(instances)):
        assert is_float_equal(result[i]["y"], ys[i])
    assert check_infer_log(base.servable_name, base.version_number, device_id=0, extra_id=None)
    assert check_infer_log(base.servable_name, base.version_number, device_id=None, extra_id=0)
