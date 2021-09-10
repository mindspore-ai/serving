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

import numpy as np

from common import serving_test, create_client
from common import start_serving_server


def is_float_equal(left, right):
    return (np.abs(left - right) < 0.00001).all()


@serving_test
def test_multi_model_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
tensor_sub = register.declare_model(model_file="tensor_sub.mindir", model_format="MindIR", with_batch_dim=False)

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    y = register.add_stage(tensor_sub, y, x3, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=["tensor_add.mindir", "tensor_sub.mindir"])
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x3 = np.array([[8.5, 7.3], [6.2, 5.4]], np.float32) * 1.1 * (i + 1)
        y = x1 + x2 - x3
        instances.append({"x1": x1, "x2": x2, "x3": x3})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y"], ys[0])
    assert is_float_equal(result[1]["y"], ys[1])
    assert is_float_equal(result[2]["y"], ys[2])


@serving_test
def test_multi_model_2_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
tensor_sub = register.declare_model(model_file="tensor_sub.mindir", model_format="MindIR", with_batch_dim=False)

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4, x5):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    y = register.add_stage(tensor_sub, y, x3, outputs_count=1)
    y = register.add_stage(tensor_add, y, x4, outputs_count=1)
    y = register.add_stage(tensor_sub, y, x5, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=["tensor_add.mindir", "tensor_sub.mindir"])
    # Client
    instances = []
    ys = []
    for i in range(10):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x3 = np.array([[8.5, 7.3], [6.2, 5.4]], np.float32) * 1.1 * (i + 1)
        x4 = np.array([[2.5, 3.3], [4.2, 5.4]], np.float32) * 1.1 * (i + 1)
        x5 = np.array([[2.7, 3.8], [4.9, 5.0]], np.float32) * 1.1 * (i + 1)
        y = x1 + x2 - x3 + x4 - x5
        instances.append({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    for i in range(10):
        assert is_float_equal(result[i]["y"], ys[i])


@serving_test
def test_multi_model_with_batch_dim_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)
tensor_sub = register.declare_model(model_file="tensor_sub.mindir", model_format="MindIR", with_batch_dim=True)

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    y = register.add_stage(tensor_sub, y, x3, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=["tensor_add.mindir", "tensor_sub.mindir"])
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x3 = np.array([[6.2, 5.4]], np.float32) * 1.1 * (i + 1)
        y = x1 + x2 - x3
        instances.append({"x1": x1, "x2": x2, "x3": x3})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y"], ys[0])
    assert is_float_equal(result[1]["y"], ys[1])
    assert is_float_equal(result[2]["y"], ys[2])


@serving_test
def test_multi_model_with_function_front_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
tensor_sub = register.declare_model(model_file="tensor_sub.mindir", model_format="MindIR", with_batch_dim=False)

def test(x1, x2):
    return x1+x2+1

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4, x5):
    y = register.add_stage(test, x1, x2, outputs_count=1)
    y = register.add_stage(tensor_sub, y, x3, outputs_count=1)
    y = register.add_stage(tensor_add, y, x4, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=["tensor_add.mindir", "tensor_sub.mindir"])
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x3 = np.array([[8.5, 7.3], [6.2, 5.4]], np.float32) * 1.1 * (i + 1)
        x4 = np.array([[2.5, 3.3], [4.2, 5.4]], np.float32) * 1.1 * (i + 1)
        x5 = np.array([[2.7, 3.8], [4.9, 5.0]], np.float32) * 1.1 * (i + 1)
        y = x1 + x2 + 1 - x3 + x4
        instances.append({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y"], ys[0])
    assert is_float_equal(result[1]["y"], ys[1])
    assert is_float_equal(result[2]["y"], ys[2])


@serving_test
def test_multi_model_with_function_tail_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
tensor_sub = register.declare_model(model_file="tensor_sub.mindir", model_format="MindIR", with_batch_dim=False)

def test(x1, x2):
    return x1+x2+1

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4, x5):
    y = register.add_stage(tensor_sub, x1, x2, outputs_count=1)
    y = register.add_stage(tensor_add, y, x3, outputs_count=1)
    y = register.add_stage(test, y, x4, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=["tensor_add.mindir", "tensor_sub.mindir"])
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x3 = np.array([[8.5, 7.3], [6.2, 5.4]], np.float32) * 1.1 * (i + 1)
        x4 = np.array([[2.5, 3.3], [4.2, 5.4]], np.float32) * 1.1 * (i + 1)
        x5 = np.array([[2.7, 3.8], [4.9, 5.0]], np.float32) * 1.1 * (i + 1)
        y = x1 - x2 + x3 + x4 + 1
        instances.append({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y"], ys[0])
    assert is_float_equal(result[1]["y"], ys[1])
    assert is_float_equal(result[2]["y"], ys[2])


@serving_test
def test_multi_model_with_function_mid_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
tensor_sub = register.declare_model(model_file="tensor_sub.mindir", model_format="MindIR", with_batch_dim=False)

def test(x1, x2):
    return x1+x2+1

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4, x5):
    y = register.add_stage(tensor_sub, x1, x2, outputs_count=1)
    y = register.add_stage(test, y, x3, outputs_count=1)
    y = register.add_stage(tensor_add, y, x4, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=["tensor_add.mindir", "tensor_sub.mindir"])
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x3 = np.array([[8.5, 7.3], [6.2, 5.4]], np.float32) * 1.1 * (i + 1)
        x4 = np.array([[2.5, 3.3], [4.2, 5.4]], np.float32) * 1.1 * (i + 1)
        x5 = np.array([[2.7, 3.8], [4.9, 5.0]], np.float32) * 1.1 * (i + 1)
        y = x1 - x2 + x3 + 1 + x4
        instances.append({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y"], ys[0])
    assert is_float_equal(result[1]["y"], ys[1])
    assert is_float_equal(result[2]["y"], ys[2])


@serving_test
def test_multi_model_with_function_interlace_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
tensor_sub = register.declare_model(model_file="tensor_sub.mindir", model_format="MindIR", with_batch_dim=False)

def test(x1, x2):
    return x1+x2+1

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4, x5, x6):
    y = register.add_stage(test, x1, x2, outputs_count=1)
    y = register.add_stage(tensor_sub, y, x3, outputs_count=1)
    y = register.add_stage(test, y, x4, outputs_count=1)
    y = register.add_stage(tensor_add, y, x5, outputs_count=1)
    y = register.add_stage(test, y, x6, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=["tensor_add.mindir", "tensor_sub.mindir"])
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x3 = np.array([[8.5, 7.3], [6.2, 5.4]], np.float32) * 1.1 * (i + 1)
        x4 = np.array([[2.5, 3.3], [4.2, 5.4]], np.float32) * 1.1 * (i + 1)
        x5 = np.array([[2.7, 3.8], [4.9, 5.0]], np.float32) * 1.1 * (i + 1)
        x6 = np.array([[3.7, 4.8], [5.9, 6.0]], np.float32) * 1.1 * (i + 1)
        y = x1 + x2 - x3 + x4 + x5 + x6 + 3
        instances.append({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "x6": x6})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y"], ys[0])
    assert is_float_equal(result[1]["y"], ys[1])
    assert is_float_equal(result[2]["y"], ys[2])


@serving_test
def test_multi_model_with_function_call_model_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
tensor_sub = register.declare_model(model_file="tensor_sub.mindir", model_format="MindIR", with_batch_dim=False)

def add_test(x1, x2):
    return tensor_add.call(x1, x2)
    
def sub_test(x1, x2):
    return tensor_sub.call(x1, x2)

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4, x5):
    y = register.add_stage(add_test, x1, x2, outputs_count=1)
    y = register.add_stage(tensor_sub, y, x3, outputs_count=1)
    y = register.add_stage(tensor_add, y, x4, outputs_count=1)
    y = register.add_stage(sub_test, y, x5, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=["tensor_add.mindir", "tensor_sub.mindir"])
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x3 = np.array([[8.5, 7.3], [6.2, 5.4]], np.float32) * 1.1 * (i + 1)
        x4 = np.array([[2.5, 3.3], [4.2, 5.4]], np.float32) * 1.1 * (i + 1)
        x5 = np.array([[2.7, 3.8], [4.9, 5.0]], np.float32) * 1.1 * (i + 1)
        y = x1 + x2 - x3 + x4 - x5
        instances.append({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y"], ys[0])
    assert is_float_equal(result[1]["y"], ys[1])
    assert is_float_equal(result[2]["y"], ys[2])


@serving_test
def test_multi_model_diff_input_output_count_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add_3_2.mindir", model_format="MindIR", with_batch_dim=True)
tensor_sub = register.declare_model(model_file="tensor_sub_2_3.mindir", model_format="MindIR", with_batch_dim=True)

@register.register_method(output_names=["y1", "y2", "y3"])
def predict(x1, x2, x3):
    y1, y2 = register.add_stage(tensor_add, x1, x2, x3, outputs_count=2)
    y1, y2, y3 = register.add_stage(tensor_sub, y1, y2, outputs_count=3)
    return y1, y2, y3
    """
    base = start_serving_server(servable_content, model_file=["tensor_add_3_2.mindir", "tensor_sub_2_3.mindir"])
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x3 = np.array([[6.2, 5.4]], np.float32) * 1.1 * (i + 1)
        # for tensor_add_3_2
        y1 = x1 + x2 + x3
        y2 = y1 + 1
        # for tensor_sub_2_3
        y1 = y1 - y2
        y2 = y1 + 1
        y3 = y1 + 2

        instances.append({"x1": x1, "x2": x2, "x3": x3})
        ys.append([y1, y2, y3])

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y1"], ys[0][0])
    assert is_float_equal(result[0]["y2"], ys[0][1])
    assert is_float_equal(result[0]["y3"], ys[0][2])

    assert is_float_equal(result[1]["y1"], ys[1][0])
    assert is_float_equal(result[1]["y2"], ys[1][1])
    assert is_float_equal(result[1]["y3"], ys[1][2])

    assert is_float_equal(result[2]["y1"], ys[2][0])
    assert is_float_equal(result[2]["y2"], ys[2][1])
    assert is_float_equal(result[2]["y3"], ys[2][2])
