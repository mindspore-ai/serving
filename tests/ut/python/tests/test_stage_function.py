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
def test_stage_function_one_function_stage_float_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)

def test_concat(x1, x2):
    return x1 + x2

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(test_concat, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    ys = []
    x1s = []
    x2s = []
    x1s.append(np.array([[101.1, 205.2], [41.3, 62.4]], np.float32))
    x2s.append(np.array([[3.5, 5.6], [7.7, 9.8]], np.float32))
    x1s.append(np.array([[41.3, 32.2], [4.1, 3.9]], np.float32))
    x2s.append(np.array([[1.4, 4.5], [9.6, 19.7]], np.float32))
    x1s.append(np.array([[11.1, 21.2], [41.9, 61.8]], np.float32))
    x2s.append(np.array([[31.5, 51.7], [71.4, 91.3]], np.float32))
    for i in range(3):
        instances.append({"x1": x1s[i], "x2": x2s[i]})
        y = x1s[i] + x2s[i]
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == ys[0]).all()
    assert (result[1]["y"] == ys[1]).all()
    assert (result[2]["y"] == ys[2]).all()


@serving_test
def test_stage_function_one_function_stage_two_output_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)

def test_concat(x1):
    return x1 + 1, x1-1

@register.register_method(output_names=["y1", "y2"])
def predict(x1):
    y1, y2 = register.add_stage(test_concat, x1, outputs_count=2)
    return y1, y2
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    y1s = []
    y2s = []
    x1s = []
    x1s.append(np.array([[101.1, 205.2], [41.3, 62.4]], np.float32))
    x1s.append(np.array([[41.3, 32.2], [4.1, 3.9]], np.float32))
    x1s.append(np.array([[11.1, 21.2], [41.9, 61.8]], np.float32))
    for i in range(3):
        instances.append({"x1": x1s[i]})
        y1s.append(x1s[i] + 1)
        y2s.append(x1s[i] - 1)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y1"] == y1s[0]).all()
    assert (result[1]["y1"] == y1s[1]).all()
    assert (result[2]["y1"] == y1s[2]).all()
    assert (result[0]["y2"] == y2s[0]).all()
    assert (result[1]["y2"] == y2s[1]).all()
    assert (result[2]["y2"] == y2s[2]).all()


@serving_test
def test_stage_function_one_function_stage_output_more_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test(x1, x2):
    return x1+x2, x1-x2, 1

@register.register_method(output_names=["y1", "y2"])
def predict(x1, x2):
    y1, y2 = register.add_stage(func_test, x1, x2, outputs_count=2)
    return y1, y2
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2}] * 3

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    if isinstance(result, dict):
        assert "Servable is not available" in result["error"] \
               or f"Call Function '{base.servable_name}.func_test' Failed" in result["error"]
    else:
        assert "Servable is not available" in result[0]["error"] \
               or f"Call Function '{base.servable_name}.func_test' Failed" in result[0]["error"]


@serving_test
def test_stage_function_one_function_stage_output_less_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test(x1, x2):
    return x1+x2

@register.register_method(output_names=["y1", "y2"])
def predict(x1, x2):
    y1, y2 = register.add_stage(func_test, x1, x2, outputs_count=2)
    return y1, y2
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2}] * 3

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    if isinstance(result, dict):
        assert "Servable is not available" in result["error"] \
               or f"Call Function '{base.servable_name}.func_test' Failed" in result["error"]
    else:
        assert "Servable is not available" in result[0]["error"] \
               or f"Call Function '{base.servable_name}.func_test' Failed" in result[0]["error"]


@serving_test
def test_stage_function_one_function_stage_error_outputs_count_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test(x1, x2):
    return x1+x2

@register.register_method(output_names=["y1", "y2"])
def predict(x1, x2):
    y1, y2 = register.add_stage(func_test, x1, x2, outputs_count=3)
    return y1, y2
    """
    try:
        start_serving_server(servable_content)
        assert False
    except RuntimeError as e:
        assert "too many values to unpack (expected 2)" in str(e)


@serving_test
def test_stage_function_one_function_stage_error_outputs_count2_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test(x1, x2):
    return x1+x2

@register.register_method(output_names=["y1", "y2"])
def predict(x1, x2):
    y1, y2 = register.add_stage(func_test, x1, x2, outputs_count=1)
    return y1, y2
    """
    try:
        start_serving_server(servable_content)
        assert False
    except RuntimeError as e:
        assert "cannot unpack non-iterable _TensorDef object" in str(e)


@serving_test
def test_stage_function_one_function_stage_input_more_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test(x1, x2, x3):
    return x1+x2

@register.register_method(output_names=["y1", "y2"])
def predict(x1, x2):
    y1, y2 = register.add_stage(func_test, x1, x2, outputs_count=2)
    return y1, y2
    """
    try:
        start_serving_server(servable_content)
        assert False
    except RuntimeError as e:
        assert "function func_test input args count 3 not match the count 2 registered in method" in str(e)


@serving_test
def test_stage_function_one_function_stage_input_less_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test(x1):
    return x1+x2

@register.register_method(output_names=["y1", "y2"])
def predict(x1, x2):
    y1, y2 = register.add_stage(func_test, x1, x2, outputs_count=2)
    return y1, y2
    """
    try:
        start_serving_server(servable_content)
        assert False
    except RuntimeError as e:
        assert "function func_test input args count 1 not match the count 2 registered in method" in str(e)


@serving_test
def test_stage_function_one_function_stage_raise_exception_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test(x1, x2):
    raise RuntimeError("runtime error text")

@register.register_method(output_names=["y1", "y2"])
def predict(x1, x2):
    y1, y2 = register.add_stage(func_test, x1, x2, outputs_count=2)
    return y1, y2
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2}] * 3

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    if isinstance(result, dict):
        assert "Servable is not available" in result["error"] \
               or f"Call Function '{base.servable_name}.func_test' Failed" in result["error"]
    else:
        assert "Servable is not available" in result[0]["error"] \
               or f"Call Function '{base.servable_name}.func_test' Failed" in result[0]["error"]


@serving_test
def test_stage_function_one_function_stage_none_outputs_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test(x1, x2):
    print("none outputs")

@register.register_method(output_names=["y1", "y2"])
def predict(x1, x2):
    y1, y2 = register.add_stage(func_test, x1, x2, outputs_count=2)
    return y1, y2
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2}] * 3

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    if isinstance(result, dict):
        assert "Servable is not available" in result["error"] \
               or f"Call Function '{base.servable_name}.func_test' Failed" in result["error"]
    else:
        assert "Servable is not available" in result[0]["error"] \
               or f"Call Function '{base.servable_name}.func_test' Failed" in result[0]["error"]


@serving_test
def test_stage_function_one_function_stage_invalid_output_dtype_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test(x1, x2):
    return x1.dtype, x2.dtype

@register.register_method(output_names=["y1", "y2"])
def predict(x1, x2):
    y1, y2 = register.add_stage(func_test, x1, x2, outputs_count=2)
    return y1, y2
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2}] * 3

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    if isinstance(result, dict):
        assert "Servable is not available" in result["error"] \
               or f"Call Function '{base.servable_name}.func_test' Failed" in result["error"]
    else:
        assert "Servable is not available" in result[0]["error"] \
               or f"Call Function '{base.servable_name}.func_test' Failed" in result[0]["error"]


@serving_test
def test_stage_function_one_function_stage_batch_size_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test_batch(instances):
    results = []
    for instance in instances:
        y = instance[0] + instance[1]
        results.append([y])
    return results

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(func_test_batch, x1, x2, outputs_count=1, batch_size=2)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        y = x1 + x2
        instances.append({"x1": x1, "x2": x2})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y"], ys[0])
    assert is_float_equal(result[1]["y"], ys[1])
    assert is_float_equal(result[2]["y"], ys[2])


@serving_test
def test_stage_function_one_function_stage_batch_size2_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test_batch(instances):
    results = []
    for instance in instances:
        y = instance[0] + instance[1]
        results.append(y)
    return results

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(func_test_batch, x1, x2, outputs_count=1, batch_size=2)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        y = x1 + x2
        instances.append({"x1": x1, "x2": x2})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y"], ys[0])
    assert is_float_equal(result[1]["y"], ys[1])
    assert is_float_equal(result[2]["y"], ys[2])


@serving_test
def test_stage_function_one_function_stage_batch_size3_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test_batch(instances):
    results = []
    for instance in instances:
        y = instance[0] + instance[1]
        yield y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(func_test_batch, x1, x2, outputs_count=1, batch_size=2)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        y = x1 + x2
        instances.append({"x1": x1, "x2": x2})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y"], ys[0])
    assert is_float_equal(result[1]["y"], ys[1])
    assert is_float_equal(result[2]["y"], ys[2])


@serving_test
def test_stage_function_one_function_stage_batch_size4_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test_batch(instances):
    results = []
    for instance in instances:
        y = instance[0] + instance[1]
        yield [y]

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(func_test_batch, x1, x2, outputs_count=1, batch_size=2)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        y = x1 + x2
        instances.append({"x1": x1, "x2": x2})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y"], ys[0])
    assert is_float_equal(result[1]["y"], ys[1])
    assert is_float_equal(result[2]["y"], ys[2])


@serving_test
def test_stage_function_one_function_stage_batch_size_equal1_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test_batch(instances):
    results = []
    for instance in instances:
        y = instance[0] + instance[1]
        yield y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(func_test_batch, x1, x2, outputs_count=1, batch_size=1)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    ys = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        y = x1 + x2
        instances.append({"x1": x1, "x2": x2})
        ys.append(y)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y"], ys[0])
    assert is_float_equal(result[1]["y"], ys[1])
    assert is_float_equal(result[2]["y"], ys[2])


@serving_test
def test_stage_function_one_function_stage_error_batch_size_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test_batch(instances):
    results = []
    for instance in instances:
        y = instance[0] + instance[1]
        yield y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(func_test_batch, x1, x2, outputs_count=1, batch_size=0)
    return y
    """
    try:
        start_serving_server(servable_content, model_file="tensor_add.mindir")
        assert False
    except RuntimeError as e:
        assert "Parameter 'batch_size' should be >= 1" in str(e)


@serving_test
def test_stage_function_one_function_stage_batch_size_two_outputs_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test_batch(instances):
    results = []
    for instance in instances:
        y1 = instance[0] + instance[1]
        y2 = instance[0] - instance[1]
        yield y1, y2

@register.register_method(output_names=["y1", "y2"])
def predict(x1, x2):
    y1, y2 = register.add_stage(func_test_batch, x1, x2, outputs_count=2, batch_size=2)
    return y1, y2
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    y1s = []
    y2s = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        y1 = x1 + x2
        y2 = x1 - x2
        instances.append({"x1": x1, "x2": x2})
        y1s.append(y1)
        y2s.append(y2)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y1"], y1s[0])
    assert is_float_equal(result[1]["y1"], y1s[1])
    assert is_float_equal(result[2]["y1"], y1s[2])
    assert is_float_equal(result[0]["y2"], y2s[0])
    assert is_float_equal(result[1]["y2"], y2s[1])
    assert is_float_equal(result[2]["y2"], y2s[2])


@serving_test
def test_stage_function_one_function_stage_batch_size_two_outputs_multi_times_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test_batch(instances):
    results = []
    for instance in instances:
        y1 = instance[0] + instance[1]
        y2 = instance[0] - instance[1]
        yield y1, y2

@register.register_method(output_names=["y1", "y2"])
def predict(x1, x2):
    y1, y2 = register.add_stage(func_test_batch, x1, x2, outputs_count=2, batch_size=2)
    y1, y2 = register.add_stage(func_test_batch, y1, y2, outputs_count=2, batch_size=2)
    return y1, y2
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    y1s = []
    y2s = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        y1, y2 = x1 + x2, x1 - x2
        y1, y2 = y1 + y2, y1 - y2
        instances.append({"x1": x1, "x2": x2})
        y1s.append(y1)
        y2s.append(y2)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y1"], y1s[0])
    assert is_float_equal(result[1]["y1"], y1s[1])
    assert is_float_equal(result[2]["y1"], y1s[2])
    assert is_float_equal(result[0]["y2"], y2s[0])
    assert is_float_equal(result[1]["y2"], y2s[1])
    assert is_float_equal(result[2]["y2"], y2s[2])


@serving_test
def test_stage_function_one_function_stage_batch_size_two_outputs2_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test_batch(instances):
    results = []
    for instance in instances:
        y1 = instance[0] + instance[1]
        y2 = instance[0] - instance[1]
        results.append([y1, y2])
    return results

@register.register_method(output_names=["y1", "y2"])
def predict(x1, x2):
    y1, y2 = register.add_stage(func_test_batch, x1, x2, outputs_count=2, batch_size=2)
    return y1, y2
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    y1s = []
    y2s = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        y1 = x1 + x2
        y2 = x1 - x2
        instances.append({"x1": x1, "x2": x2})
        y1s.append(y1)
        y2s.append(y2)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y1"], y1s[0])
    assert is_float_equal(result[1]["y1"], y1s[1])
    assert is_float_equal(result[2]["y1"], y1s[2])
    assert is_float_equal(result[0]["y2"], y2s[0])
    assert is_float_equal(result[1]["y2"], y2s[1])
    assert is_float_equal(result[2]["y2"], y2s[2])


@serving_test
def test_stage_function_one_function_stage_batch_size_input_more_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test_batch(instances):
    results = []
    for instance in instances:
        y1 = instance[0] + instance[1]
        y2 = instance[0] - instance[1]
        results.append([y1, y2])
    return results

@register.register_method(output_names=["y1", "y2"])
def predict(x1, x2, x3):
    y1, y2 = register.add_stage(func_test_batch, x1, x2, x3, outputs_count=2, batch_size=2)
    return y1, y2
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    y1s = []
    y2s = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x3 = np.array([[1.5, 2.6], [3.7, 4.8]], np.float32) * 1.1 * (i + 1)
        y1 = x1 + x2
        y2 = x1 - x2
        instances.append({"x1": x1, "x2": x2, "x3": x3})
        y1s.append(y1)
        y2s.append(y2)

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert is_float_equal(result[0]["y1"], y1s[0])
    assert is_float_equal(result[1]["y1"], y1s[1])
    assert is_float_equal(result[2]["y1"], y1s[2])
    assert is_float_equal(result[0]["y2"], y2s[0])
    assert is_float_equal(result[1]["y2"], y2s[1])
    assert is_float_equal(result[2]["y2"], y2s[2])


@serving_test
def test_stage_function_one_function_stage_batch_size_input_less_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test_batch(instances):
    results = []
    for instance in instances:
        y1 = instance[0] + instance[1]
        y2 = instance[0] - instance[1]
        results.append([y1, y2])
    return results

@register.register_method(output_names=["y1", "y2"])
def predict(x1):
    y1, y2 = register.add_stage(func_test_batch, x1, outputs_count=2, batch_size=2)
    return y1, y2
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    for i in range(3):
        x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32) * 1.1 * (i + 1)
        instances.append({"x1": x1})

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    if isinstance(result, dict):
        assert "Servable is not available" in result["error"] \
               or f"Call Function '{base.servable_name}.func_test_batch' Failed" in result["error"]
    else:
        assert "Servable is not available" in result[0]["error"] \
               or f"Call Function '{base.servable_name}.func_test_batch' Failed" in result[0]["error"]


@serving_test
def test_stage_function_one_function_stage_batch_size_output_more_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test_batch(instances):
    results = []
    for instance in instances:
        y1 = instance[0] + instance[1]
        y2 = instance[0] - instance[1]
        results.append([y1, y2])
    return results

@register.register_method(output_names=["y1", "y2"])
def predict(x1, x2):
    y1, y2, y3 = register.add_stage(func_test_batch, x1, x2, outputs_count=3, batch_size=2)
    return y1, y2
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    for i in range(3):
        x1 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[1.5, 2.6], [3.7, 4.8]], np.float32) * 1.1 * (i + 1)
        instances.append({"x1": x1, "x2": x2})

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    if isinstance(result, dict):
        assert "Servable is not available" in result["error"] \
               or f"Call Function '{base.servable_name}.func_test_batch' Failed" in result["error"]
    else:
        assert "Servable is not available" in result[0]["error"] \
               or f"Call Function '{base.servable_name}.func_test_batch' Failed" in result[0]["error"]


@serving_test
def test_stage_function_one_function_stage_batch_size_output_less_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test_batch(instances):
    results = []
    for instance in instances:
        y1 = instance[0] + instance[1]
        y2 = instance[0] - instance[1]
        results.append([y1, y2])
    return results

@register.register_method(output_names=["y1"])
def predict(x1, x2):
    y1 = register.add_stage(func_test_batch, x1, x2, outputs_count=1, batch_size=2)
    return y1
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    for i in range(3):
        x1 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[1.5, 2.6], [3.7, 4.8]], np.float32) * 1.1 * (i + 1)
        instances.append({"x1": x1, "x2": x2})

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    if isinstance(result, dict):
        assert "Servable is not available" in result["error"] \
               or f"Call Function '{base.servable_name}.func_test_batch' Failed" in result["error"]
    else:
        assert "Servable is not available" in result[0]["error"] \
               or f"Call Function '{base.servable_name}.func_test_batch' Failed" in result[0]["error"]


@serving_test
def test_stage_function_one_function_stage_batch_size_output_less2_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test_batch(instances):
    results = []
    for instance in instances:
        y1 = instance[0] + instance[1]
        y2 = instance[0] - instance[1]
        yield y1, y2

@register.register_method(output_names=["y1"])
def predict(x1, x2):
    y1 = register.add_stage(func_test_batch, x1, x2, outputs_count=1, batch_size=2)
    return y1
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    for i in range(3):
        x1 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[1.5, 2.6], [3.7, 4.8]], np.float32) * 1.1 * (i + 1)
        instances.append({"x1": x1, "x2": x2})

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    if isinstance(result, dict):
        assert "Servable is not available" in result["error"] \
               or f"Call Function '{base.servable_name}.func_test_batch' Failed" in result["error"]
    else:
        assert "Servable is not available" in result[0]["error"] \
               or f"Call Function '{base.servable_name}.func_test_batch' Failed" in result[0]["error"]


@serving_test
def test_stage_function_one_function_stage_batch_size_raise_exception_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test_batch(instances):
    raise RuntimeError("runtime error test")

@register.register_method(output_names=["y1"])
def predict(x1, x2):
    y1 = register.add_stage(func_test_batch, x1, x2, outputs_count=1, batch_size=2)
    return y1
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    for i in range(3):
        x1 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[1.5, 2.6], [3.7, 4.8]], np.float32) * 1.1 * (i + 1)
        instances.append({"x1": x1, "x2": x2})

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    if isinstance(result, dict):
        assert "Servable is not available" in result["error"] \
               or f"Call Function '{base.servable_name}.func_test_batch' Failed" in result["error"]
    else:
        assert "Servable is not available" in result[0]["error"] \
               or f"Call Function '{base.servable_name}.func_test_batch' Failed" in result[0]["error"]


@serving_test
def test_stage_function_one_function_stage_batch_size_none_return_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test_batch(instances):
    pass

@register.register_method(output_names=["y1"])
def predict(x1, x2):
    y1 = register.add_stage(func_test_batch, x1, x2, outputs_count=1, batch_size=2)
    return y1
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    for i in range(3):
        x1 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[1.5, 2.6], [3.7, 4.8]], np.float32) * 1.1 * (i + 1)
        instances.append({"x1": x1, "x2": x2})

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    if isinstance(result, dict):
        assert "Servable is not available" in result["error"] \
               or f"Call Function '{base.servable_name}.func_test_batch' Failed" in result["error"]
    else:
        assert "Servable is not available" in result[0]["error"] \
               or f"Call Function '{base.servable_name}.func_test_batch' Failed" in result[0]["error"]


@serving_test
def test_stage_function_one_function_stage_batch_size_invalid_output_dtype_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def func_test_batch(instances):
    results = []
    for instance in instances:
        y1 = instance[0] + instance[1]
        y2 = instance[0] - instance[1]
        results.append([y1.dtype, y2.dtype])
    return results

@register.register_method(output_names=["y1"])
def predict(x1, x2):
    y1 = register.add_stage(func_test_batch, x1, x2, outputs_count=1, batch_size=2)
    return y1
    """
    base = start_serving_server(servable_content, model_file="tensor_add.mindir")
    # Client
    instances = []
    for i in range(3):
        x1 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32) * 1.1 * (i + 1)
        x2 = np.array([[1.5, 2.6], [3.7, 4.8]], np.float32) * 1.1 * (i + 1)
        instances.append({"x1": x1, "x2": x2})

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    if isinstance(result, dict):
        assert "Servable is not available" in result["error"] \
               or f"Call Function '{base.servable_name}.func_test_batch' Failed" in result["error"]
    else:
        assert "Servable is not available" in result[0]["error"] \
               or f"Call Function '{base.servable_name}.func_test_batch' Failed" in result[0]["error"]


@serving_test
def test_servable_postprocess_result_count_less():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)

def postprocess(instances):
    count = len(instances)
    for i in range(count -1):
        yield i
    
@register.register_method(output_names=["y"])
def add_common(x1, x2):
    y = register.add_stage(model, x1, x2, outputs_count=1)
    y = register.add_stage(postprocess, y, outputs_count=1, batch_size=4, tag="Postprocess")
    return y
"""
    base = start_serving_server(servable_content)
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
    assert "Postprocess Failed" in str(result[1]["error"]) or 'servable is not available' in str(result[1]["error"])


@serving_test
def test_servable_postprocess_result_count_more():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)

def postprocess(instances):
    count = len(instances)
    for i in range(count + 1):
        yield i
    
@register.register_method(output_names=["y"])
def add_common(x1, x2):
    y = register.add_stage(model, x1, x2, outputs_count=1)
    y = register.add_stage(postprocess, y, outputs_count=1, batch_size=4, tag="Postprocess")
    return y
"""
    base = start_serving_server(servable_content)
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
def test_stage_function_preprocess_result_count_less():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)

def preprocess(instances):
    count = len(instances)
    for i in range(count-1):
        yield i
    
@register.register_method(output_names=["y"])
def add_common(x1, x2):
    x3 = register.add_stage(preprocess, x1, outputs_count=1, batch_size=4, tag="Preprocess")
    y = register.add_stage(model, x1, x2, outputs_count=1)
    return x3
"""
    base = start_serving_server(servable_content)
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
    if isinstance(result, list):
        assert "Preprocess Failed" in str(result[1]["error"]) or "servable is not available" in str(result[1]["error"])
    else:
        assert "Preprocess Failed" in str(result["error"]) or "servable is not available" in str(result["error"])


@serving_test
def test_stage_function_preprocess_result_count_more():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)

def preprocess(instances):
    count = len(instances)
    for i in range(count+1):
        yield i
    
@register.register_method(output_names=["y"])
def add_common(x1, x2):
    x3 = register.add_stage(preprocess, x1, outputs_count=1, batch_size=4, tag="Preprocess")
    y = register.add_stage(model, x1, x2, outputs_count=1)
    return x3
"""
    base = start_serving_server(servable_content)
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
