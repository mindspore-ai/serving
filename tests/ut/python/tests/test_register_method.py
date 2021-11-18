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
from common import start_serving_server
from mindspore_serving import server


@serving_test
def test_register_method_with_model_success():
    """
    Feature: test register method
    Description: method with only python function stage, python function has model.call
    Expectation: success to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2):
    y = model.call(x1, x2)
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(call_model, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, version_number=1, start_version_number=1)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    y = x1 + x2
    instances = [{"x1": x1, "x2": x2}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()


@serving_test
def test_register_method_without_add_stage_success():
    """
    Feature: test register method
    Description: method without any stages
    Expectation: success to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

@register.register_method(output_names=["x1", "x2"])
def predict(x1, x2):
    return x1, x2
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2}] * 3

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert len(result) == 3
    assert (result[0]["x1"] == x1).all()
    assert (result[0]["x2"] == x2).all()


@serving_test
def test_register_method_without_register_method_failed():
    """
    Feature: test register method
    Description: without any methods
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
    """
    try:
        start_serving_server(servable_content)
        assert False
    except RuntimeError as e:
        assert "There is no method registered for servable" in str(e)


@serving_test
def test_register_method_two_input_one_output_one_model_stage_input_more_failed():
    """
    Feature: test register method
    Description: model input count not equal to model stage input count
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(tensor_add, x1, x2, x3, outputs_count=1)
    return y
    """
    try:
        start_serving_server(servable_content, model_file="tensor_add.mindir")
        assert False
    except RuntimeError as e:
        assert "The inputs count 3 in register_method not equal to the count 2 defined in model" in str(e)


@serving_test
def test_register_method_two_input_one_output_one_model_stage_input_less_failed():
    """
    Feature: test register method
    Description: model input count not equal to model stage input count
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(tensor_add, x1, outputs_count=1)
    return y
    """
    try:
        start_serving_server(servable_content, model_file="tensor_add.mindir")
        assert False
    except RuntimeError as e:
        assert "The inputs count 1 in register_method not equal to the count 2 defined in model" in str(e)


@serving_test
def test_register_method_two_input_one_output_one_model_stage_input_less2_failed():
    """
    Feature: test register method
    Description: model input count not equal to some model stage input count
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    y = register.add_stage(tensor_add, y, outputs_count=1)
    return y
    """
    try:
        start_serving_server(servable_content, model_file="tensor_add.mindir")
        assert False
    except RuntimeError as e:
        assert "RegisterInputOutputInfo failed, inputs count 1 not match old count 2" in str(e)


@serving_test
def test_register_method_two_input_one_output_one_model_stage_input_less3_failed():
    """
    Feature: test register method
    Description: model input count not equal to model stage input count
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(tensor_add, x1, outputs_count=1)
    return y
    
@register.register_method(output_names="y")
def predict2(x1, x2, x3):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y
    """
    try:
        start_serving_server(servable_content, model_file="tensor_add.mindir")
        assert False
    except RuntimeError as e:
        assert "RegisterInputOutputInfo failed, inputs count 2 not match old count 1" in str(e)


@serving_test
def test_register_method_two_input_one_output_one_model_stage_with_batch_dim_input_more_failed():
    """
    Feature: test register method
    Description: model input count not equal to model stage input count, with_batch_dim is True
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(tensor_add, x1, x2, x3, outputs_count=1)
    return y
    """
    try:
        start_serving_server(servable_content, model_file="tensor_add.mindir")
        assert False
    except RuntimeError as e:
        assert "The inputs count 3 in register_method not equal to the count 2 defined in model" in str(e)


@serving_test
def test_register_method_two_input_one_output_one_model_stage_with_batch_dim_input_less_failed():
    """
    Feature: test register method
    Description: model input count not equal to model stage input count, with_batch_dim is True
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(tensor_add, x1, outputs_count=1)
    return y
    """
    try:
        start_serving_server(servable_content, model_file="tensor_add.mindir")
        assert False
    except RuntimeError as e:
        assert "The inputs count 1 in register_method not equal to the count 2 defined in model" in str(e)


@serving_test
def test_register_method_two_input_two_output_one_model_stage_output_more_failed():
    """
    Feature: test register method
    Description: model output count not equal to model stage output count, with_batch_dim is True
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add_2_2.mindir", model_format="MindIR", with_batch_dim=True)

@register.register_method(output_names=["y1", "y2"])
def predict(x1, x2):
    y1, y2, y3 = register.add_stage(tensor_add, x1, x2, outputs_count=3)
    return y1, y2
    """
    try:
        start_serving_server(servable_content, model_file="tensor_add_2_2.mindir")
        assert False
    except RuntimeError as e:
        assert "The outputs count 3 in register_method not equal to the count 2 defined in model" in str(e)


@serving_test
def test_register_method_three_input_two_output_one_model_stage_output_less_failed():
    """
    Feature: test register method
    Description: model output count not equal to model stage output count, with_batch_dim is True
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add_2_3.mindir", model_format="MindIR", with_batch_dim=True)

@register.register_method(output_names=["y1", "y2"])
def predict(x1, x2):
    y1, y2 = register.add_stage(tensor_add, x1, x2, outputs_count=2)
    return y1, y2
    """
    try:
        start_serving_server(servable_content, model_file="tensor_add_2_3.mindir")
        assert False
    except RuntimeError as e:
        assert "The outputs count 2 in register_method not equal to the count 3 defined in model" in str(e)


@serving_test
def test_register_method_three_input_two_output_one_model_stage_output_less2_failed():
    """
    Feature: test register method
    Description: model output count not equal to some model stage output count, with_batch_dim is True
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add_2_3.mindir", model_format="MindIR", with_batch_dim=True)

@register.register_method(output_names=["y1", "y2"])
def predict(x1, x2):
    y1, y2, y3 = register.add_stage(tensor_add, x1, x2, outputs_count=3)
    y1, y2 = register.add_stage(tensor_add, y1, y2, outputs_count=2)
    return y1, y2
    """
    try:
        start_serving_server(servable_content, model_file="tensor_add_2_3.mindir")
        assert False
    except RuntimeError as e:
        assert "RegisterInputOutputInfo failed, outputs count 2 not match old count 3" in str(e)


@serving_test
def test_register_method_three_input_two_output_one_model_stage_output_less3_failed():
    """
    Feature: test register method
    Description: model output count not equal to some model stage output count, with_batch_dim is True
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add_2_3.mindir", model_format="MindIR", with_batch_dim=True)

@register.register_method(output_names=["y1", "y2"])
def predict(x1, x2):
    y1, y2, y3 = register.add_stage(tensor_add, x1, x2, outputs_count=3)
    return y1, y2

@register.register_method(output_names=["y1", "y2"])
def predict2(x1, x2):
    y1, y2 = register.add_stage(tensor_add, x1, x2, outputs_count=2)
    return y1, y2
    """
    try:
        start_serving_server(servable_content, model_file="tensor_add_2_3.mindir")
        assert False
    except RuntimeError as e:
        assert "RegisterInputOutputInfo failed, outputs count 2 not match old count 3" in str(e)


@serving_test
def test_register_method_model_file_repeat_failed():
    """
    Feature: test register method
    Description: same model file repeatedly used in diff declare_model
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)
tensor_add2 = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=True)

@register.register_method(output_names=["y"])
def predict(x1, x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y
    """
    try:
        start_serving_server(servable_content, model_file="tensor_add.mindir")
        assert False
    except RuntimeError as e:
        assert "model file 'tensor_add.mindir' has already been used" in str(e)


@serving_test
def test_register_method_model_file_repeat2_failed():
    """
    Feature: test register method
    Description: same model file repeatedly used in diff declare_model
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file=["tensor_add.mindir", "tensor_sub.mindir"], model_format="MindIR")
tensor_add2 = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")

@register.register_method(output_names=["y"])
def predict(x1, x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y
    """
    try:
        start_serving_server(servable_content, model_file=["tensor_add.mindir", "tensor_sub.mindir"])
        assert False
    except RuntimeError as e:
        assert "model file 'tensor_add.mindir' has already been used" in str(e)


@serving_test
def test_register_method_model_file_repeat3_failed():
    """
    Feature: test register method
    Description: same model file repeatedly used in diff declare_model
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add2 = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")
tensor_add = register.declare_model(model_file=["tensor_add.mindir", "tensor_sub.mindir"], model_format="MindIR")

@register.register_method(output_names=["y"])
def predict(x1, x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y
    """
    try:
        start_serving_server(servable_content, model_file=["tensor_add.mindir", "tensor_sub.mindir"])
        assert False
    except RuntimeError as e:
        assert "model file 'tensor_add.mindir' has already been used" in str(e)


@serving_test
def test_register_method_method_registered_repeat_failed():
    """
    Feature: test register method
    Description: methods with same name
    Expectation: failed to start serving server.
    """
    servable_content = r"""
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")

@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y

@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y
"""
    try:
        start_serving_server(servable_content)
        assert False
    except RuntimeError as e:
        assert "Method add_cast has been registered more than once." in str(e)


@serving_test
def test_register_method_input_arg_invalid_failed():
    """
    Feature: test register method
    Description: method input args invalid
    Expectation: failed to start serving server.
    """
    servable_content = r"""
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")

@register.register_method(output_names=["y"])
def add_cast(x1, **x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y
"""
    try:
        start_serving_server(servable_content)
        assert False
    except RuntimeError as e:
        assert "'add_cast' input x2 cannot be VAR_KEYWORD !" in str(e)


@serving_test
def test_register_method_input_arg_invalid2_failed():
    """
    Feature: test register method
    Description: method input args invalid
    Expectation: failed to start serving server.
    """
    servable_content = r"""
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")

@register.register_method(output_names=["y"])
def add_cast(x1, *x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y
"""
    try:
        start_serving_server(servable_content)
        assert False
    except RuntimeError as e:
        assert "'add_cast' input x2 cannot be VAR_POSITIONAL !" in str(e)


@serving_test
def test_register_method_function_stage_invalid_input_failed():
    """
    Feature: test register method
    Description: stage input args invalid
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")

def func_test(x1, x2):
    return x1+1, x2+1

@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.add_stage(func_test, x1, np.ones([2,2]), outputs_count=2)
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y
"""
    try:
        start_serving_server(servable_content)
        assert False
    except RuntimeError as e:
        assert "Each value of parameter *args is a placeholder for data and" in str(e)


@serving_test
def test_register_method_function_stage_invalid_input2_failed():
    """
    Feature: test register method
    Description: stage input args invalid
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")

def postprocess(y, data):
    return y
    
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    y = register.add_stage(postprocess, y, np.ones([2,2]), outputs_count=1)
    return y
"""
    try:
        start_serving_server(servable_content)
        assert False
    except RuntimeError as e:
        assert "Each value of parameter *args is a placeholder for data and" in str(e)


@serving_test
def test_register_method_model_stage_invalid_input_failed():
    """
    Feature: test register method
    Description: stage input args invalid
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")

@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.add_stage(tensor_add, x1, np.ones([2,2]), outputs_count=1)
    return y
"""
    try:
        start_serving_server(servable_content)
        assert False
    except RuntimeError as e:
        assert "Each value of parameter *args is a placeholder for data and" in str(e)


@serving_test
def test_register_method_invalid_return_failed():
    """
    Feature: test register method
    Description: method return invalid
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")
    
@register.register_method(output_names=["y", "data"])
def add_cast(x1, x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y, np.ones([2,2])
"""
    try:
        start_serving_server(servable_content)
        assert False
    except RuntimeError as e:
        assert "Each value returned is a placeholder for data and must come from the method" in str(e)


@serving_test
def test_register_method_function_stage_batch_input_count_not_same_failed():
    """
    Feature: test register method
    Description: function stage input count diff in diff method
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")

def func_test_batch(instances):
    results = []
    for instance in instances:
        y1 = instance[0] + instance[1]
        y2 = instance[0] - instance[1]
        results.append([y1, y2])
    return results

@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    x1, x2 = register.add_stage(func_test_batch, x1, x2, outputs_count=2, batch_size=4)
    return y

@register.register_method(output_names=["y"])
def add_cast2(x1, x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    x1, x2 = register.add_stage(func_test_batch, x1, x2, y, outputs_count=2, batch_size=4)
    return y
"""
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content)
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert f"'{base.servable_name}.func_test_batch' inputs count 3 " \
               f"not match last registered count 2" in str(e)


@serving_test
def test_register_method_function_stage_batch_input_count_not_same2_failed():
    """
    Feature: test register method
    Description: function stage input count diff in diff method
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")

def func_test_batch(instances):
    results = []
    for instance in instances:
        y1 = instance[0] + instance[1]
        y2 = instance[0] - instance[1]
        results.append([y1, y2])
    return results

@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.add_stage(func_test_batch, x1, x2, outputs_count=2, batch_size=4)
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y

@register.register_method(output_names=["y"])
def add_cast2(x1, x2, x3):
    x1, x2 = register.add_stage(func_test_batch, x1, x2, x3, outputs_count=2, batch_size=4)
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y
"""
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content)
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert f"'{base.servable_name}.func_test_batch' inputs count 3 " \
               f"not match last registered count 2" in str(e)


@serving_test
def test_register_method_function_stage_batch_output_count_not_same_failed():
    """
    Feature: test register method
    Description: function stage output count diff in diff method
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")

def func_test_batch(instances):
    results = []
    for instance in instances:
        y1 = instance[0] + instance[1]
        y2 = instance[0] - instance[1]
        results.append([y1, y2])
    return results

@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    x1, x2 = register.add_stage(func_test_batch, x1, x2, outputs_count=2, batch_size=4)
    return y

@register.register_method(output_names=["y"])
def add_cast2(x1, x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    x1, x2, x3 = register.add_stage(func_test_batch, x1, x2, outputs_count=3, batch_size=4)
    return y
"""
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content)
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert f"'{base.servable_name}.func_test_batch' outputs count 3 " \
               f"not match last registered count 2" in str(e)


@serving_test
def test_register_method_function_stage_batch_output_count_not_same2_failed():
    """
    Feature: test register method
    Description: function stage output count diff in diff method
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")

def func_test_batch(instances):
    results = []
    for instance in instances:
        y1 = instance[0] + instance[1]
        y2 = instance[0] - instance[1]
        results.append([y1, y2])
    return results

@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.add_stage(func_test_batch, x1, x2, outputs_count=2, batch_size=4)
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y

@register.register_method(output_names=["y"])
def add_cast2(x1, x2):
    x1, x2, x3 = register.add_stage(func_test_batch, x1, x2, outputs_count=3, batch_size=4)
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y
"""
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content)
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert f"'{base.servable_name}.func_test_batch' outputs count 3 " \
               f"not match last registered count 2" in str(e)


@serving_test
def test_register_method_method_output_count_not_match_output_names_failed():
    """
    Feature: test register method
    Description: outputs count registered not equal to the count return in function
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")

@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y, x2
"""
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content)
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert "Method return output size 2 not match registered 1" in str(e)


@serving_test
def test_register_method_method_python_function_batch_size_exist_inconsistently_failed():
    """
    Feature: test register method
    Description: python function used in multi add_stage, one with batch_size, other without batch_size
    Expectation: failed to start serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")

def stage_test_fun(x1, x2):
    return x1+x2

@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.add_stage(stage_test_fun, x1, x2, outputs_count=1)
    return y

@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.add_stage(stage_test_fun, x1, x2, outputs_count=1, batch_size=4)
    return y
"""
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content)
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert "parameter 'batch_size' in multiple 'add_stage' should be enabled or disabled consistently" in str(e)
