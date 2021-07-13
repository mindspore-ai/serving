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

from common import start_serving_server
from common import serving_test, create_client


@serving_test
def test_call_model_two_input_one_output_normal_success():
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
    base = start_serving_server(servable_content)
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
def test_call_model_two_input_one_output_multi_times_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2, x3, x4):
    y1 = model.call(x1, x2)
    y2 = model.call(x3, x4)
    return y1 + y2

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4):
    y = register.add_stage(call_model, x1, x2, x3, x4, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[2.1, 3.2], [4.3, 5.4]], np.float32)
    x4 = np.array([[3.5, 4.6], [5.7, 6.8]], np.float32)
    y = x1 + x2 + x3 + x4
    instances = [{"x1": x1, "x2": x2, "x3": x3, "x4": x4}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()


@serving_test
def test_call_model_two_input_one_output_multi_times_2success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2, x3, x4):
    y1 = model.call(x1, x2)
    y2 = model.call(x3, x4)
    y = model.call(y1, y2)
    return y

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4):
    y = register.add_stage(call_model, x1, x2, x3, x4, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[2.1, 3.2], [4.3, 5.4]], np.float32)
    x4 = np.array([[3.5, 4.6], [5.7, 6.8]], np.float32)
    y = x1 + x2 + x3 + x4
    instances = [{"x1": x1, "x2": x2, "x3": x3, "x4": x4}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()


@serving_test
def test_call_model_two_input_one_output_batch_call_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2, x3, x4):
    instances = []
    instances.append([x1, x2])
    instances.append((x3, x4))
    outputs  = model.call(instances) # return [[x1+x2], [x3+x4]]
    y1 = outputs[0][0]
    y2 = outputs[1][0]
    
    instances = []
    instances.append((y1, y2))
    outputs = model.call(instances)
    y = outputs[0][0]
    return y

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4):
    y = register.add_stage(call_model, x1, x2, x3, x4, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[2.1, 3.2], [4.3, 5.4]], np.float32)
    x4 = np.array([[3.5, 4.6], [5.7, 6.8]], np.float32)
    y = x1 + x2 + x3 + x4
    instances = [{"x1": x1, "x2": x2, "x3": x3, "x4": x4}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()


@serving_test
def test_call_model_batch_call_one_input_one_output_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add_1_1.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2, x3, x4):
    instances = []
    instances.append([x1])
    instances.append([x2])
    instances.append([x3])   
    outputs = model.call(instances)
    y1 = outputs[0][0]
    y2 = outputs[1][0]
    y3 = outputs[2][0]
    y4 = model.call(x4)
    return y1+y2+y3+y4

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4):
    y = register.add_stage(call_model, x1, x2, x3, x4, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add_1_1.mindir")
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[2.1, 3.2], [4.3, 5.4]], np.float32)
    x4 = np.array([[3.5, 4.6], [5.7, 6.8]], np.float32)
    y = x1 + x2 + x3 + x4
    instances = [{"x1": x1, "x2": x2, "x3": x3, "x4": x4}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()


@serving_test
def test_call_model_batch_call_one_input_two_output_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add_1_2.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2, x3, x4):
    _, y1 = model.call(x1) # one instance
    _, y2 = model.call(x2) # one instance
    _, y3 = model.call(x3) # one instance
    _, y4 = model.call(x4) # one instance
    return y1+y2+y3+y4

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4):
    y = register.add_stage(call_model, x1, x2, x3, x4, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add_1_2.mindir")
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[2.1, 3.2], [4.3, 5.4]], np.float32)
    x4 = np.array([[3.5, 4.6], [5.7, 6.8]], np.float32)
    y = x1 + x2 + x3 + x4 + 4
    instances = [{"x1": x1, "x2": x2, "x3": x3, "x4": x4}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()


@serving_test
def test_call_model_batch_call_one_input_two_output_batch_call_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add_1_2.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2, x3, x4):
    instances = []
    instances.append([x1]) # one input
    outputs = model.call(instances) # batch call, one instance
    _, y1 = outputs[0]

    instances = []
    instances.append([x2]) # one input
    outputs = model.call(instances) # batch call, one instance
    _, y2 = outputs[0]
    
    instances = []
    instances.append([x3]) # one input
    instances.append([x4])   
    outputs = model.call(instances) # batch call, two instances
    _, y3 = outputs[0]
    _, y4 = outputs[1]

    return y1+y2+y3+y4

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4):
    y = register.add_stage(call_model, x1, x2, x3, x4, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file="tensor_add_1_2.mindir")
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[2.1, 3.2], [4.3, 5.4]], np.float32)
    x4 = np.array([[3.5, 4.6], [5.7, 6.8]], np.float32)
    y = x1 + x2 + x3 + x4 + 4
    instances = [{"x1": x1, "x2": x2, "x3": x3, "x4": x4}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()


@serving_test
def test_call_model_two_input_one_output_none_instances_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2):
    y = model.call()
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(call_model, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert "Model(tensor_add.mindir).call() failed: no inputs provided" in result["error"]


@serving_test
def test_call_model_two_input_one_output_zero_instances_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2):
    y = model.call([])
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(call_model, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert "Model(tensor_add.mindir).call() failed: Input instances count is 0" in result["error"]


@serving_test
def test_call_model_two_input_one_output_invalid_inputs_format_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2):
    y = model.call([x1, x2]) # expect to be model.call([[x1, x2]])
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(call_model, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert "inputs format invalid" in result["error"]


@serving_test
def test_call_model_two_input_one_output_zero_inputs_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2):
    y = model.call([[]])
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(call_model, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert "The inputs count 0 of instance 0 is not equal to the inputs count 2 of the model" in result["error"]


@serving_test
def test_call_model_two_input_one_output_data_size_error_failed():
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
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2, 3.3], [3.3, 4.4, 5.5]], np.float32)
    x2 = np.array([[5.5, 6.6, 7.7], [7.7, 8.8, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert "Given model input 0 size 24 not match the size 16 defined in model" in result["error"]


@serving_test
def test_call_model_two_input_one_output_data_type_error_failed():
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
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.int32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.int32)
    instances = [{"x1": x1, "x2": x2}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert "Given model input 0 data type kMSI_Int32 not match the data type kMSI_Float32 defined in model" in \
           result["error"]


@serving_test
def test_call_model_two_input_one_output_call_batch_data_size_error_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2, x3, x4):
    instances = []
    instances.append((x1, x2))
    instances.append((x3, x4))
    ys = model.call(instances)
    return ys[0][0] + ys[1][0]

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4):
    y = register.add_stage(call_model, x1, x2, x3, x4, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2, 3.3], [3.3, 4.4, 5.5]], np.float32)
    x2 = np.array([[5.5, 6.6, 7.7], [7.7, 8.8, 8.8]], np.float32)
    x3 = np.array([[1.1, 2.2, 3.3], [3.3, 4.4, 5.5]], np.float32)
    x4 = np.array([[5.5, 6.6, 7.7], [7.7, 8.8, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2, "x3": x3, "x4": x4}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert "Given model input 0 size 24 not match the size 16 defined in model" in result["error"]


@serving_test
def test_call_model_two_input_one_output_call_batch_data_type_error_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2, x3, x4):
    instances = []
    instances.append((x1, x2))
    instances.append((x3, x4))
    ys = model.call(instances)
    return ys[0][0] + ys[1][0]

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4):
    y = register.add_stage(call_model, x1, x2, x3, x4, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.int32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.int32)
    x3 = np.array([[1.1, 2.2], [3.3, 4.4]], np.int32)
    x4 = np.array([[5.5, 6.6], [7.7, 8.8]], np.int32)
    instances = [{"x1": x1, "x2": x2, "x3": x3, "x4": x4}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert "Given model input 0 data type kMSI_Int32 not match the data type kMSI_Float32 defined in model" in \
           result["error"]


@serving_test
def test_call_model_two_input_one_output_more_inputs_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2, x3):
    y = model.call(x1, x2, x3)
    return y

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(call_model, x1, x2, x3, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2, "x3": x3}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert "The inputs count 3 of instance 0 is not equal to the inputs count 2 of the model" in result["error"]


@serving_test
def test_call_model_two_input_one_output_batch_call_more_inputs_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2, x3):
    y = model.call([[x1, x2, x3]])
    return y[0][0]

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(call_model, x1, x2, x3, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2, "x3": x3}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert "The inputs count 3 of instance 0 is not equal to the inputs count 2 of the model" in result["error"]


@serving_test
def test_call_model_two_input_one_output_batch_call_more_inputs2_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2, x3):
    y = model.call([[x1, x2], [x1, x2, x3]])
    return y[0][0]

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(call_model, x1, x2, x3, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2, "x3": x3}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert "The inputs count 3 of instance 1 is not equal to the inputs count 2 of the model" in result["error"]


@serving_test
def test_call_model_two_input_one_output_less_inputs_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2, x3):
    y = model.call(x1)
    return y

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(call_model, x1, x2, x3, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2, "x3": x3}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert "The inputs count 1 of instance 0 is not equal to the inputs count 2 of the model" in result["error"]


@serving_test
def test_call_model_two_input_one_output_batch_call_less_inputs_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2, x3):
    y = model.call([[x1]])
    return y[0][0]

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(call_model, x1, x2, x3, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2, "x3": x3}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert "The inputs count 1 of instance 0 is not equal to the inputs count 2 of the model" in result["error"]


@serving_test
def test_call_model_two_input_one_output_batch_call_less_inputs2_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2, x3):
    y = model.call([[x1], [x1, x2]])
    return y[0][0]

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(call_model, x1, x2, x3, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2, "x3": x3}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert "The inputs count 1 of instance 0 is not equal to the inputs count 2 of the model" in result["error"]


@serving_test
def test_call_model_two_input_one_output_batch_call_less_inputs3_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2, x3):
    y = model.call([[x1, x2], [x1]])
    return y[0][0]

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(call_model, x1, x2, x3, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2, "x3": x3}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert "The inputs count 1 of instance 1 is not equal to the inputs count 2 of the model" in result["error"]


@serving_test
def test_call_model_two_input_one_output_invalid_model_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
model_invalid = register.model.Model("tensor_add_test.mindir")

def call_model(x1, x2):
    y = model_invalid.call(x1, x2)
    return y[0][0]

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(call_model, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert "Model(tensor_add_test.mindir).call() failed: the model is not declared" in result["error"]


@serving_test
def test_call_model_two_input_one_output_with_stage_model_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2):
    y1 = model.call(x1, x2)
    return y1

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4):
    y1 = register.add_stage(call_model, x1, x2, outputs_count=1)
    y2 = register.add_stage(model, y1, x3, outputs_count=1)
    y3 = register.add_stage(call_model, y2, x4, outputs_count=1)
    return y3
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[2.1, 3.2], [4.3, 5.4]], np.float32)
    x4 = np.array([[3.5, 4.6], [5.7, 6.8]], np.float32)
    y = x1 + x2 + x3 + x4
    instances = [{"x1": x1, "x2": x2, "x3": x3, "x4": x4}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()


@serving_test
def test_call_model_two_input_one_output_invalid_subgraph_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2):
    y = model.call(x1, x2, subgraph=1)
    return y[0][0]

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(call_model, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    instances = [{"x1": x1, "x2": x2}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert "The model does not have subgraph of index 1, the subgraph count of the model is 1" in result["error"]


@serving_test
def test_call_model_two_input_one_output_two_subgraph_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file=["tensor_add.mindir", "tensor_sub.mindir"], model_format="MindIR", 
                               with_batch_dim=False)

def call_model(x1, x2, x3):
    y = model.call(x1, x2, subgraph=0)  # x1+x2
    y = model.call(y, x3, subgraph=1)   # y-x3
    return y

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(call_model, x1, x2, x3, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=["tensor_add.mindir", "tensor_sub.mindir"])
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[7.5, 8.6], [9.7, 10.8]], np.float32)
    y = x1 + x2 - x3
    instances = [{"x1": x1, "x2": x2, "x3": x3}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()


@serving_test
def test_call_model_diff_input_output_two_subgraph_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file=["tensor_add_2_3.mindir", "tensor_sub_3_2.mindir"], model_format="MindIR", 
                               with_batch_dim=False)

def call_model(x1, x2, x3, x4, x5):
    y1, y2, y3 = model.call(x1, x2, subgraph=0)  # tensor_add_2_3: 2 input, 3 output
    y4, y5 = model.call(x3, x4, x5, subgraph=1)   # tensor_sub_3_2: 3 input, 2 output
    return y1+y4

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4, x5):
    y = register.add_stage(call_model, x1, x2, x3, x4, x5, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=["tensor_add_2_3.mindir", "tensor_sub_3_2.mindir"])
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[7.5, 8.6], [9.7, 10.8]], np.float32)
    x4 = np.array([[8.5, 10.6], [6.7, 12.8]], np.float32)
    x5 = np.array([[9.5, 11.6], [8.7, 13.8]], np.float32)
    y = (x1 + x2) + (x3 - x4 - x5)
    instances = [{"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()


@serving_test
def test_call_model_diff_input_output_two_subgraph2_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file=["tensor_add_3_2.mindir", "tensor_sub_2_3.mindir"], model_format="MindIR", 
                               with_batch_dim=False)

def call_model(x1, x2, x3, x4, x5):
    y1, y2 = model.call(x1, x2, x3, subgraph=0)  # tensor_add_3_2: 3 input, 2 output
    y3, y4, y5 = model.call(x4, x5, subgraph=1)   # tensor_sub_2_3: 2 input, 3 output
    return y1+y3

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4, x5):
    y = register.add_stage(call_model, x1, x2, x3, x4, x5, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=["tensor_add_3_2.mindir", "tensor_sub_2_3.mindir"])
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[7.5, 8.6], [9.7, 10.8]], np.float32)
    x4 = np.array([[8.5, 10.6], [6.7, 12.8]], np.float32)
    x5 = np.array([[9.5, 11.6], [8.7, 13.8]], np.float32)
    y = (x1 + x2 + x3) + (x4 - x5)
    instances = [{"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()


@serving_test
def test_call_model_diff_input_output_two_subgraph_inputs_count_not_match_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file=["tensor_add_3_2.mindir", "tensor_sub_2_3.mindir"], model_format="MindIR", 
                               with_batch_dim=False)

def call_model(x1, x2, x3, x4, x5):
    y1, y2 = model.call(x1, x2, x3, subgraph=0)  # tensor_add_3_2: 3 input, 2 output
    y3, y4, y5 = model.call(x4, x5, x3, subgraph=1)   # tensor_sub_2_3: 2 input, 3 output
    return y1+y3

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4, x5):
    y = register.add_stage(call_model, x1, x2, x3, x4, x5, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=["tensor_add_3_2.mindir", "tensor_sub_2_3.mindir"])
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[7.5, 8.6], [9.7, 10.8]], np.float32)
    x4 = np.array([[8.5, 10.6], [6.7, 12.8]], np.float32)
    x5 = np.array([[9.5, 11.6], [8.7, 13.8]], np.float32)
    instances = [{"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert "The inputs count 3 of instance 0 is not equal to the inputs count 2 of the model" in result["error"]


@serving_test
def test_call_model_two_input_one_output_two_model_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
tensor_sub = register.declare_model(model_file="tensor_sub.mindir", model_format="MindIR", with_batch_dim=False)
def call_model(x1, x2, x3):
    y = tensor_add.call(x1, x2)  # x1+x2
    y = tensor_sub.call(y, x3)   # y-x3
    return y

@register.register_method(output_names="y")
def predict(x1, x2, x3):
    y = register.add_stage(call_model, x1, x2, x3, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=["tensor_add.mindir", "tensor_sub.mindir"])
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[7.5, 8.6], [9.7, 10.8]], np.float32)
    y = x1 + x2 - x3
    instances = [{"x1": x1, "x2": x2, "x3": x3}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()


@serving_test
def test_call_model_diff_input_output_two_model_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add_2_3.mindir", model_format="MindIR", with_batch_dim=False)
tensor_sub = register.declare_model(model_file="tensor_sub_3_2.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2, x3, x4, x5):
    y1, y2, y3 = tensor_add.call(x1, x2)  # tensor_add_2_3: 2 input, 3 output
    y4, y5 = tensor_sub.call(x3, x4, x5)   # tensor_sub_3_2: 3 input, 2 output
    return y1+y4

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4, x5):
    y = register.add_stage(call_model, x1, x2, x3, x4, x5, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=["tensor_add_2_3.mindir", "tensor_sub_3_2.mindir"])
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[7.5, 8.6], [9.7, 10.8]], np.float32)
    x4 = np.array([[8.5, 10.6], [6.7, 12.8]], np.float32)
    x5 = np.array([[9.5, 11.6], [8.7, 13.8]], np.float32)
    y = (x1 + x2) + (x3 - x4 - x5)
    instances = [{"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()


@serving_test
def test_call_model_diff_input_output_two_model2_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add_3_2.mindir", model_format="MindIR", with_batch_dim=False)
tensor_sub = register.declare_model(model_file="tensor_sub_2_3.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2, x3, x4, x5):
    y1, y2 = tensor_add.call(x1, x2, x3)  # tensor_add_3_2: 3 input, 2 output
    y3, y4, y5 = tensor_sub.call(x4, x5)   # tensor_sub_2_3: 2 input, 3 output
    return y1+y3

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4, x5):
    y = register.add_stage(call_model, x1, x2, x3, x4, x5, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=["tensor_add_3_2.mindir", "tensor_sub_2_3.mindir"])
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[7.5, 8.6], [9.7, 10.8]], np.float32)
    x4 = np.array([[8.5, 10.6], [6.7, 12.8]], np.float32)
    x5 = np.array([[9.5, 11.6], [8.7, 13.8]], np.float32)
    y = (x1 + x2 + x3) + (x4 - x5)
    instances = [{"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()


@serving_test
def test_call_model_diff_input_output_two_model_inputs_count_not_match_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add_3_2.mindir", model_format="MindIR", with_batch_dim=False)
tensor_sub = register.declare_model(model_file="tensor_sub_2_3.mindir", model_format="MindIR", with_batch_dim=False)

def call_model(x1, x2, x3, x4, x5):
    y1, y2 = tensor_add.call(x1, x2)  # tensor_add_3_2: 3 input, 2 output
    y3, y4, y5 = tensor_sub.call(x4, x5)   # tensor_sub_2_3: 2 input, 3 output
    return y1+y3

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4, x5):
    y = register.add_stage(call_model, x1, x2, x3, x4, x5, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=["tensor_add_3_2.mindir", "tensor_sub_2_3.mindir"])
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    x3 = np.array([[7.5, 8.6], [9.7, 10.8]], np.float32)
    x4 = np.array([[8.5, 10.6], [6.7, 12.8]], np.float32)
    x5 = np.array([[9.5, 11.6], [8.7, 13.8]], np.float32)
    instances = [{"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert "The inputs count 2 of instance 0 is not equal to the inputs count 3 of the model" in result["error"]


@serving_test
def test_call_model_diff_input_output_two_model_with_bach_dim_success():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add_3_2.mindir", model_format="MindIR", with_batch_dim=True)
tensor_sub = register.declare_model(model_file="tensor_sub_2_3.mindir", model_format="MindIR", with_batch_dim=True)

def call_model(x1, x2, x3, x4, x5):
    y1, y2 = tensor_add.call(x1, x2, x3)  # tensor_add_3_2: 3 input, 2 output
    y3, y4, y5 = tensor_sub.call(x4, x5)   # tensor_sub_2_3: 2 input, 3 output
    return y1+y3

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4, x5):
    y = register.add_stage(call_model, x1, x2, x3, x4, x5, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=["tensor_add_3_2.mindir", "tensor_sub_2_3.mindir"])
    # Client
    x1 = np.array([[3.3, 4.4]], np.float32)
    x2 = np.array([[7.7, 8.8]], np.float32)
    x3 = np.array([[9.7, 10.8]], np.float32)
    x4 = np.array([[6.7, 12.8]], np.float32)
    x5 = np.array([[8.7, 13.8]], np.float32)
    y = (x1 + x2 + x3) + (x4 - x5)
    instances = [{"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}]

    client = create_client("localhost:5500", base.servable_name, "predict")
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()
