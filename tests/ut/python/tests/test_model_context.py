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
"""Test Model DeviceInfo"""

import os
import numpy as np
from common import serving_test, start_serving_server, create_client
from mindspore_serving.server.register import Context, GPUDeviceInfo, CPUDeviceInfo
from mindspore_serving.server.register import AscendDeviceInfo, GpuOptions, AclOptions


@serving_test
def test_model_context_device_info_set_get_success():
    """
    Feature: Model Device info
    Description: Test set and get device info
    Expectation: the values gotten are equal to the values set.
    """
    context = Context(thread_num=3, thread_affinity_core_list=[1, 2, 3], enable_parallel=True)
    model_context = context.model_context
    assert model_context.thread_num == 3
    assert set(model_context.thread_affinity_core_list) == {1, 2, 3}
    assert model_context.enable_parallel == 1

    # declare model and start_servable and load model and build model
    gpu_device_info = GPUDeviceInfo(precision_mode="fp16")
    gpu_map = gpu_device_info.context_map
    assert gpu_map["precision_mode"] == "fp16"
    assert gpu_map["device_type"] == "gpu"

    cpu_device_info = CPUDeviceInfo(precision_mode="fp16")
    cpu_map = cpu_device_info.context_map
    assert cpu_map["precision_mode"] == "fp16"
    assert cpu_map["device_type"] == "cpu"

    ascend_device_info = AscendDeviceInfo(insert_op_cfg_path="some path of insert_op_cfg_path",
                                          input_format="NHWC1C0",
                                          input_shape="input_op_name1: n1,c2,h3,w4;input_op_name2: n4,c3,h2,w1",
                                          output_type="FP16",
                                          precision_mode="allow_mix_precision",
                                          op_select_impl_mode="high_precision",
                                          fusion_switch_config_path="some path of fusion_switch_config_path",
                                          buffer_optimize_mode="l1_and_l2_optimize")
    ascend310_map = ascend_device_info.context_map
    assert ascend310_map["insert_op_cfg_path"] == "some path of insert_op_cfg_path"
    assert ascend310_map["input_format"] == "NHWC1C0"
    assert ascend310_map["input_shape"] == "input_op_name1: n1,c2,h3,w4;input_op_name2: n4,c3,h2,w1"
    assert ascend310_map["output_type"] == "FP16"
    assert ascend310_map["precision_mode"] == "allow_mix_precision"
    assert ascend310_map["op_select_impl_mode"] == "high_precision"
    assert ascend310_map["fusion_switch_config_path"] == "some path of fusion_switch_config_path"
    assert ascend310_map["buffer_optimize_mode"] == "l1_and_l2_optimize"
    assert ascend310_map["device_type"] == "ascend"

    context.append_device_info(gpu_device_info)
    context.append_device_info(cpu_device_info)
    context.append_device_info(ascend_device_info)

    assert len(model_context.device_list) == 3
    assert model_context.device_list[0]["device_type"] == "gpu"
    assert model_context.device_list[1]["precision_mode"] == "fp16"
    assert model_context.device_list[2]["precision_mode"] == "allow_mix_precision"


@serving_test
def test_model_context_device_info_repeat_append_ascend_failed():
    """
    Feature: Model Device info
    Description: Repeat append AscendDeviceInfo
    Expectation: raise RuntimeError
    """
    context = Context()
    context.append_device_info(AscendDeviceInfo())
    try:
        context.append_device_info(AscendDeviceInfo())
        assert False
    except RuntimeError as e:
        assert "Device info of type ascend has already been appended" in str(e)


@serving_test
def test_model_context_options_set_get_success():
    """
    Feature: Model options
    Description: Test set and get options
    Expectation: the values gotten are equal to the values set.
    """
    gpu_options = GpuOptions(precision_mode="fp16")
    gpu_device_list = gpu_options.context.model_context.device_list

    assert gpu_device_list[0]["device_type"] == "gpu"
    assert gpu_device_list[0]["precision_mode"] == "fp16"

    acl_options = AclOptions(insert_op_cfg_path="some path of insert_op_cfg_path",
                             input_format="NHWC1C0",
                             input_shape="input_op_name1: n1,c2,h3,w4;input_op_name2: n4,c3,h2,w1",
                             output_type="FP16",
                             precision_mode="allow_mix_precision",
                             op_select_impl_mode="high_precision",
                             fusion_switch_config_path="some path of fusion_switch_config_path",
                             buffer_optimize_mode="l1_and_l2_optimize")
    acl_device_list = acl_options.context.model_context.device_list

    assert acl_device_list[0]["insert_op_cfg_path"] == "some path of insert_op_cfg_path"
    assert acl_device_list[0]["input_format"] == "NHWC1C0"
    assert acl_device_list[0]["input_shape"] == "input_op_name1: n1,c2,h3,w4;input_op_name2: n4,c3,h2,w1"
    assert acl_device_list[0]["output_type"] == "FP16"
    assert acl_device_list[0]["precision_mode"] == "allow_mix_precision"
    assert acl_device_list[0]["op_select_impl_mode"] == "high_precision"
    assert acl_device_list[0]["fusion_switch_config_path"] == "some path of fusion_switch_config_path"
    assert acl_device_list[0]["buffer_optimize_mode"] == "l1_and_l2_optimize"
    assert acl_device_list[0]["device_type"] == "ascend"


@serving_test
def test_model_context_gpu_device_info_serving_server_success():
    """
    Feature: Model Device info
    Description: Test set gpu device info
    Expectation: Serving server work well.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
from mindspore_serving.server.register import Context, GPUDeviceInfo, CPUDeviceInfo
from mindspore_serving.server.register import AscendDeviceInfo, GpuOptions, AclOptions

context = Context()
context.append_device_info(GPUDeviceInfo(precision_mode="fp16"))
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False,
                               context = context)

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(model, x1, x2, outputs_count=1)
    return y
    """
    os.environ["SERVING_ENABLE_GPU_DEVICE"] = "1"
    base = start_serving_server(servable_content, device_type="GPU")
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
def test_model_context_cpu_device_info_serving_server_success():
    """
    Feature: Model Device info
    Description: Test set cpu device info
    Expectation: Serving server work well.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
from mindspore_serving.server.register import Context, GPUDeviceInfo, CPUDeviceInfo
from mindspore_serving.server.register import AscendDeviceInfo, GpuOptions, AclOptions

context = Context()
context.append_device_info(CPUDeviceInfo(precision_mode="fp16"))
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False,
                               context = context)

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(model, x1, x2, outputs_count=1)
    return y
    """
    os.environ["SERVING_ENABLE_CPU_DEVICE"] = "1"
    base = start_serving_server(servable_content, device_type="CPU")
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
def test_model_context_ascend_device_info_serving_server_success():
    """
    Feature: Model Device info
    Description: Test set ascend device info
    Expectation: Serving server work well.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
from mindspore_serving.server.register import Context, GPUDeviceInfo, CPUDeviceInfo
from mindspore_serving.server.register import AscendDeviceInfo, GpuOptions, AclOptions

context = Context()
context.append_device_info(AscendDeviceInfo(input_format="NHWC1C0"))
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False,
                               context = context)

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(model, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, device_type="Ascend")
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
def test_model_context_all_device_info_serving_server_success():
    """
    Feature: Model Device info
    Description: Test set cpu, gpu, ascend device info, and serving select one device info based on inference so
    Expectation: Serving server work well.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
from mindspore_serving.server.register import Context, GPUDeviceInfo, CPUDeviceInfo
from mindspore_serving.server.register import AscendDeviceInfo, GpuOptions, AclOptions

context = Context()
context.append_device_info(AscendDeviceInfo(input_format="NHWC1C0"))
context.append_device_info(GPUDeviceInfo(precision_mode="fp16"))
context.append_device_info(CPUDeviceInfo(precision_mode="fp16"))
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False,
                               context = context)

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(model, x1, x2, outputs_count=1)
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
def test_model_context_acl_options_serving_server_success():
    """
    Feature: Model Device info
    Description: Test set ascend options
    Expectation: Serving server work well.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
from mindspore_serving.server.register import Context, GPUDeviceInfo, CPUDeviceInfo
from mindspore_serving.server.register import AscendDeviceInfo, GpuOptions, AclOptions

options = AclOptions(input_format="NHWC1C0")
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False,
                               options = options)

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(model, x1, x2, outputs_count=1)
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
def test_model_context_gpu_options_serving_server_success():
    """
    Feature: Model Device info
    Description: Test set gpu options
    Expectation: Serving server work well.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
from mindspore_serving.server.register import Context, GPUDeviceInfo, CPUDeviceInfo
from mindspore_serving.server.register import AscendDeviceInfo, GpuOptions, AclOptions

options = GpuOptions(precision_mode="fp16")
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False,
                               options = options)

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(model, x1, x2, outputs_count=1)
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
