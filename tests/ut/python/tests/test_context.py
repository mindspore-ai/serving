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

from common import serving_test
from mindspore_serving.server.register import Context, GPUDeviceInfo, CPUDeviceInfo
from mindspore_serving.server.register import Ascend310DeviceInfo, Ascend710DeviceInfo


@serving_test
def test_model_device_info_set_get_success():
    """
    Feature: Model Device info
    Description: Test set and get device info
    Expectation: the values gotten are equal to the values set.
    """
    context = Context(thread_num=3, thread_affinity_core_list=[1, 2, 3], enable_parallel=True)
    model_context = context.model_context
    assert model_context.thread_num == 3
    assert set(model_context.thread_affinity_core_list) == {1, 2, 3}
    assert model_context.enable_parallel

    # declare model and start_servable and load model and build model
    gpu_device_info = GPUDeviceInfo(precision_mode="fp16")
    gpu_map = gpu_device_info.context_map
    assert gpu_map["precision_mode"] == "fp16"
    assert gpu_map["device_type"] == "gpu"

    cpu_device_info = CPUDeviceInfo(precision_mode="fp16")
    cpu_map = cpu_device_info.context_map
    assert cpu_map["precision_mode"] == "fp16"
    assert cpu_map["device_type"] == "cpu"

    ascend310_device_info = Ascend310DeviceInfo(insert_op_cfg_path="some path of insert_op_cfg_path",
                                                input_format="NHWC1C0",
                                                input_shape="input_op_name1: n1,c2,h3,w4;input_op_name2: n4,c3,h2,w1",
                                                output_type="FP16",
                                                precision_mode="allow_mix_precision",
                                                op_select_impl_mode="high_precision",
                                                fusion_switch_config_path="some path of fusion_switch_config_path",
                                                buffer_optimize_mode="l1_and_l2_optimize")
    ascend310_map = ascend310_device_info.context_map
    assert ascend310_map["insert_op_cfg_path"] == "some path of insert_op_cfg_path"
    assert ascend310_map["input_format"] == "NHWC1C0"
    assert ascend310_map["input_shape"] == "input_op_name1: n1,c2,h3,w4;input_op_name2: n4,c3,h2,w1"
    assert ascend310_map["output_type"] == "FP16"
    assert ascend310_map["precision_mode"] == "allow_mix_precision"
    assert ascend310_map["op_select_impl_mode"] == "high_precision"
    assert ascend310_map["fusion_switch_config_path"] == "some path of fusion_switch_config_path"
    assert ascend310_map["buffer_optimize_mode"] == "l1_and_l2_optimize"
    assert ascend310_map["device_type"] == "ascend310"

    ascend710_device_info = Ascend710DeviceInfo(insert_op_cfg_path="some path of insert_op_cfg_path",
                                                input_format="NHWC1C0",
                                                input_shape="input_op_name1: n1,c2,h3,w4;input_op_name2: n4,c3,h2,w1",
                                                output_type="FP16",
                                                precision_mode="allow_mix_precision",
                                                op_select_impl_mode="high_precision",
                                                fusion_switch_config_path="some path of fusion_switch_config_path",
                                                buffer_optimize_mode="l1_and_l2_optimize")
    ascend710_map = ascend710_device_info.context_map
    assert ascend710_map["insert_op_cfg_path"] == "some path of insert_op_cfg_path"
    assert ascend710_map["input_format"] == "NHWC1C0"
    assert ascend710_map["input_shape"] == "input_op_name1: n1,c2,h3,w4;input_op_name2: n4,c3,h2,w1"
    assert ascend710_map["output_type"] == "FP16"
    assert ascend710_map["precision_mode"] == "allow_mix_precision"
    assert ascend710_map["op_select_impl_mode"] == "high_precision"
    assert ascend710_map["fusion_switch_config_path"] == "some path of fusion_switch_config_path"
    assert ascend710_map["buffer_optimize_mode"] == "l1_and_l2_optimize"
    assert ascend710_map["device_type"] == "ascend710"

    context.append_device_info(gpu_device_info)
    context.append_device_info(cpu_device_info)
    context.append_device_info(ascend310_device_info)
    context.append_device_info(ascend710_device_info)

    assert len(model_context.device_list) == 4
    assert model_context.device_list[0]["device_type"] == "gpu"
    assert model_context.device_list[1]["precision_mode"] == "fp16"
    assert model_context.device_list[2]["precision_mode"] == "allow_mix_precision"
    assert model_context.device_list[3]["input_format"] == "NHWC1C0"
