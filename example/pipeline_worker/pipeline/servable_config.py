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
"""add model servable config"""

from mindspore_serving.server import register
from mindspore_serving.server.register import PipelineServable

# when with_batch_dim is set to False, only 2x2 add is supported
# when with_batch_dim is set to True(default), Nx2 add is supported, while N is viewed as batch
# float32 inputs/outputs
register.declare_servable(servable_file=["tensor_add.mindir", "tensor_sub.mindir"],
                          model_format="MindIR", with_batch_dim=False)


# register add_common method in add
@register.register_method(output_names=["y"])
def add_common(x1, x2):  # only support float32 inputs
    """method add_common data flow definition, only call model servable"""
    y = register.call_servable(x1, x2, subgraph=0)
    return y


@register.register_method(output_names=["y"])
def sub_common(x1, x2):  # only support float32 inputs
    """method sub_common data flow definition, only call model servable"""
    y = register.call_servable(x1, x2, subgraph=1)
    return y


servable1 = PipelineServable(servable_name="pipeline", method="add_common", version_number=1)
servable2 = PipelineServable(servable_name="pipeline", method="sub_common", version_number=1)


@register.register_pipeline(output_names=["y1", "y2"])
def predict(x1, x2):
    y1 = servable1.run(x1, x2)
    y2 = servable2.run(x1, x2)
    return y1, y2
