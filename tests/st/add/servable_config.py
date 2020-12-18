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

import numpy as np
from mindspore_serving.worker import register


def add_trans_datatype(x1, x2):
    """define preprocess, this example has one input and one output"""
    return x1.astype(np.float32), x2.astype(np.float32)


# when with_batch_dim set to False, only support 2x2 add
# when with_batch_dim set to True(default), support Nx2 add, while N is view as batch
# float32 inputs/outputs
register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)


# register add_common method in add
@register.register_method(output_names=["y"])
def add_common(x1, x2):  # only support float32 inputs
    """method add_common data flow definition, only call model servable"""
    y = register.call_servable(x1, x2)
    return y


# register add_cast method in add
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    """method add_cast data flow definition, only call preprocess and model servable"""
    x1, x2 = register.call_preprocess(add_trans_datatype, x1, x2)  # cast input to float32
    y = register.call_servable(x1, x2)
    return y
