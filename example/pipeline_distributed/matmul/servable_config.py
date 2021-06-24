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
"""Distributed matmul config python file"""
import numpy as np
from mindspore_serving.server import distributed
from mindspore_serving.server import register
from mindspore_serving.server.register import PipelineServable

distributed.declare_servable(rank_size=8, stage_size=1, with_batch_dim=False)


def add_preprocess(x):
    """define preprocess, this example has one input and one output"""
    x = np.add(x, x)
    return x


@register.register_method(output_names=["y"])
def fun1(x):
    x = register.call_preprocess(add_preprocess, x)
    y = register.call_servable(x, subgraph=0)
    return y


@register.register_method(output_names=["y"])
def fun2(x):
    y = register.call_servable(x, subgraph=1)
    return y


servable1 = PipelineServable(servable_name="matmul", method="fun1", version_number=0)
servable2 = PipelineServable(servable_name="matmul", method="fun2", version_number=0)


@register.register_pipeline(output_names=["x", "z"])
def predict(x, y):
    x = servable1.run(x)
    for i in range(10):
        print(i)
        z = servable2.run(y)
    return x, z
