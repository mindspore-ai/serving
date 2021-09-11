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
from mindspore_serving.server import register

model = register.declare_model(model_file=["matmul_0.mindir", "matmul_1.mindir"], model_format="MindIR",
                               with_batch_dim=False)


def process(x, y):
    z1 = model.call(x, subgraph=0)  # 128,96 matmul 16,96 -> reduce sum axis 0-> 16
    z2 = model.call(y, subgraph=1)  # 8,96 matmul 16,96 -> reduce sum axis 0-> 16
    return z1 + z2


@register.register_method(output_names=["z"])
def predict(x, y):
    z = register.add_stage(process, x, y, outputs_count=1)
    return z
