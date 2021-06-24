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
'''distributed inference
The sample can be run on Ascend 910 AI processor.
'''
import numpy as np
from net import Net
from mindspore import context, Model, Tensor, export
from mindspore.communication import init


def test_inference():
    """distributed inference after distributed training"""
    context.set_context(mode=context.GRAPH_MODE)
    init(backend_name="hccl")
    context.set_auto_parallel_context(full_batch=True, parallel_mode="semi_auto_parallel",
                                      device_num=8, group_ckpt_save_file="./group_config.pb")

    predict_data = create_predict_data()
    network = Net(matmul_size=(96, 16), init_val=0.5)
    model = Model(network)
    model.infer_predict_layout(Tensor(predict_data))
    # pylint: disable=protected-access
    export(model._predict_network, Tensor(predict_data), file_name="matmul_0", file_format="MINDIR")

    network_1 = Net(matmul_size=(96, 16), init_val=1.5)
    model_1 = Model(network_1)
    model_1.infer_predict_layout(Tensor(predict_data))
    # pylint: disable=protected-access
    export(model_1._predict_network, Tensor(predict_data), file_name="matmul_1", file_format="MINDIR")

def create_predict_data():
    """user-defined predict data"""
    inputs_np = np.random.randn(128, 96).astype(np.float32)
    return Tensor(inputs_np)
