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
'''net
The sample can be run on Ascend 910 AI processor.
'''
import os
from shutil import copyfile
import numpy as np
import mindspore.context as context
from mindspore import Tensor, Parameter, ops, export
from mindspore.nn import Cell


class Net(Cell):
    """Net"""

    def __init__(self, matmul_size, init_val, transpose_a=False, transpose_b=False):
        """init"""
        super().__init__()
        matmul_np = np.full(matmul_size, init_val, dtype=np.float32)
        self.matmul_weight = Parameter(Tensor(matmul_np))
        self.matmul = ops.MatMul(transpose_a=transpose_a, transpose_b=transpose_b)
        self.sum = ops.ReduceSum()

    def construct(self, inputs):
        """construct"""
        x = self.matmul(inputs, self.matmul_weight)
        x = self.sum(x, 0)
        return x


def export_net():
    """Export matmul net , and copy output model `matmul_0.mindir` and `matmul_1.mindir` to directory ../matmul/1"""
    context.set_context(mode=context.GRAPH_MODE)
    network = Net(matmul_size=(96, 16), init_val=0.5)
    # subgraph 0: 128,96 matmul 16,96 -> 128,16 reduce sum axis 0-> 16
    predict_data = np.random.randn(128, 96).astype(np.float32)
    # pylint: disable=protected-access
    export(network, Tensor(predict_data), file_name="matmul_0", file_format="MINDIR")

    # subgraph 1: 8,96 matmul 16,96 -> 8,16 reduce sum axis 0-> 16
    predict_data = np.random.randn(8, 96).astype(np.float32)
    # pylint: disable=protected-access
    export(network, Tensor(predict_data), file_name="matmul_1", file_format="MINDIR")

    dst_dir = '../matmul/1'
    try:
        os.mkdir(dst_dir)
    except OSError:
        pass

    dst_file = os.path.join(dst_dir, 'matmul_0.mindir')
    copyfile('matmul_0.mindir', dst_file)
    print("copy matmul_0.mindir to " + dst_dir + " success")

    dst_file = os.path.join(dst_dir, 'matmul_1.mindir')
    copyfile('matmul_1.mindir', dst_file)
    print("copy matmul_1.mindir to " + dst_dir + " success")


if __name__ == "__main__":
    export_net()
