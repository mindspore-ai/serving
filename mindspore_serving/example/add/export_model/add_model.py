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
"""add model generator"""

import os
from shutil import copyfile
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore.train.serialization import export

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.TensorAdd()

    def construct(self, x_, y_):
        return self.add(x_, y_)


def export_net():
    x = np.ones([2, 2]).astype(np.float32)
    y = np.ones([2, 2]).astype(np.float32)
    add = Net()
    output = add(Tensor(x), Tensor(y))
    export(add, Tensor(x), Tensor(y), file_name='tensor_add', file_format='MINDIR')
    dst_dir = '../add/1'
    try:
        os.mkdir(dst_dir)
    except OSError:
        pass
    try:
        dst_file = os.path.join(dst_dir, 'tensor_add.mindir')
        if os.path.exists('tensor_add.mindir'):
            copyfile('tensor_add.mindir', dst_file)
            print("copy tensor_add.mindir to " + dst_dir + " success")
        elif os.path.exists('tensor_add'):
            copyfile('tensor_add', dst_file)
            print("copy tensor_add to " + dst_dir + " success")
    except:
        print("copy tensor_add.mindir to " + dst_dir + " failed")

    print(x)
    print(y)
    print(output.asnumpy())


if __name__ == "__main__":
    export_net()
