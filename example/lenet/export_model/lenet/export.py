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
"""export checkpoint file into air, onnx, mindir models"""

import os
import numpy as np
from easydict import EasyDict as ed
import mindspore
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export
from .src.lenet import LeNet5

config = ed({
    'num_classes': 10,
    'batch_size': 2,
    'image_height': 32,
    'image_width': 32
})

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
context.set_context(device_id=0)


def export_lenet():
    """define lenet network"""
    network = LeNet5(config.num_classes)
    # load network checkpoint
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    ckpt_file = os.path.join(cur_dir, 'lenet_ascend_v111_offical_cv_mnist_bs32_acc98.ckpt')
    param_dict = load_checkpoint(ckpt_file)
    load_param_into_net(network, param_dict)

    # export network
    inputs = Tensor(np.ones([config.batch_size, 1, config.image_height, config.image_width]), mindspore.float32)
    export(network, inputs, file_name="lenet", file_format="MINDIR")


if __name__ == "__main__":
    export_lenet()
