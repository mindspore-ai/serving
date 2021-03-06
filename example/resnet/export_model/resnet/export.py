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
"""
##############export checkpoint file into air and onnx models#################
python export.py
"""
import argparse
import numpy as np

from mindspore import Tensor, export
from mindspore import load_checkpoint, load_param_into_net


def export_resnet(network_dataset, ckpt_file, output_file):
    """export resnet"""

    if network_dataset == 'resnet50_cifar10':
        from .src.config import config1 as config
        from .src.resnet import resnet50 as resnet
    elif network_dataset == 'resnet50_imagenet2012':
        from .src.config import config2 as config
        from .src.resnet import resnet50 as resnet
    elif network_dataset == 'resnet101_imagenet2012':
        from .src.config import config3 as config
        from .src.resnet import resnet101 as resnet
    elif network_dataset == 'se-resnet50_imagenet2012':
        from .src.config import config4 as config
        from .src.resnet import se_resnet50 as resnet
    else:
        raise ValueError("network and dataset is not support.")

    net = resnet(config.class_num)

    if ckpt_file is not None:
        param_dict = load_checkpoint(ckpt_file)
        load_param_into_net(net, param_dict)

    input_arr = Tensor(np.zeros([1, 3, 224, 224], np.float32))
    export(net, input_arr, file_name=output_file, file_format="MINDIR")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='resnet export')
    parser.add_argument('--network_dataset', type=str, default='resnet50_cifar10', choices=['resnet50_cifar10',
                                                                                            'resnet50_imagenet2012',
                                                                                            'resnet101_imagenet2012',
                                                                                            "se-resnet50_imagenet2012"],
                        help='network and dataset name.')
    parser.add_argument('--ckpt_file', type=str, default='', help='resnet ckpt file.')
    parser.add_argument('--output_file', type=str, default='', help='resnet output air name.')
    args_opt = parser.parse_args()
    export_resnet(args_opt.network_dataset, args_opt.ckpt_file, args_opt.output_file)
