# -*- coding: utf-8 -*-
import argparse
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore import ops, nn, export
from mindspore import Tensor, context
import mindspore as ms
import numpy as np

bs = 1
default_temperature = Tensor(np.array([1]), ms.float32)

class temperature_TopK(nn.Cell):
    def __init__(self):
        super(temperature_TopK, self).__init__()
        self.divide = P.Div()
        self.topk = P.TopK(sorted=True)
        self.top_k_num = 100
        # self.reshape = ops.reshape()

    def construct(self, x, temperature=default_temperature):
        x = ops.reshape(x, (x.shape[0], x.shape[-1]))
        x = self.divide(x, temperature)
        logit, pargs = self.topk(x, self.top_k_num)
        return logit, pargs


class ArgmaxPost(nn.Cell):
    def __init__(self):
        super(ArgmaxPost, self).__init__()
        self.argmax = ops.Argmax(output_type=ms.int32)
        # self.reshape = ops.reshape()

    def construct(self, x):
        x = ops.reshape(x, (x.shape[0], x.shape[-1]))
        output = self.argmax(x)
        return output


def run_mindspore(output_dir):
    if output_dir is None or output_dir == "":
        raise ValueError("invalid output_dir, got {}".format(output_dir))
    context.set_context(
        save_graphs=False,
        mode=context.GRAPH_MODE,
        device_target="Ascend",
        pynative_synchronize=False,
        device_id=0
    )
    topk_topp_ = temperature_TopK()
    argmax_ = ArgmaxPost()

    input_ids_dyn = Tensor(shape=[None, None, None], dtype=mstype.float16)
    temperature_ = Tensor(shape=[None, ], dtype=mstype.float32)

    export(topk_topp_, input_ids_dyn, temperature_, file_name=(output_dir + "/topk.mindir"), file_format='MINDIR')
    export(argmax_, input_ids_dyn, file_name=(output_dir + "/argmax.mindir"), file_format='MINDIR')
    print("export finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        required=True,
        help='output dir for exporting post_sampling models')
    args = parser.parse_args()
    run_mindspore(args.output_dir)
