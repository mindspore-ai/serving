# -*- coding: utf-8 -*-
import numpy as np
import os
import json
import sys

import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore import ops, nn, export
from mindspore import Tensor, context
import mindspore as ms

from config.serving_config import *



bs = 1
vocab_size = Baseconfig.vocab_size
top_k_num = Baseconfig.top_k_num


class temperature_TopK(nn.Cell):
    def __init__(self):
        super(temperature_TopK, self).__init__()
        self.divide = P.Div()
        self.topk = P.TopK(sorted=True)
        self.softmax = nn.Softmax()
        self.top_k_num = 100
        # self.reshape = ops.reshape()

    def construct(self, x, temperature):
        x = ops.reshape(x, (x.shape[0], x.shape[-1]))
        x = self.divide(x, temperature)
        logit, pargs = self.topk(x, self.top_k_num)
        logit = self.softmax(logit)
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


def run_mindspore():
    context.set_context(
        save_graphs=False,
        mode=context.GRAPH_MODE,
        device_target="Ascend",
        pynative_synchronize=False,
    )
    topk_topp_ = temperature_TopK()
    argmax_ = ArgmaxPost()

    input_ids_dyn = Tensor(shape=[None, None, None], dtype=mstype.float16)
    temperature_ = Tensor(shape=[None, ], dtype=mstype.float32)

    export(topk_topp_, input_ids_dyn, temperature_, file_name=argmax_model[0], file_format='MINDIR')
    export(argmax_, input_ids_dyn, file_name=topk_model[0], file_format='MINDIR')
    print("export finished")


def run_mslite():
    import mindspore_lite as mslite
    context = mslite.Context()
    context.ascend.device_id = 0
    context.ascend.provider = "ge"

    context.target = ["Ascend"]

    model = mslite.Model()
    model.build_from_file("./extends/dyn_batch_argmax_post_model.mindir", mslite.ModelType.MINDIR, context)
    inputs = model.get_inputs()
    print('inputs: ', inputs)
    inputs_np = np.random.rand(bs, vocab_size).astype(np.float16)

    temperature_ = np.tile(np.array([0.7], np.float16), (bs,))

    inputs[0].set_data_from_numpy(inputs_np)
    inputs[1].set_data_from_numpy(temperature_)
    print('inputs: ', inputs)
    samples = model.predict(inputs)
    print(model.get_inputs())
    print("========================", len(samples))


if __name__ == '__main__':
    run_mindspore()
    #run_mslite()
