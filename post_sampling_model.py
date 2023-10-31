# -*- coding: utf-8 -*-
import numpy as np
import os
import json

import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore import ops, nn, export
from mindspore import Tensor, context
import mindspore

from config.serving_config import Baseconfig

bs = Baseconfig.batch_size
vocab_size = Baseconfig.vocab_size
top_k_num = Baseconfig.top_k_num


class temperature_TopK(nn.Cell):
    def __init__(self):
        super(temperature_TopK, self).__init__()
        self.divide = P.Div()
        self.topk = P.TopK(sorted=True)
        self.topk_num = top_k_num
        self.softmax = nn.Softmax()

    def construct(self, x, temperature, topk_n):
        x = self.divide(x, temperature)

        logit, pargs = self.topk(x, topk_n)
        logit = self.softmax(logit)
        return logit, pargs


class ArgmaxPost(nn.Cell):
    def __init__(self):
        super(ArgmaxPost, self).__init__()
        self.argmax = ops.Argmax(output_type=mindspore.int32)

    def construct(self, x):
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
    inputs_np = Tensor(np.random.rand(bs, vocab_size).astype(np.float16), mstype.float16)

    temperature_ = Tensor(np.tile(np.array([0.7], np.float16), (bs,)), mstype.float16)

    export(topk_topp_, inputs_np, temperature_, file_name=f"extends/topk_post_calc_bz{bs}", file_format='MINDIR')
    export(argmax_, inputs_np, file_name=f"extends/argmax_post_calc_bz{bs}", file_format='MINDIR')
    print("export finished")


def run_mslite():
    import mindspore_lite as mslite
    context = mslite.Context()
    context.ascend.device_id = 0
    context.ascend.provider = "ge"

    context.target = ["Ascend"]

    model = mslite.Model()
    model.build_from_file("/home/sc/gsp/LLM-Serving/extends/1018_topk_post_calc_bz1_k100.mindir", mslite.ModelType.MINDIR, context)
    inputs = model.get_inputs()
    print('inputs: ', inputs)
    inputs_np = np.random.rand(bs, vocab_size).astype(np.float16)

    temperature_ = np.tile(np.array([0.7], np.float16), (bs,))
    inputs[0].set_data_from_numpy(inputs_np)
    inputs[1].set_data_from_numpy(temperature_)
    print('inputs: ', inputs)
    samples = model.predict(inputs)
    print(model.get_inputs())
    print("========================",len(samples))


if __name__ == '__main__':
    run_mindspore()    
