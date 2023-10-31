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

"""servable config for model"""
import os
from typing import List
import logging
import time


import numpy as np
from easydict import EasyDict

tokenizer_path = '/home/sc/gsp/LLM-Serving/checkpoint_download/llama/tokenizer.model'
device = 4


Baseconfig = EasyDict({
    'frequency_penalty': 1.5,
    'presence_penalty': 0.3,
    'max_generate_length': 300,
    'top_k_num': 100,
    'top_p': 1.0,
    'end_token': 2,
    'seq_length': [310, 600, 1024, 4096],    
    'vocab_size': 32000,
    'batch_size': 1,
    'model_type': 0,    # 0 is dyn-shape model, 1 is static-shape model
    'post_sampling': 0,  # 0 is argmax, 1 is topk + softmax
})

prefill_model_path = [
                    "/home/gch/mf/test_mf/mindir/dyn_pred_test/split/full/rank_0/split_net_graph.mindir",
                    "/home/gch/mf/test_mf/mindir/dyn_pred_test/split/full/rank_1/split_net_graph.mindir"
                     ]

decode_model_path = [
                    "/home/gch/mf/test_mf/mindir/dyn_pred_test/split/inc/rank_0/split_net_graph.mindir",
                    "/home/gch/mf/test_mf/mindir/dyn_pred_test/split/inc/rank_1/split_net_graph.mindir"
                    ]


prefill_model_4p = [
                    "/home/gch/mf/test_mf/mindir/dyn_predict_70b/split/full/rank_0/split_net_graph.mindir",
                    "/home/gch/mf/test_mf/mindir/dyn_predict_70b/split/full/rank_1/split_net_graph.mindir",
                    "/home/gch/mf/test_mf/mindir/dyn_predict_70b/split/full/rank_2/split_net_graph.mindir",
                    "/home/gch/mf/test_mf/mindir/dyn_predict_70b/split/full/rank_3/split_net_graph.mindir"
                     ]

decode_model_4p = [
                "/home/gch/mf/test_mf/mindir/dyn_predict_70b/split/inc/rank_0/split_net_graph.mindir",
                "/home/gch/mf/test_mf/mindir/dyn_predict_70b/split/inc/rank_1/split_net_graph.mindir",
                "/home/gch/mf/test_mf/mindir/dyn_predict_70b/split/inc/rank_2/split_net_graph.mindir",
                "/home/gch/mf/test_mf/mindir/dyn_predict_70b/split/inc/rank_3/split_net_graph.mindir"
                 ]

argmax_model = ["//home/sc/MindSpore_Serving/extends/argmax_post_calc_bz1.mindir"]

topk_model = ["/home/sc/MindSpore_Serving/extends/topk_post_calc_bz1.mindir"]

ctx_path_2p = '/home/gch/mf/test_mf/lite_config_ctx.ini'
inc_path_2p = '/home/gch/mf/test_mf/lite_config_inc.ini'

ctx_4p_path = f"/home/gch/mf/test_mf2/code/lite_config_ctx_4p_{device}{device + 1}{device + 2}{device + 3}.ini"
inc_4p_path = f'/home/gch/mf/test_mf2/code/lite_config_inc_4p_{device}{device + 1}{device + 2}{device + 3}.ini'
print(ctx_4p_path)
print(inc_4p_path)
AgentConfig = EasyDict({
    'ctx_setting': ctx_4p_path,
    'inc_setting': inc_4p_path,
    'post_model_setting': '/home/sc/MindSpore_Serving/extends/config.ini',
    'npu_nums': 4,
    'prefill_model': prefill_model_4p,
    'decode_model': decode_model_4p,
    'argmax_model': argmax_model,
    'topk_model': topk_model,
    'AgentPorts': [8900, 8901, 8902, 8903],
    'device_start': device
})

AgentIP: str = "localhost"

ModelName: str = "llama_dyn"

TOPP_NUM = 100

def get_token():
    from mindformers import LlamaTokenizer
    # preprocessor
    tokenizer = LlamaTokenizer(tokenizer_path)
    print("llama_7b")
    return tokenizer
    

def topk_fun(logits, topk=5):
    """Get topk"""
    print(logits, len(logits))
    topk_start_time = time.time()
    
    topk_getkv_time = time.time()
    sorted_array = [(k, v) for k, v in enumerate(logits)]
    logging.info('getkv time is {}'.format((time.time() - topk_getkv_time) * 1000))
    topk_sort_time = time.time()
    sorted_array.sort(key=lambda x: x[1], reverse=True)
    logging.info('sort time is {}'.format((time.time() - topk_start_time) * 1000))
    topk_array = sorted_array[:topk]
    print("topk arr", topk_array)
    zip_time = time.time()
    index, value = zip(*topk_array)
    logging.info('zip time is {}'.format((time.time() - topk_start_time) * 1000))
    ttt = time.time()
    print("index is {}".format(index))
    index = np.array([index])
    print("value is {}".format(value))
    value = np.array([value])
    logging.info('index_value is {}'.format((time.time() - ttt) * 1000))
    logging.info('topk time is {}'.format((time.time() - topk_start_time) * 1000))
    return value, index


def topk(x, top_k, axis=-1, largest=True, sort=True):
    """numpy implemented topk sample."""
    # safety check
    if x.shape[axis] < top_k:
        top_k = x.shape[axis] - 1
    if largest:
        topk_index = np.argpartition(-x, top_k, axis=axis)
    else:
        topk_index = np.argpartition(x, top_k, axis=axis)
    topk_index = np.take(topk_index, np.arange(top_k), axis=axis)
    topk_data = np.take_along_axis(x, topk_index, axis=axis)
    if sort:
        sort_index = (
            np.argsort(-topk_data, axis=axis)
            if largest
            else np.argsort(topk_data, axis=axis)
        )
        topk_data = np.take_along_axis(topk_data, sort_index, axis=axis)
        topk_index = np.take_along_axis(topk_index, sort_index, axis=axis)
    print('topk_data: ', topk_data)
    print('topk_index: ', topk_index)
    return topk_data, topk_index

def post_funciton(reply):
    # postprocessor
    return reply


if __name__ == "__main__":
    test_tokenizer_res()
