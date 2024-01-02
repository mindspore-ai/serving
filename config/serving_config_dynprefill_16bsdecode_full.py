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
# import models.tokenizer
from serving_utils.register import import_all_modules_for_register


device = 0


import_all_modules_for_register()

SERVER_APP_HOST = 'localhost'

SERVER_APP_PORT = 61116

prefill_model_path = [
    # "/home/renyujin/ft-predict-opt/mindformers/research/baichuan2/baichuan2_13b_bs1_seq1500_pa224/baichuan2_13b_prefill_graph.mindir"
    # "/home/renyujin/ft-predict-opt/mindformers/research/baichuan2/baichuan2_13b_bs1_seq1500_pa224_2l/baichuan2_13b_prefill_graph.mindir"
    # "/home/renyujin/ft-predict-opt/mindformers/research/baichuan2/baichuan2_13b_bs1_dyn_pa224_2l/baichuan2_13b_prefill_graph.mindir"
    "/home/renyujin/ft-predict-opt/mindformers/research/baichuan2/baichuan2_13b_bs1_dyn_pa224/baichuan2_13b_prefill_graph.mindir"
]

decode_model_path = [
    # "/home/renyujin/ft-predict-opt/mindformers/research/baichuan2/baichuan2_13b_bs16_seq4096_pa224/baichuan2_13b_inc_graph.mindir"
    # "/home/renyujin/ft-predict-opt/mindformers/research/baichuan2/baichuan2_13b_bs16_seq4096_pa224_2l/baichuan2_13b_inc_graph.mindir",
    "/home/renyujin/ft-predict-opt/mindformers/research/baichuan2/baichuan2_13b_bs16_seq4096_pa224/baichuan2_13b_inc_graph.mindir"
]

argmax_model = ["extends/argmax.mindir"]
topk_model = ["extends/topk.mindir"]
# ctx_path = '/mnt/disk0/z00838343/baichuan_pa/configs/full.ini'
ctx_path = "/mnt/disk0/z00838343/baichuan_pa/configs/910b_default_prefill.cfg"
inc_path = ['/mnt/disk0/z00838343/baichuan_pa/configs/inc.ini']

post_model_ini = '/mnt/disk0/z00838343/baichuan_pa/configs/baichuan2/config.ini'
tokenizer_path = '/mnt/disk0/z00838343/baichuan_pa/2_layer_model/tokenizer.model'

Baseconfig = EasyDict({
    'frequency_penalty': 1.5,
    'presence_penalty': 0.3,
    'max_generate_length': 4096,
    'max_top_k': 500,
    'top_k_num': 100,
    'top_p': 1.0,
    'end_token': 2,
    'decode_seq_length': 4096,
    'seq_length': [1500, 2048, 4096],
    'vocab_size': 125696,
    'batch_size': 16,
    'dyn_batch_size': [16],
    'prefill_batch_size': 1,
    'model_type': 1,  # 0 is dyn-shape model, 1 is static-shape model
    'batch_waiting_time': 0.0,
    # 'seq_type': 'dyn',  # 'dyn'
    'decode_batch_waiting_time': 0.00,
    'batching_strategy': 'continuous',  # static
    'tokenizer': 'Baichuan2Tokenizer',  # if import tokenizer, setting None # InternLMTokenizer for internlm
    'tokenizer_path': tokenizer_path,
    'input_function': 'common',  # for interNLM : common
    'zactivate_len':  [4096],
    'slice_model' : True
})

AgentConfig = EasyDict({
    'ctx_setting': ctx_path,
    'inc_setting': inc_path,
    'post_model_setting': post_model_ini,
    'prefill_model': prefill_model_path,
    'decode_model': decode_model_path,
    'argmax_model': argmax_model,
    'topk_model': topk_model,
    'AgentPorts': [61117],
    'device_start': device,
    "enable_host_post_sampling": False
})

AgentIP: str = "localhost"
ModelName: str = "baichuan2pa" # "baichuan2"  # internlm_7b for internlm


def llama_inputs_for_warmup(seq_length, batch_size, full_model, valid_length=None):
    input_ids = np.ones([batch_size, seq_length], dtype=np.int32)
    current_index = np.array([1] * batch_size, dtype=np.int32)

    if full_model:
        init_reset = np.array([False] * 1, dtype=np.bool_)
    else:
        init_reset = np.array([True] * 1, dtype=np.bool_)

    if valid_length is None:
        batch_valid_length = np.array([1] * batch_size, dtype=np.int32)
    else:
        batch_valid_length = np.array(valid_length * batch_size, dtype=np.int64)

    if Baseconfig['batching_strategy'] == 'continuous':
        decode_index = np.array(range(batch_size), dtype=np.int64)
        inputs_list = [input_ids, current_index, batch_valid_length, decode_index]
    else:
        inputs_list = [input_ids, current_index, init_reset, batch_valid_length]

    input_extra_list = ExtraInput(input_ids, current_index, init_reset, full_model, batch_valid_length)
    inputs_list.extend(input_extra_list)
    return inputs_list


def baichuan2_inputs_for_warmup(seq_length, batch_size, full_model, valid_length=None):
    input_ids = np.ones([batch_size, seq_length], dtype=np.int32)
    current_index = np.array([1] * batch_size, dtype=np.int32)

    if full_model:
        init_reset = np.array([False] * 1, dtype=np.bool_)
    else:
        init_reset = np.array([True] * 1, dtype=np.bool_)

    if valid_length is None:
        batch_valid_length = np.array([1] * batch_size, dtype=np.int32)
    else:
        batch_valid_length = np.array(valid_length * batch_size, dtype=np.int32)

    if Baseconfig['batching_strategy'] == 'continuous':
        decode_index = np.array(range(batch_size), dtype=np.int64)
        inputs_list = [input_ids, current_index, batch_valid_length, decode_index]
    else:
        batch_index = np.array([128] * batch_size, dtype=np.int32)
        inputs_list = [input_ids, current_index, init_reset, batch_valid_length, batch_index]

    input_extra_list = ExtraInput(input_ids, current_index, init_reset, full_model, batch_valid_length)
    # inputs_list.extend(input_extra_list)

    print('inputs_list: ', inputs_list, flush=True)
    return inputs_list

def baichuan2_pa_inputs_for_warmup(seq_length, batch_size, full_model, valid_length=None):
    # block_size = 128
    # max_num_blocks_pre_seq = seq_length // block_size
    # for i in range(batch_size):
    #     block_table = self.cache_engines[i].block_table[1:]
    #     padded_table = block_table + [ -1 for _ in range(self.max_num_blocks_per_seq - len(self.cache_engines[i].block_table) + 1)]
    # export.py:
    # input_ids, input_position, init_reset, batch_valid_length, batch_index, block_tables, slot_mapping
    full_seq_len = Baseconfig.seq_length[-1]
    inc_seq_len = PageAttentionConfig.decode_seq_length
    current_index = np.array([0] * batch_size, dtype=np.int32)
    if full_model:
        input_ids = np.zeros([batch_size, full_seq_len], dtype=np.int32)
        for i in range(batch_size):
            input_ids[i][0] = 16829
        init_reset = np.array([False] * 1, dtype=np.bool_)
        slot_mapping = np.full((full_seq_len,), 0, dtype=np.int32)
        slot_mapping[0] = 128
    else:
        input_ids = np.full([batch_size, 1], 16829, dtype=np.int32)
        init_reset = np.array([True] * 1, dtype=np.bool_)
        slot_mapping = np.full((batch_size,), 128, dtype=np.int32)
    # if valid_length is None:
    #     batch_valid_length = np.array([1] * batch_size, dtype=np.int32)
    # else:
    #     batch_valid_length = np.array(valid_length * batch_size, dtype=np.int32)
    batch_valid_length = np.array([1] * batch_size, dtype=np.int32)
    batch_index = np.array(list(range(batch_size)), dtype=np.int32)
    block_tables = np.full((batch_size, inc_seq_len // 128), -1, dtype=np.int32)
    logging.debug("block tables shape: %s, seq_length is: %s", block_tables.shape, inc_seq_len)
    for i in range(batch_size):
        block_tables[i][0] = 1
    if full_model:
        inputs_list = [input_ids, current_index, init_reset, batch_valid_length, batch_index, slot_mapping]
    else:
        inputs_list = [input_ids, current_index, init_reset, batch_valid_length, batch_index, block_tables, slot_mapping]
    for item in inputs_list:
        logging.debug("baichuan2pa warm up inputs: %s, %s", item, item.shape)
    print('baichuan2 pa inputs_list: ', inputs_list, flush=True)
    return inputs_list

def internlm_inputs_for_warmup(seq_length, batch_size, full_model):
    input_ids = np.ones([batch_size, seq_length], dtype=np.int32)
    current_index = np.array([1] * batch_size, dtype=np.int32)

    if full_model:
        init_reset = np.array([False] * batch_size, dtype=np.bool_)
    else:
        init_reset = np.array([True] * batch_size, dtype=np.bool_)
    batch_valid_length = np.array([1] * batch_size, dtype=np.int32)

    if Baseconfig['batching_strategy'] == 'continuous':
        decode_index = np.array(range(batch_size), dtype=np.int64)
        inputs_list = [input_ids, decode_index, current_index, init_reset, batch_valid_length]
    else:
        inputs_list = [input_ids, current_index, init_reset, batch_valid_length]

    return inputs_list


WARMUP_MODEL_INPUTS_MAP = {
    "llama": llama_inputs_for_warmup,
    "internlm": llama_inputs_for_warmup,    # 和llama一致
    "baichuan2": baichuan2_inputs_for_warmup,    # 和llama一致
    "baichuan2pa": baichuan2_pa_inputs_for_warmup
}


def get_warmup_inputs(seq_length=Baseconfig.seq_length[0] if len(Baseconfig.seq_length) >= 1 else 2048,
                      batch_size=Baseconfig.batch_size,
                      full_model=True,
                      model_name=ModelName, valid_length=None):
    model_prefix = model_name.split('_')[0]
    if model_prefix in WARMUP_MODEL_INPUTS_MAP.keys():
        func = WARMUP_MODEL_INPUTS_MAP[model_prefix]
        return func(seq_length, batch_size, full_model, valid_length)
    else:
        print("model not support warmup : ", model_name)


def get_inputs_custom(input_ids=None, current_index=None, valid_length=None,
                      init_reset=None, is_first_iteration=True, InputExtraList=[], **kwargs):
    act_len = InputExtraList[0]
    if not is_first_iteration:
        inputs_tmp = []
        for i in range(len(current_index)):
            current_index_tmp = int(current_index[i]) - i * input_ids.shape[1]  # multibatch
            inputs_tmp.append(input_ids[i][current_index_tmp:current_index_tmp + 1])
        input_ids = np.array(inputs_tmp, dtype=np.int32)
    inputs = [input_ids, current_index, init_reset, valid_length, act_len]

    return inputs


def ExtraInput(input_ids, current_index, init_reset, is_prefill, valid_length, **kwargs):
    def get_act_length(seq_len, act_len_list):
        for seq in act_len_list:
            if seq_len <= seq:
                act_len = np.zeros((seq), np.int64)
                break
            act_len = np.zeros((act_len_list[-1]), np.int64)
        # return act_len
        return []

    if not is_prefill:
        max_seq = 0
        for i in range(len(valid_length)):
            max_seq = max(max_seq, valid_length[i] + 1)
        return [get_act_length(max_seq, Baseconfig.zactivate_len)]
    max_prefill_length = 0
    for item in valid_length:
        max_prefill_length = max(max_prefill_length, item)
    act_len = get_act_length(max_prefill_length + 1, Baseconfig.zactivate_len)
    # return [act_len]
    return []
