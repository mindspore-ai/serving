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
from utils.register import import_all_modules_for_register


device = 4


import_all_modules_for_register()

SERVER_APP_HOST = 'localhost'

SERVER_APP_PORT = 19285

prefill_model_path = [
    "/path/to/prefill_graph.mindir"
]
decode_model_path = [
    "/path/to/inc_graph.mindir"
]
argmax_model = ["/path/to/argmax.mindir"]
topk_model = ["/path/to/topk.mindir"]
ctx_path = '/path/to/prompt_config.ini'
inc_path = '/path/to/decoder_config.ini'
post_model_ini = '/path/to/post_model.ini'
tokenizer_path = '/path/to/tokenizer.model'

Baseconfig = EasyDict({
    'frequency_penalty': 1.5,
    'presence_penalty': 0.3,
    'max_generate_length': 4096,
    'max_top_k': 500,
    'top_k_num': 100,
    'top_p': 1.0,
    'end_token': 2,
    'seq_length': [310, 600, 1024, 2048],
    'vocab_size': 32000,
    'batch_size': 8,
    'dyn_batch_size': [1, 2, 4, 8],
    'prefill_batch_size': 1,
    'model_type': 0,  # 0 is dyn-shape model, 1 is static-shape model
    'batch_waiting_time': 0.0,
    'seq_type': 'dyn',  # 'dyn'
    'decode_batch_waiting_time': 0.00,
    'batching_strategy': 'continuous',
    'tokenizer': 'LlamaTokenizer',  # if import tokenizer, setting None # InternLMTokenizer for internlm
    'tokenizer_path': tokenizer_path,
    'input_function': 'common' # for interNLM : common
})

AgentConfig = EasyDict({
    'ctx_setting': ctx_path,
    'inc_setting': inc_path,
    'post_model_setting': post_model_ini,
    'prefill_model': prefill_model_path,
    'decode_model': decode_model_path,
    'argmax_model': argmax_model,
    'topk_model': topk_model,
    'AgentPorts': [6877, 6878],
    'device_start': device
})

AgentIP: str = "localhost"
ModelName: str = "llama_dyn"  # internlm_7b for internlm


def llama_inputs_for_warmup(seq_length, batch_size, full_model):
    input_ids = np.ones([batch_size, seq_length], dtype=np.int32)
    current_index = np.array([1] * batch_size, dtype=np.int32)

    if full_model:
        init_reset = np.array([False] * batch_size, dtype=np.bool)
    else:
        init_reset = np.array([True] * batch_size, dtype=np.bool)

    batch_valid_length = np.array([1] * batch_size, dtype=np.int64)

    if Baseconfig['batching_strategy'] == 'continuous':
        decode_index = np.array(range(batch_size), dtype=np.int64)
        inputs_list = [input_ids,  current_index, batch_valid_length, decode_index]
    else:
        inputs_list = [input_ids, current_index, init_reset, batch_valid_length]

    return inputs_list


def internlm_inputs_for_warmup(seq_length, batch_size, full_model):
    input_ids = np.ones([batch_size, seq_length], dtype=np.int32)
    current_index = np.array([1] * batch_size, dtype=np.int32)

    if full_model:
        init_reset = np.array([False] * batch_size, dtype=np.bool)
    else:
        init_reset = np.array([True] * batch_size, dtype=np.bool)

    batch_valid_length = np.array([1] * batch_size, dtype=np.int32)

    if Baseconfig['batching_strategy'] == 'continuous':
        decode_index = np.array(range(batch_size), dtype=np.int64)
        inputs_list = [input_ids, decode_index, current_index, init_reset, batch_valid_length]
    else:
        inputs_list = [input_ids, current_index, init_reset, batch_valid_length]

    return inputs_list


WARMUP_MODEL_INPUTS_MAP = {
    "llama": llama_inputs_for_warmup,
    "internlm": internlm_inputs_for_warmup,
}


def get_warmup_inputs(seq_length=Baseconfig.seq_length[0] if len(Baseconfig.seq_length) >= 1 else 2048,
                      batch_size=Baseconfig.batch_size,
                      full_model=True,
                      model_name=ModelName):
    model_prefix = model_name.split('_')[0]
    if model_prefix in WARMUP_MODEL_INPUTS_MAP.keys():
        func = WARMUP_MODEL_INPUTS_MAP[model_prefix]
        return func(seq_length, batch_size, full_model)
    else:
        print("model not support warmup : ", model_name)


def get_inputs_custom(input_ids=None, current_index=None, valid_length=None,
                   init_reset=None, is_first_iteration=True, InputExtraList=[], **kwargs):
        mask = InputExtraList[0]
        freq_cos = InputExtraList[1]
        freq_sin = InputExtraList[2]
        if not is_first_iteration:
            inputs_tmp = []
            for i in range(len(current_index)):
                current_index_tmp = int(current_index[i]) - i * input_ids.shape[1]  # multibatch
                inputs_tmp.append(input_ids[i][current_index_tmp:current_index_tmp + 1])
            input_ids = np.array(inputs_tmp, dtype=np.int32)
        if is_first_iteration:
            inputs = [input_ids, current_index, init_reset, valid_length, mask, freq_cos, freq_sin]
        else:
            inputs = [input_ids, current_index, init_reset, valid_length]
        return inputs


def ExtraInput(input_ids, current_index, init_reset, is_prefill, valid_length, **kwargs):
    from enum import Enum
    class SeqExtendMethod(Enum):
        """Stores the acceptable string identifiers for seq length extend method"""
        PI = "PI"
        NTK = "NTK"
        NONE = "None"
    
    def precompute_freqs_cis(
        dim: int,
        end: int,
        real_seqlen: int,
        theta: float = 10000.0,
        pretrain_seqlen=2048,
        extend_method=SeqExtendMethod.NONE.value):
        """
        Precompute of freqs and mask for rotary embedding.
        """
        ratio = 1.
        if extend_method != SeqExtendMethod.NONE.value and end > pretrain_seqlen:
            ratio = end / pretrain_seqlen
        if extend_method == SeqExtendMethod.NTK.value:
            theta *= ratio
        freqs_base = np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) # (head_dim // 2, )
        freqs = 1.0 / (theta ** (freqs_base / dim)) # (head_dim // 2, )
        if extend_method == SeqExtendMethod.PI.value:
            t = np.arange(0, end / ratio, 1 / ratio).astype(np.float32)
        else:
            t = np.arange(0, end, 1).astype(np.float32)  # type: ignore # (seq_len,)
        freqs = np.outer(t, freqs)  # type: ignore (seq_len, head_dim // 2)
        emb = np.concatenate((freqs, freqs), axis=-1)
        freqs_cos = np.cos(emb)
        freqs_sin = np.sin(emb)

        return freqs_cos[:real_seqlen,:], freqs_sin[:real_seqlen,:]

    def get_mask(input_ids):
        input_mask = np.not_equal(input_ids, 0)
        seq_length = input_ids.shape[1]
        input_shape = np.shape(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)
        # Mask the padded inputs
        mask_left = np.reshape(input_mask, shape_left)
        mask_right = np.reshape(input_mask, shape_right)
        attention_mask = mask_left * mask_right

        ones = np.ones(shape=(seq_length, seq_length), dtype=input_ids.dtype)
        # Default lower triangle mask matrix
        lower_triangle_mask = np.tril(ones)
        lower_traiangle = np.expand_dims(lower_triangle_mask, 0)
        out = lower_traiangle * attention_mask
        return out.astype(np.float16)
    if is_prefill:
        freq_cos, freq_sin = precompute_freqs_cis(128, 3210, input_ids.shape[-1], pretrain_seqlen=4096)
        extraList = [get_mask(input_ids), freq_cos, freq_sin]
    else:
        extraList = [None, None, None]

    return extraList
