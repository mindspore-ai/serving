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
import numpy as np
from .tokenizer.llama_tokenizer import LlamaTokenizer
from easydict import EasyDict


Baseconfig = EasyDict({
    'frequency_penalty': 1.5,
    'presence_penalty': 0.3,
    'max_generate_length': 500,
    'top_k_num': 3,
    'top_p': 1.0,
    'end_token': 9,
    'seq_length': 2048,
    'vocab_size': 32000,
    'batch_size': 1,
    
    
})


def get_token():
    # preprocessor
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    tokenizer = LlamaTokenizer(os.path.join(cur_dir, "./tokenizer.model"))
    return tokenizer


def topk_fun(logits, topk=5):
    """Get topk"""
    target_column = logits[0].tolist()
    sorted_array = [(k, v) for k, v in enumerate(target_column)]
    sorted_array.sort(key=lambda x: x[1], reverse=True)
    topk_array = sorted_array[:topk]
    index, value = zip(*topk_array)
    index = np.array([index])
    value = np.array([value])
    return value, index


def post_funciton(reply):
    # postprocessor
    return reply