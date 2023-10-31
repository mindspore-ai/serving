import time
import logging
import sys
from typing import List

import numpy as np

from .model_init import DisModel
from lib.entry import EntryMetaData
from config.serving_config import topk_fun, topk, Baseconfig
import shared_memory
from worker.worker_to_agent import *
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from enum import Enum

pool = ThreadPoolExecutor(max_workers=20, thread_name_prefix='test_thread')

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
    freqs_cos = np.cos(emb) # (seq_len, head_dim)
    freqs_sin = np.sin(emb) # (seq_len, head_dim)

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


# class worker
class Worker:
    def __init__(self, agent_ip: str,
                 agentports: List[int],
                 model_name: str) -> None:
        self.model = DisModel()
        self.agent_ip = agent_ip
        self.agent_ports = agentports
        self.model_name = model_name
        # 申请4块共享内存
        self.shm = shared_memory.SharedMemory(create=True, size=1024 * 1024 * 1024)
        self.shms = []
        self.shm_names = []
        self.valid_length = None
        self.seq_length = None
        self.batch_size = 1
        self.current_index = 0
        self.vocab_size = Baseconfig.vocab_size
        self.seq_length_list = Baseconfig.seq_length
        for i in range(5):
            tmp = shared_memory.SharedMemory(create=True, size=1024 * 1024 * 1024)
            self.shms.append(tmp)
            self.shm_names.append(tmp.name)

    def _init_worker(self) -> None:
        logging.info(">>>>>>>>>>>>>>>>>_init_worker in worker, server llm model is: {}".format(self.model_name))
        self.model.init(self.agent_ip, self.agent_ports, self.model_name, self.shm_names)
    
    @staticmethod
    def _get_seq_length_dynmic_dinning(seq_list, seq_length):
        for data in seq_list:
            if seq_length < data:
                return data
        return seq_list[-1]

    def _padding(self, origin_inputs):
        pad_length = 0
        current_seq_length = origin_inputs.shape[-1]
        seq_length = self._get_seq_length_dynmic_dinning(self.seq_length_list, current_seq_length)
        if current_seq_length > seq_length:
            logging.error('input sequence length is over max in serving system!')
      
        pad_length = seq_length - origin_inputs.shape[-1]
        input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, 0))
        
        return input_ids, seq_length

    def _get_valid_length(self, origin_inputs):
        batch_size, _ = origin_inputs.shape
        valid_length_each_example = []
        for i in range(batch_size):
        # As the nonzero returns the index and we need length
            valid_length_each_example.append(np.max(np.argwhere(origin_inputs[i] != 0)) + 1)
        valid_length = np.array(valid_length_each_example, dtype=np.int32)
        return valid_length, batch_size

    def _predict(self,
                 input_ids: List[List[int]],
                 is_prefill: bool,
                 frequency_penalty: float = 0.0,
                 presence_penalty: float = 0.0,
                 frequency_list=None,
                 **generate_parms) -> List:
        time_start = time.time()
        outputs = []
        # Init outputs with original inputs
        if is_prefill:
            origin_inputs = np.array(input_ids)
            self.valid_length, self.batch_size = self._get_valid_length(origin_inputs)
            input_ids, seq_length = self._padding(origin_inputs)
            current_index_ = [self.valid_length[i] - 1 + i * seq_length for i in range(self.batch_size)]
            
            self.current_index = np.array(current_index_, np.int32)
        else:
            self.valid_length + 1
            self.current_index + 1
        # If target length exceeds seq_length, use seq_length instead
        # A list of the frequency of each token
        # For first graph, not_init should be false
        init_true = True
        init = init_true and not is_prefill
        logging.info("pre-process time is {} ".format((time.time() - time_start) * 1000))
        mask_time = time.time()
        if is_prefill:
            mask = get_mask(input_ids)
            freq_cos, freq_sin = precompute_freqs_cis(128, 1234, input_ids.shape[-1], pretrain_seqlen=4096)
        else:
            mask = None
            freq_cos = None
            freq_sin = None
        
        logging.info("mask time is {} ".format((time.time() - mask_time) * 1000))
        # Call a single inference with input size of (bs, seq_length)
        call = time.time()
        result, shm = self.model.callV3(self.shms, np.array(input_ids, np.int32),
                                        self.current_index, self.valid_length, init, is_prefill, mask, freq_cos, freq_sin,
                                        **generate_parms)
        if is_prefill:
            logging.info("PrefillTime {} ".format((time.time() - call) * 1000))
        else:
            logging.info("DecodeTime {} ".format((time.time() - call) * 1000))
        return result #, frequency_list

    def get_generate_parms(self, entry_metadata_list):
        do_sample_list = []
        top_k_list = []
        top_p_list = []
        temperature_list = []
        repetition_penalty = []
        for item in entry_metadata_list:
            entry_data = item.get_entry_data()
            do_sample_list.append(entry_data.do_sample)
            top_k_list.append(entry_data.top_k)
            top_p_list.append(entry_data.top_p)
            temperature_list.append(entry_data.temperature)
            repetition_penalty.append(entry_data.repetition_penalty)

        parms = {
            "do_sample_list": do_sample_list,
            "top_k_list": top_k_list,
            "top_p_list": top_p_list,
            "temperature_list": temperature_list,
            "repetition_penalty": repetition_penalty,
        }

        return parms

    def predict(self, entry_metadata_list: List[EntryMetaData], config=Baseconfig):
        if_prefill = entry_metadata_list[0].is_prompt
        inputs_ids = []  # length is batch size
        for item in entry_metadata_list:
            entry_data = item.get_entry_data()
            token_ids = entry_data.get_all_tokens()
            inputs_ids.append(token_ids)
            # add frequency list
        generate_parms = self.get_generate_parms(entry_metadata_list)
        time_start = time.time()
        # add frequency list
        outputs = self._predict(inputs_ids, if_prefill,
                                **generate_parms)#, top_p, topk_num, frequency_penalty, presence_penalty, frequency_list)
        return outputs  # , frequency_lists
