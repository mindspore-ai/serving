import sys
import time
import logging
from typing import List

import numpy as np

from .model_init_multimodel import DisModel
from serving_utils.entry import EntryMetaData, EntryStatus
from config.serving_config import Baseconfig, ExtraInput, ModelName
from multiprocessing import shared_memory
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

pool = ThreadPoolExecutor(max_workers=20, thread_name_prefix='test_thread')


class AddtionalInput:
    def __init__(self) -> None:
        self.ExtraInput = ExtraInput
    def getExtraInput(self, input_ids, current_index, init_reset, is_prefill, valid_length, **generate_parms):
        return self.ExtraInput(input_ids, current_index, init_reset, is_prefill, valid_length, **generate_parms)

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

        # 0 : input_ids, current_index, valid_length, init_reset
        # 1 : mask=mask,
        # 2 : freq_cos
        # 3 : freq_sin
        # 4 : gen_params, top_k top_p ...
        # 5 : predict output

        for i in range(6):
            tmp = shared_memory.SharedMemory(create=True, size=1024 * 1024 * 1024)
            self.shms.append(tmp)
            self.shm_names.append(tmp.name)

    def _init_worker(self) -> None:
        logging.info(">>>>>>>>>>>>>>>>>_init_worker in worker, server llm model is: {}".format(self.model_name))
        try:
            self.model.init(self.agent_ip, self.agent_ports, self.model_name, self.shm_names)
        except ConnectionError:
            self.model.reset_agent_status(self.agent_ip, self.agent_ports)
            # sys.exit()
            self.model.init(self.agent_ip, self.agent_ports, self.model_name, self.shm_names)

    def __del__(self):
        for shm in self.shms:
            shm.close()
    
    @staticmethod
    def _get_seq_length_dynmic_dinning(seq_list, seq_length):
        for data in seq_list:
            if seq_length < data:
                return data
        return seq_list[-1]

    def _padding(self, origin_inputs, seq_length):
        pad_ids = []
        for item in origin_inputs:
            pad_length = seq_length - len(item)
            if pad_length < 0:
                logging.error('input sequence length is over max in serving system!')
            pad_item = np.pad(item, (0, pad_length), 'constant', constant_values=0)
            pad_ids.append(pad_item)
        return np.array(pad_ids)
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
                 valid_batch_flag: List[int],
                 current_batch_size=None,
                 **generate_parms) -> List:
        time_start = time.time()
        outputs = []
        # Init outputs with original inputs
        if is_prefill:
            max_length = 0
            for item in input_ids:
                max_length = max(max_length, len(item))
            if 'seq_type' in Baseconfig and Baseconfig.seq_type == 'dyn':
                seq_length = max_length
            elif len(Baseconfig.seq_length) > 1:
                seq_length = self._get_seq_length_dynmic_dinning(self.seq_length_list, max_length)
            else:
                if 'seq_length' not in Baseconfig or (Baseconfig.seq_length == [] and Baseconfig.seq_type != 'dyn'):
                    logging.error('seq length is None ! using default 2048')
                    seq_length = 2048
                else:
                    seq_length = Baseconfig.seq_length[0]
            input_ids = self._padding(input_ids, seq_length)
            logging.debug("seq_length is {}, input_ids after padding is {}".format(seq_length, input_ids))
            self.valid_length, self.batch_size = self._get_valid_length(input_ids)
            logging.debug("valid length is {}".format(self.valid_length))
            current_index_ = [self.valid_length[i] - 1 + i * seq_length for i in range(self.batch_size)]
            self.current_index = np.array(current_index_, np.int32)
        # If target length exceeds seq_length, use seq_length instead
        # A list of the frequency of each token
        # For first graph, not_init should be false
        init_true = True
        init = init_true and not is_prefill
        addtional_input = AddtionalInput()
        logging.info("pre-process time is {} ".format((time.time() - time_start) * 1000))
        mask_time = time.time()
        input_list = addtional_input.getExtraInput(input_ids, self.current_index, init, is_prefill, self.valid_length, **generate_parms)
        if input_list is None:
            logging.error('extra inputs by customer is None,please check it in server config!')
        logging.info("mask time is {} ".format((time.time() - mask_time) * 1000))
        # Call a single inference with input size of (bs, seq_length)
        call = time.time()
        result, shm = self.model.callV3(self.shms, np.array(input_ids, np.int32),
                                        self.current_index, self.valid_length, init, is_prefill, valid_batch_flag,
                                        InputExtraList=input_list, current_batch_size=current_batch_size,
                                        **generate_parms)
        if is_prefill:
            logging.info("PrefillTime {} ".format((time.time() - call) * 1000))
        else:
            logging.info("DecodeTime {} ".format((time.time() - call) * 1000))
        return result

    def get_generate_parms(self, entry_metadata_list):
        do_sample_list = []
        top_k_list = []
        top_p_list = []
        temperature_list = []
        repetition_penalty = []
        decode_index_list = []
        for item in entry_metadata_list:
            entry_data = item.get_entry_data()
            do_sample_list.append(entry_data.do_sample)
            top_k_list.append(entry_data.top_k)
            top_p_list.append(entry_data.top_p)
            temperature_list.append(entry_data.temperature)
            repetition_penalty.append(entry_data.repetition_penalty)
            decode_index_list.append(entry_data.decode_index)

        parms = {
            "do_sample_list": do_sample_list,
            "top_k_list": top_k_list,
            "top_p_list": top_p_list,
            "temperature_list": temperature_list,
            "repetition_penalty": repetition_penalty,
            "decode_index_list": decode_index_list,
        }

        return parms

    def predict(self, current_batch_size, entry_metadata_list: List[EntryMetaData], config=Baseconfig):
        if_prefill = entry_metadata_list[0].is_prompt
        inputs_ids = []  # length is batch size
        valid_batch_flag = []
        for item in entry_metadata_list:
            entry_data = item.get_entry_data()
            token_ids = entry_data.get_all_tokens()
            if if_prefill:
                inputs_ids.append(token_ids)
            else:
                inputs_ids.append(token_ids[-1])
            logging.debug("batch_item status is {}".format(entry_data.get_status()))
            if entry_data.get_status() == EntryStatus.RUNNING:
                valid_batch_flag.append(1)
            else:
                valid_batch_flag.append(0)
        generate_parms = self.get_generate_parms(entry_metadata_list)
        current_batch_size_dyn = current_batch_size

        outputs = self._predict(inputs_ids, if_prefill, valid_batch_flag, current_batch_size=current_batch_size_dyn, **generate_parms)
        return outputs

    def stop(self):
        return self.model.stop()
