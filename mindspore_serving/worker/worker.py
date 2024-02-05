import time
import logging
from typing import List
import numpy as np
from multiprocessing import shared_memory
from .model_init_multimodel import DisModel
from mindspore_serving.serving_utils.entry import EntryMetaData, EntryStatus
from mindspore_serving.config.config import ServingConfig
from mindspore_serving.models.build_inputs import build_inputs


# class worker
class Worker:
    def __init__(self, config: ServingConfig) -> None:
        self.model = DisModel()
        self.config = config
        self.shms = []
        self.shm_names = []
        self.valid_length = None
        self.seq_length = None
        self.batch_size = 1
        self.current_index = 0
        self.vocab_size = config.model_config.vocab_size
        self.seq_length_list = config.model_config.seq_length
        self.extra_func = build_inputs(config.extra_inputs, module_type='extra_inputs')
        # 0 : input_ids, current_index, valid_length, init_reset
        # 1 : mask=mask,
        # 2 : freq_cos
        # 3 : freq_sin
        # 4 : gen_params, top_k top_p ...
        # 5 : predict output
        # 6 : logprob
		# 7 : block table # 128个slot组成一个block
        # 8 : slot mapping #

        shm_name_num = 9 if self.config.model_config.page_attention else 7  # 根据模型分配
        for i in range(shm_name_num):
            tmp = shared_memory.SharedMemory(create=True, size=1024 * 1024 * 1024)
            self.shms.append(tmp)
            self.shm_names.append(tmp.name)

    def _init_worker(self) -> None:
        try:
            self.model.init(self.config, self.shm_names)
        except ConnectionError:
            self.model.reset_agent_status(self.config)
            self.model.init(self.config, self.shm_names)

    @staticmethod
    def _get_seq_length_dynmic_dinning(seq_list, seq_length):
        for data in seq_list:
            if seq_length < data:
                return data
        return seq_list[-1]

    @staticmethod
    def _padding(origin_inputs, seq_length, default_padding_values):
        pad_ids = []
        for item in origin_inputs:
            pad_length = seq_length - len(item)
            if pad_length < 0:
                logging.error('input sequence length is over max in serving system!')
            pad_item = np.pad(item, (0, pad_length), 'constant', constant_values=default_padding_values)
            pad_ids.append(pad_item)

        logging.debug("prefill _padding result list is {}".format(pad_ids))
        return np.array(pad_ids)

    @staticmethod
    def _get_valid_length(origin_inputs, default_padding_values):
        batch_size, _ = origin_inputs.shape
        valid_length_each_example = []
        for i in range(batch_size):
            # As the nonzero returns the index and we need length
            valid_length_each_example.append(np.max(np.argwhere(origin_inputs[i] != default_padding_values)) + 1)
        valid_length = np.array(valid_length_each_example, dtype=np.int32)
        return valid_length, batch_size

    # pa
    def _get_seq_length(self, input_ids, is_prefill):
        max_length = 0
        if not is_prefill:
            if self.config.model_config.page_attention:
                return self.config.pa_config.decode_seq_length
        for item in input_ids:
            if isinstance(item, list):
                max_length = max(max_length, len(item))
            else:
                max_length = max(max_length, 1)
        if self.config.model_config.seq_type == 'dyn':
            seq_length = max_length
        elif len(self.config.model_config.seq_length) > 1:
            seq_length = self._get_seq_length_dynmic_dinning(self.seq_length_list, max_length)
        else:
            if len(self.config.model_config.seq_length) == 0 and self.config.model_config.seq_type != 'dyn':
                logging.error('seq length is None ! using default 2048')
                seq_length = 2048
            else:
                seq_length = self.config.model_config.seq_length[0]
        return seq_length

    def _predict(self,
                 input_ids: List[List[int]],
                 is_prefill: bool,
                 valid_batch_flag: List[int],
                 current_batch_size=None,
                 **generate_parms) -> List:
        time_start = time.time()
        # Init outputs with original inputs
        seq_length = self._get_seq_length(input_ids, is_prefill)
        logging.info("decode_seq_length: %s", seq_length)
        generate_parms["seq_length"] = seq_length
        if is_prefill:
            default_padding_values = 0
            if self.config.model_config.pad_token_id:
                default_padding_values = self.config.model_config.pad_token_id
            input_ids = self._padding(input_ids, seq_length, default_padding_values)
            self.valid_length, self.batch_size = self._get_valid_length(input_ids, default_padding_values)
            current_index_ = [self.valid_length[i] - 1 + i * seq_length for i in range(self.batch_size)]
            self.current_index = np.array(current_index_, np.int32)
        # If target length exceeds seq_length, use seq_length instead
        # A list of the frequency of each token
        # For first graph, not_init should be false
        init_true = True
        init = init_true and not is_prefill

        logging.info("pre-process time is {} ".format((time.time() - time_start) * 1000))
        mask_time = time.time()
        extra_input_list = self.extra_func.get_extra_inputs(input_ids, self.current_index, init, is_prefill,
                                                            self.valid_length,
                                                            zactivate_len=self.config.model_config.zactivate_len)
        if extra_input_list is None:
            logging.error('extra inputs by customer is None,please check it in server config!')
        logging.info("mask time is {} ".format((time.time() - mask_time) * 1000))
        # Call a single inference with input size of (bs, seq_length)
        call = time.time()
        result, shm = self.model.call(self.shms, np.array(input_ids, np.int32), self.current_index,
                                      self.valid_length, init, is_prefill, valid_batch_flag,
                                      extra_inputs=extra_input_list, current_batch_size=current_batch_size,
                                      **generate_parms)
        if is_prefill:
            logging.info("PrefillTime {} ".format((time.time() - call) * 1000))
        else:
            logging.info("DecodeTime {} ".format((time.time() - call) * 1000))
        return result

    @staticmethod
    def get_generate_parms(page_attention, entry_metadata_list):
        do_sample_list = []
        top_k_list = []
        top_p_list = []
        temperature_list = []
        repetition_penalty = []
        decode_index_list = []
        cache_engine_list = []
        for item in entry_metadata_list:
            entry_data = item.get_entry_data()
            do_sample_list.append(entry_data.do_sample)
            top_k_list.append(entry_data.top_k)
            top_p_list.append(entry_data.top_p)
            temperature_list.append(entry_data.temperature)
            repetition_penalty.append(entry_data.repetition_penalty)
            decode_index_list.append(entry_data.decode_index)
            if page_attention:
                cache_engine_list.append(item.cache_engine)
        parms = {
            "do_sample_list": do_sample_list,
            "top_k_list": top_k_list,
            "top_p_list": top_p_list,
            "temperature_list": temperature_list,
            "repetition_penalty": repetition_penalty,
            "decode_index_list": decode_index_list,
        }
        if page_attention:
            parms["cache_engine_list"] = cache_engine_list
        return parms

    def predict(self, current_batch_size, entry_metadata_list: List[EntryMetaData]):
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
            if entry_data.get_status() == EntryStatus.RUNNING:
                valid_batch_flag.append(1)
            else:
                valid_batch_flag.append(0)
        generate_parms = self.get_generate_parms(self.config.model_config.page_attention, entry_metadata_list)
        current_batch_size_dyn = current_batch_size
        outputs = self._predict(inputs_ids, if_prefill, valid_batch_flag, current_batch_size=current_batch_size_dyn,
                                **generate_parms)

        return outputs

    def stop(self):
        self.model.stop()
        for shm in self.shms:
            shm.close()
