import numpy as np
from mindspore_serving.models.base_inputs import *
from mindspore_serving.serving_utils.register import Registers
import logging

__all__ = ['WizardCoderBasicInputs', 'WizardCoderExtraInputs', 'WizardCoderWarmupInputs']


@Registers.BASIC_INPUTS.register()
class WizardCoderBasicInputs(BaseBasicInputs):
    def __init__(self):
        super(WizardCoderBasicInputs, self).__init__()

    def get_inputs(self, input_ids, current_index, init_reset, batch_valid_length, use_current_index, *args):
        if len(args) != 1 and len(args[0]) != 1:
            batch_size, _ = input_ids.shape
            decode_index = np.array(range(batch_size), dtype=np.int64)
        else:
            decode_index = args[0]
        if use_current_index:
            inputs_list = [input_ids, current_index, batch_valid_length, decode_index]
        else:
            model_type = args[1]
            if model_type == "dyn":
                inputs_list = [input_ids, batch_valid_length, decode_index]
            else:
                inputs_list = [input_ids, current_index, batch_valid_length]
        return inputs_list


@Registers.EXTRA_INPUTS.register()
class WizardCoderExtraInputs(BaseExtraInputs):
    def __init__(self):
        super(WizardCoderExtraInputs, self).__init__()

    def get_extra_inputs(self, input_ids, current_index, init_reset, is_prefill, valid_length, **kwargs):
        zactivate_len = kwargs.pop('zactivate_len')
        if zactivate_len is None or len(zactivate_len) == 0:
            raise ValueError("invalid zactivate_len to get_extra_inputs")

        def get_act_length(seq_len, act_len_list):
            tmp_act_len = np.zeros((act_len_list[-1]), np.int64)
            for seq in act_len_list:
                if seq_len <= seq:
                    tmp_act_len = np.zeros((seq), np.int64)
                    break
            return tmp_act_len

        if not is_prefill:
            max_seq = 0
            for i in range(len(valid_length)):
                max_seq = max(max_seq, valid_length[i] + 1)
            return [get_act_length(max_seq, zactivate_len)]
        max_prefill_length = 0
        for item in valid_length:
            max_prefill_length = max(max_prefill_length, item)
        act_len = get_act_length(max_prefill_length + 1, zactivate_len)
        return [act_len]


@Registers.WARMUP_INPUTS.register()
class WizardCoderWarmupInputs(BaseWarmupInputs):
    def __init__(self):
        super(WizardCoderWarmupInputs, self).__init__()

    def get_warmup_inputs_pa(self, seq_length, batch_size, full_model, **kwargs):
        full_seq_len = seq_length
        inc_seq_len = kwargs.pop('decode_seq_length')  # 这里需验证
        block_size = kwargs.pop('block_size')
        logging.debug("full_seq_len：%s, inc_seq_len:%s", full_seq_len, inc_seq_len)
        if full_model:
            input_ids = np.zeros([batch_size, full_seq_len], dtype=np.int32)
            for i in range(batch_size):
                input_ids[i][0] = 16829
            slot_mapping = np.full((full_seq_len,), 0, dtype=np.int32)
            slot_mapping[0] = 128
        else:
            slot_mapping = np.full((batch_size,), 128, dtype=np.int32)
        logging.debug("block size: %s", block_size)
        block_tables = np.full((batch_size, inc_seq_len // block_size), -1, dtype=np.int32)
        logging.debug("block tables shape: %s, seq_length is: %s", block_tables.shape, inc_seq_len)
        logging.debug("slot_mapping shape: %s", slot_mapping.shape)
        for i in range(batch_size):
            block_tables[i][0] = 1
        return slot_mapping, block_tables
    # 加入page attention判断

    def get_warmup_inputs(self, seq_length, batch_size, full_model, use_current_index=True, valid_length=None, page_attention=False, **kwargs):
        print("===========================get_warmup_inputs")
        input_ids = np.ones([batch_size, seq_length], dtype=np.int32)
        current_index = np.array([1] * batch_size, dtype=np.int32)

        if use_current_index:
            valid_length_dtype = np.int64
        else:
            valid_length_dtype = np.int32

        if valid_length is None:
            batch_valid_length = np.array([1] * batch_size, dtype=valid_length_dtype)
        else:
            batch_valid_length = np.array(valid_length * batch_size, dtype=valid_length_dtype)

        decode_index = np.array(range(batch_size), dtype=np.int64)
        if page_attention:
            inputs_list = [input_ids, batch_valid_length]
        else:
            if use_current_index:
                inputs_list = [input_ids, current_index, batch_valid_length, decode_index]
            else:
                model_type = kwargs.get('model_type', "dyn")
                if model_type == "dyn":
                    inputs_list = [input_ids, batch_valid_length, decode_index]
                else:
                    init_reset = np.array([False], np.bool_)
                    inputs_list = [input_ids, current_index, init_reset, batch_valid_length]

        if page_attention:
            print("==============================================page_attention:", page_attention)
            slot_mapping, block_tables = self.get_warmup_inputs_pa(seq_length, batch_size, full_model, **kwargs)
            if full_model:
                print("prefii append slot_mapping")
                inputs_list.append(slot_mapping)
            else:
                print("decode append slot_mapping")
                inputs_list.append(block_tables)
                inputs_list.append(slot_mapping)
        else:
            model_type = kwargs.pop('model_type')
            logging.debug(f"model_type in wizardCoderWarmup: {model_type}")
            # dyn model type need 'act_len' parameter
            if model_type == "dyn":
                extra_cls = WizardCoderExtraInputs()
                input_extra_list = extra_cls.get_extra_inputs(input_ids, current_index, None, full_model,
                                                              batch_valid_length, **kwargs)
                inputs_list.extend(input_extra_list)
        return inputs_list
