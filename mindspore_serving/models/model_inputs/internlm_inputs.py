import numpy as np
from mindspore_serving.models.base_inputs import *
from mindspore_serving.serving_utils.register import Registers

__all__ = ['InternLMBasicInputs', 'InternLMExtraInputs', 'InternLMWarmupInputs']


@Registers.BASIC_INPUTS.register()
class InternLMBasicInputs(BaseBasicInputs):
    def __init__(self):
        super(InternLMBasicInputs, self).__init__()

    def get_inputs(self, input_ids, current_index, init_reset, batch_valid_length, use_current_index, *args):
        if len(args) != 1:
            batch_size, _ = input_ids.shape
            decode_index = np.array(range(batch_size), dtype=np.int64)
        else:
            decode_index = args[0]

        if use_current_index:
            inputs_list = [input_ids, current_index, batch_valid_length, decode_index]
        else:
            inputs_list = [input_ids, batch_valid_length, decode_index]
        return inputs_list


@Registers.EXTRA_INPUTS.register()
class InternLMExtraInputs(BaseExtraInputs):
    def __init__(self):
        super(InternLMExtraInputs, self).__init__()

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
class InternLMWarmupInputs(BaseWarmupInputs):
    def __init__(self):
        super(InternLMWarmupInputs, self).__init__()

    def get_warmup_inputs(self, seq_length, batch_size, full_model, use_current_index=True, valid_length=None, **kwargs):
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
        if use_current_index:
            inputs_list = [input_ids, current_index, batch_valid_length, decode_index]
        else:
            inputs_list = [input_ids, batch_valid_length, decode_index]
            extra_cls = InternLMExtraInputs()
            input_extra_list = extra_cls.get_extra_inputs(input_ids, current_index, None, full_model,
                                                          batch_valid_length, **kwargs)
            inputs_list.extend(input_extra_list)
        return inputs_list
