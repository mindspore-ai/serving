import numpy as np


class BaseBasicInputs:
    def __init__(self):
        pass

    def get_inputs(self, input_ids, current_index, init_reset, batch_valid_length, use_current_index, *args):
        if input_ids is None or current_index is None or batch_valid_length is None:
            raise ValueError('empty input params for BaseBasicInputs')
        if init_reset is None:
            init_reset = np.array([True])

        if use_current_index:
            return [input_ids, current_index, init_reset, batch_valid_length]
        else:
            return [input_ids, init_reset, batch_valid_length]


class BaseExtraInputs:
    def __init__(self):
        pass

    def get_extra_inputs(self, input_ids, current_index, init_reset, is_prefill, valid_length, **kwargs):
        pass


class BaseWarmupInputs:
    def __init__(self):
        pass

    def get_warmup_inputs(self, seq_length, batch_size, full_model, use_current_index=True, valid_length=None, **kwargs):
        pass
