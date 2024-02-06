import logging
from typing import List

from mindspore_serving.serving_utils.constant import *
from mindspore_serving.serving_utils.entry import EntryMetaData, EntryStatus


class CompletionOutput:
    """
       index : current index of output token
       text: text of current iteration step generation
       token_ids: all tokens of one request so for.
    """

    def __init__(
            self,
            index: int,
            special: bool,
            logprob: float,
            text: str
    ) -> None:
        self.index = index
        self.text = text
        self.logprob = logprob
        self.special = special


class ResponseOutput:
    """output put in ascynio queue used by stream return"""

    def __init__(
            self,
            request_id: str,
            prompt: str,
            prompt_token_ids: List[int],
            outputs: List[CompletionOutput],
            finished: bool,
            finish_reason: str,
            output_tokens_len: int
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs
        self.finished = finished
        self.finish_reason = finish_reason
        self.output_tokens_len = output_tokens_len

    @classmethod
    def generate_result(cls,
                        output_token: int,
                        output_token_logprob: float,
                        entry_meta_data: EntryMetaData,
                        output_str: str,
                        eos_id,
                        reason: str = None):
        output_tokens_len = entry_meta_data.get_entry_data().get_output_len()
        request_id = entry_meta_data.request_id
        status = entry_meta_data.get_entry_data().get_status()
        finished_reason = None
        finished = False

        if status != EntryStatus.RUNNING or status != EntryStatus.WAITING:
            if status != EntryStatus.WAITING_BATCH:
                finished_reason = EntryStatus.get_finished_reason(status)
        if output_token == INPUT_EMPTY_TOKEN[0]:
            finished = True
            completion_out = CompletionOutput(0, text=reason, logprob=0.0, special=False)
            entry_meta_data.get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            return cls(request_id,
                       entry_meta_data.get_prompt(),
                       entry_meta_data.get_entry_data().get_prompt_token(),
                       [completion_out],
                       finished,
                       "prompt_token_ids_empty",
                       0)
        
        if output_token == INPUT_OUT_OF_TOKEN[0]:
            print(f'>>>>>>request {request_id} input is too large, out of input length of model')
            finished = True
            completion_out = CompletionOutput(0, text=reason, logprob=0.0, special=False)
            entry_meta_data.get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            return cls(request_id,
                       entry_meta_data.get_prompt(),
                       entry_meta_data.get_entry_data().get_prompt_token(),
                       [completion_out],
                       finished,
                       "prompt_out_of_range",
                       0)

        if status == EntryStatus.PADDING_INVAILED:
            finished = False

        elif output_token == PREDICT_FAILED_CODE:
            entry_meta_data.get_entry_data().prefix_index = 0
            entry_meta_data.get_entry_data().read_index = 0
            output_str = RETURN_REASON_PREDICT_FAILED
            finished = True
            finished_reason = "predict_failed"

        elif output_token == eos_id:
            entry_meta_data.get_entry_data().prefix_index = 0
            entry_meta_data.get_entry_data().read_index = 0
            finished = True
            finished_reason = "eos"

        elif entry_meta_data.entry_data.get_output_len() == entry_meta_data.entry_data.max_token_len:
            entry_meta_data.get_entry_data().prefix_index = 0
            entry_meta_data.get_entry_data().read_index = 0
            logging.debug("stop inference because of iteration is max_len, request is {}".format(request_id))
            finished = True
            finished_reason = "length"

        is_special = True
        if output_str != "":
            is_special = False

        completion_out = CompletionOutput(output_token, text=output_str, logprob=output_token_logprob, special=is_special)

        return cls(request_id,
                   entry_meta_data.get_prompt(),
                   entry_meta_data.get_entry_data().get_prompt_token(),
                   [completion_out],
                   finished,
                   finished_reason,
                   output_tokens_len)


class Counter:
    def __init__(self, start=0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


class ModelInfo:

    def __init__(
            self,
             docker_label: str,
             max_batch_total_tokens: int,
             max_concurrent_requests: int,
             max_input_length: int,
             max_total_tokens: int,
             model_dtype: str,
             model_id: str
    ) -> None:
        self.docker_label = docker_label
        self.max_batch_total_tokens = max_batch_total_tokens
        self.max_concurrent_requests = max_concurrent_requests
        self.max_input_length = max_input_length
        self.max_total_tokens = max_total_tokens
        self.max_batch_total_tokens = max_batch_total_tokens
        self.model_dtype = model_dtype
        self.model_id = model_id