from typing import Dict, List, Optional, Union

from serving_utils.entry import *


class CompletionOutput:
    """
       index : current index of output token
       text: text of current iteration step generation
       token_ids: all tokens of one request so for.
    """

    def __init__(
            self,
            index: int,
            text: str,
            finished
    ) -> None:
        self.index = index
        self.text = text
        self.finish_reason = finished

    def finished(self) -> bool:
        return self.finish_reason is not None

    def get_text(self):
        return self.text


class ResponseOutput:
    """output put in ascynio queue used by stream return"""

    def __init__(
            self,
            request_id: str,
            prompt: str,
            prompt_token_ids: List[int],
            outputs: List[CompletionOutput],
            finished: bool,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs
        self.finished = finished

    @classmethod
    def generate_result(cls,
                        output: int,
                        entry_meta_data: EntryMetaData,
                        output_str: str,
                        eos_id,
                        reason: str = None):
        index = entry_meta_data.get_entry_data().get_output_len()
        token_ids = entry_meta_data.get_entry_data().get_all_tokens()
        request_id = entry_meta_data.request_id

        status = entry_meta_data.get_entry_data().get_status()
        finished_reason = None
        finished = False

        if status != EntryStatus.RUNNING or status != EntryStatus.WAITING:
            if status != EntryStatus.WAITING_BATCH:
                finished_reason = EntryStatus.get_finished_reason(status)

        if status == EntryStatus.INPUT_OUTOFRANGE:
            print(f'>>>>>>request {entry_meta_data.request_id} input is too large, out of input length of model')
            finished = True
            finished_reason = EntryStatus.get_finished_reason(status)
            completion_out = CompletionOutput(index, text=reason, finished=finished_reason)
            entry_meta_data.entry_data.status = EntryStatus.FINISHED_STOPPED
            return cls(entry_meta_data.request_id,
                       entry_meta_data.get_prompt(),
                       entry_meta_data.get_entry_data().get_prompt_token(),
                       [completion_out],
                       finished)

        if output == eos_id:
            finished = True

        elif entry_meta_data.entry_data.get_output_len() == entry_meta_data.entry_data.max_token_len:
            print(f"stop inference because of iteration is max_len, request is {request_id}")
            finished = True

        completion_out = CompletionOutput(index, text=output_str, finished=finished_reason)
        return cls(entry_meta_data.request_id,
                   entry_meta_data.get_prompt(),
                   entry_meta_data.get_entry_data().get_prompt_token(),
                   [completion_out],
                   finished)


class Counter:
    def __init__(self, start=0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0
