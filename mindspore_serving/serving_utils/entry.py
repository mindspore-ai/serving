from typing import Dict, List, Optional, Union
import enum
from mindspore_serving.schedule.cache_engine import ServingBlockMemPool, ServingCacheEngine


class EntryStatus(enum.Enum):
    """Status of a entry."""
    WAITING = enum.auto()  # waiting for inference
    RUNNING = enum.auto()  # doing inference
    WAITING_BATCH = enum.auto()  # just using in static batch
    FINISHED_STOPPED = enum.auto()  # inference over, request down
    FINISHED_LENGTH_CAPPED = enum.auto()  # over max_toekn_len
    FINISHED_ABORTED = enum.auto()  # expection
    FINISHED_IGNORED = enum.auto()  # ingore this request
    PADDING_INVAILED = enum.auto()  # padding is invaild
    INPUT_OUTOFRANGE = enum.auto()
    EMPTY_PROMPT_TOKEN = enum.auto()

    @staticmethod
    def is_finished(status: "EntryStatus") -> bool:
        return status in [
            EntryStatus.FINISHED_STOPPED,
            EntryStatus.FINISHED_LENGTH_CAPPED,
            EntryStatus.FINISHED_ABORTED,
            EntryStatus.FINISHED_IGNORED,
            EntryStatus.PADDING_INVAILED,
            EntryStatus.INPUT_OUTOFRANGE
        ]

    @staticmethod
    def get_finished_reason(status):
        finish_reason = ""
        if status == EntryStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == EntryStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif status == EntryStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == EntryStatus.FINISHED_IGNORED:
            finish_reason = "length"
        elif status == EntryStatus.INPUT_OUTOFRANGE:
            finish_reason = "prompt_out_of_range"
        return finish_reason


# one sequence of a sample used to infer
class EntryData:
    def __init__(self,
                 prompt_tokens: List[int],
                 max_token_len: int = 5000,
                 do_sample: bool = False,
                 tok_k: int = 1,
                 top_p: float = 1.0,
                 temperature: float = 1.0,
                 repetition_penalty: float = 1.0,
                 decode_index: int = 0):
        self.is_finished = False
        self._prompt_tokens = prompt_tokens
        self._output_tokens: List[int] = []
        self.status = EntryStatus.WAITING
        self.max_token_len = max_token_len
        self.do_sample = do_sample
        self.top_k = tok_k
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.decode_index = decode_index
        self.frequency_list = []
        self.prefix_index = 0
        self.read_index = 0

    def get_status(self) -> EntryStatus:
        return self.status

    def set_status(self, status: EntryStatus) -> None:
        self.status = status

    def set_decode_index(self, decode_index):
        self.decode_index = decode_index

    def get_decode_index(self):
        return self.decode_index

    def updata_output_tokens(self, token: int) -> None:
        self._output_tokens.append(token)

    def get_prompt_token(self) -> List[int]:
        return self._prompt_tokens

    def get_len(self) -> int:
        return len(self._output_tokens) + len(self._prompt_tokens)

    def get_prompt_len(self) -> int:
        return len(self._prompt_tokens)

    def get_output_token(self):
        return self._output_tokens

    def get_output_len(self) -> int:
        return len(self._output_tokens)

    def get_all_tokens(self) -> List[int]:
        return self._prompt_tokens + self._output_tokens

    def get_last_token(self) -> int:
        if not self._output_tokens:
            return self._prompt_tokens[-1]
        return self._output_tokens[-1]

    def set_finished_statue(self, statue):
        self.is_finished = statue

    def get_finished_statue(self):
        return self.is_finished

    def get_max_token_len(self) -> int:
        return self.max_token_len


class Entry:
    def __init__(self,
                 prompt: str,
                 prompt_tokens: List[int],
                 max_token_len):
        self.prompt = prompt
        self.entry_data = EntryData(prompt_tokens)
        self.entry_id: int
        self.output_logprobs: List[Dict[int, float]] = []
        self.output_tokens: List[int] = []
        self.output_text = ""
        self.status = EntryStatus.WAITING
        self.max_token_len = max_token_len

    def append_token_id(
            self,
            token: int,
    ) -> None:
        self.entry_data.updata_output_tokens()

    def get_entry_data(self):
        return self.entry_data

    def get_len(self) -> int:
        return self.entry_data.get_len()

    def get_prompt_len(self) -> int:
        return self.entry_data.get_prompt_len()

    def get_output_len(self) -> int:
        return self.entry_data.get_output_len()

    def get_token_ids(self) -> List[int]:
        return self.entry_data.get_all_tokens()

    def get_last_token_id(self) -> int:
        return self.entry_data.get_last_token()

    def get_output_token_ids(self) -> List[int]:
        return self.entry_data.get_output_token()

    def is_finished(self) -> bool:
        return EntryStatus.is_finished(self.status)

    def set_finished_statue(self, statue: bool) -> None:
        self.entry_data.set_finished_statue(statue)


class EntryMetaData:
    """
       entry meta used in combine batch
    """

    def __init__(
            self,
            page_attention: bool,     # 增加page_attention
            request_id: str,
            is_prompt: bool,
            entry_data: EntryData,
            entry_id: int,
            prompt: str,
            block_size: int
    ) -> None:
        self.request_id = request_id
        self.is_prompt = is_prompt
        self.entry_data = entry_data
        self.entry_id = entry_id
        self.prompt = prompt
        if page_attention:
            self.cache_engine = ServingCacheEngine(block_size=block_size, pool=ServingBlockMemPool.instance())


    def get_prompt(self) -> str:
        return self.prompt

    def get_entry_id(self) -> int:
        return self.entry_id

    def get_entry_data(self) -> EntryData:
        return self.entry_data

    def get_token(self):
        """get token of a request used to conduct inference"""
        return self.entry_data.get_all_tokens()

    def get_infer_stage(self):
        return self.is_prompt

    def set_is_prompt(self, statue: bool) -> None:
        self.is_prompt = statue



