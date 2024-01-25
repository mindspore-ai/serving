from pydantic import BaseModel
from typing import Optional, List


class Parameters(BaseModel):
    # mode: Optional[int] = 0
    do_sample: bool = True
    repetition_penalty: Optional[float] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_new_tokens: Optional[int] = None
    return_full_text: bool = True
    return_protocol = "sse"

    decoder_input_details: bool = False
    details: bool = False
    seed: int = 0
    stop: List[str] = []
    top_n_tokens: int = 0
    truncate: bool = False
    typical_p: int = 0
    watermark: bool = False


class ClientRequest(BaseModel):
    # Prompt
    inputs: str
    # Generation parameters
    parameters: Optional[Parameters] = None
    # Whether to stream output tokens
    stream: bool = False


class Response(BaseModel):
    generated_text: str


class Token(BaseModel):
    text: str


class StreamResponse(BaseModel):
    generated_text: Optional[str] = None
    token: Optional[Token] = None


class ValidatorUtil:
    @staticmethod
    def validate_top_k(num, max_num):
        if isinstance(num, int) and 0 < num < max_num:
            return True
        else:
            return False

    @staticmethod
    def validate_top_p(num):
        if isinstance(num, float) and 0 < num <= 1:
            return True
        else:
            return False

    @staticmethod
    def validate_temperature(num):
        if isinstance(num, float) and 1e-5 < num <= 65536:
            return True
        else:
            return False
