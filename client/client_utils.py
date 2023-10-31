from pydantic import BaseModel
from typing import Optional


class Parameters(BaseModel):
    # mode: Optional[int] = 0
    do_sample: bool = True
    repetition_penalty: Optional[float] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_new_tokens: Optional[int] = None
    return_full_text: bool = True



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
