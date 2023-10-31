# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MindSpore Serving server app"""

import uuid
import asyncio
import logging
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

import uvicorn
import json
from client_utils import ClientRequest, Parameters, Response, StreamResponse, Token

from server.llm_server_post import LLMServer
import time

logging.basicConfig(level=logging.DEBUG,
                    filename='./output/server_app.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

app = FastAPI()
llm_server = None


async def get_full_res(request, results):
    all_texts = ''
    async for result in results:
        prompt_ = result.prompt
        answer_texts = [output.text for output in result.outputs]
        text = answer_texts[0]
        if text is None:
            text = ""
        all_texts += text

    ret = {
        "generated_text": all_texts,
    }
    yield (json.dumps(ret) + '\n').encode("utf-8")


async def get_stream_res(request, results):
    all_texts = ''
    async for result in results:
        prompt_ = result.prompt
        answer_texts = [output.text for output in result.outputs]

        text = answer_texts[0]
        if text is None:
            text = ""
        all_texts += text
        ret = {
            "token": {
                "text": text
            },
        }
        print(ret)
        yield ("data:" + json.dumps(ret) + '\n').encode("utf-8")
    print(all_texts)
    return_full_text = request.parameters.return_full_text
    if return_full_text:
        ret = {
            "generated_text": all_texts,
        }
        yield ("data:" + json.dumps(ret) + '\n').encode("utf-8")


def send_request(request: ClientRequest):
    print('request: ', request)
    request_id = str(uuid.uuid1())

    if request.parameters.do_sample is None:
        request.parameters.do_sample = False
    if request.parameters.top_k is None:
        request.parameters.top_k = 100
    if request.parameters.top_p is None:
        request.parameters.top_p = 1.0
    if request.parameters.temperature is None:
        request.parameters.temperature = 1.0
    if request.parameters.repetition_penalty is None:
        request.parameters.repetition_penalty = 1.0
    if request.parameters.max_new_tokens is None:
        request.parameters.max_new_tokens = 300

    parms = {
        "prompt": request.inputs,
        "do_sample": request.parameters.do_sample,
        "top_k": request.parameters.top_k,
        "top_p": request.parameters.top_p,
        "temperature": request.parameters.temperature,
        "repetition_penalty": request.parameters.repetition_penalty,
        "max_token_len": request.parameters.max_new_tokens
    }
    print('generate_answer...')
    results = llm_server.generate_answer(request_id, **parms)
    return results


@app.post("/models/llama2")
async def async_generator(request: ClientRequest):
    results = send_request(request)

    if request.stream:
        print('get_stream_res...')
        return StreamingResponse(get_stream_res(request, results))
    else:
        print('get_full_res...')
        return StreamingResponse(get_full_res(request, results))


@app.post("/models/llama2/generate")
async def async_full_generator(request: ClientRequest):
    results = send_request(request)

    print('get_full_res...')
    return StreamingResponse(get_full_res(request, results))


@app.post("/models/llama2/generate_stream")
async def async_stream_generator(request: ClientRequest):
    results = send_request(request)

    print('get_stream_res...')
    return StreamingResponse(get_stream_res(request, results))


if __name__ == "__main__":
    llm_server = LLMServer()
    port = 9800
    print('server port is: ', port)

    uvicorn.run(app, host='localhost', port=port)
