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
from multiprocessing import Process

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

import uvicorn
import json

from config.serving_config import Baseconfig, SERVER_APP_HOST, SERVER_APP_PORT, ModelName
from server.llm_server_post import LLMServer
from client.client_utils import ClientRequest, Parameters


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


async def get_full_res_sse(request, results):
    all_texts = ''
    async for result in results:
        answer_texts = [output.text for output in result.outputs]
        text = answer_texts[0]
        if text is None:
            text = ""
        all_texts += text

    ret = {"event": "message", "retry": 15000, "generated_text": all_texts}
    yield json.dumps(ret)


async def get_stream_res(request, results):
    all_texts = ''
    index = 0
    async for result in results:
        prompt_ = result.prompt
        answer_texts = [output.text for output in result.outputs]

        text = answer_texts[0]
        if text is None:
            text = ""
        else:
            index += 1
        all_texts += text
        ret = {
            "token": {
                "text": text,
                "index": index
            },
        }
        print(ret, index)
        yield ("data:" + json.dumps(ret) + '\n').encode("utf-8")
    print(all_texts)
    return_full_text = request.parameters.return_full_text
    if return_full_text:
        ret = {
            "generated_text": all_texts,
        }
        yield ("data:" + json.dumps(ret) + '\n').encode("utf-8")


async def get_stream_res_sse(request, results):
    all_texts = ""
    index = 0
    async for result in results:
        answer_texts = [output.text for output in result.outputs]
        text = answer_texts[0]
        if text is None:
            text = ""
        else:
            index += 1
        all_texts += text
        ret = {"event": "message", "retry": 15000, "data": text}
        yield json.dumps(ret)

    print(all_texts)

    if request.parameters.return_full_text:
        ret = {"event": "message", "retry": 15000, "data": all_texts}
        yield json.dumps(ret)


def send_request(request: ClientRequest):
    print('request: ', request)
    request_id = str(uuid.uuid1())

    if request.parameters is None:
        request.parameters = Parameters()

    if request.parameters.do_sample is None:
        request.parameters.do_sample = False
    if request.parameters.top_k is None:
        request.parameters.top_k = 3
    if request.parameters.top_p is None:
        request.parameters.top_p = 1.0
    if request.parameters.temperature is None:
        request.parameters.temperature = 1.0
    if request.parameters.repetition_penalty is None:
        request.parameters.repetition_penalty = 1.0
    if request.parameters.max_new_tokens is None:
        request.parameters.max_new_tokens = 300
    if request.parameters.return_protocol is None:
        request.parameters.return_protocol = "sse"

    if request.parameters.top_k < 0:
        request.parameters.top_k = 0
    #if request.parameters.top_k > 100:
        #request.parameters.top_k = 100
    if request.parameters.top_p < 0.01:
        request.parameters.top_p = 0.01
    if request.parameters.top_p > 1.0:
        request.parameters.top_p = 1.0
    #if request.parameters.max_new_tokens is None:
        # request.parameters.max_new_tokens = 128

    params = {
        "prompt": request.inputs,
        "do_sample": request.parameters.do_sample,
        "top_k": request.parameters.top_k,
        "top_p": request.parameters.top_p,
        "temperature": request.parameters.temperature,
        "repetition_penalty": request.parameters.repetition_penalty,
        "max_token_len": request.parameters.max_new_tokens
    }
    print('generate_answer...')
    global llm_server
    results = llm_server.generate_answer(request_id, **params)
    return results


@app.post("/models/llama2")
async def async_generator(request: ClientRequest):
    results = send_request(request)

    if request.stream:
        if request.parameters.return_protocol == "sse":
            print('get_stream_res_sse...')
            return EventSourceResponse(get_stream_res_sse(request, results))
        else:
            print('get_stream_res...')
            return StreamingResponse(get_stream_res(request, results))
    else:
        if request.parameters.return_protocol == "sse":
            print('get_full_res_sse...')
            return EventSourceResponse(get_full_res_sse(request, results))
        else:
            print('get_full_res...')
            return StreamingResponse(get_full_res(request, results))


@app.post("/models/llama2/generate")
async def async_full_generator(request: ClientRequest):
    results = send_request(request)
    if request.parameters.return_protocol == "sse":
        print('get_full_res_sse...')
        return EventSourceResponse(get_full_res_sse(request, results))
    else:
        print('get_full_res...')
        return StreamingResponse(get_full_res(request, results))


@app.post("/models/llama2/generate_stream")
async def async_stream_generator(request: ClientRequest):
    results = send_request(request)
    if request.parameters.return_protocol == "sse":
        print('get_stream_res_sse...')
        return EventSourceResponse(get_stream_res_sse(request, results))
    else:
        print('get_stream_res...')
        return StreamingResponse(get_stream_res(request, results))


def update_internlm_request(request: ClientRequest):
    if request.inputs:
        request.inputs = "<s><|User|>:{}<eoh>\n<|Bot|>:".format(request.inputs)


@app.post("/models/internlm")
async def async_internlm_generator(request: ClientRequest):
    update_internlm_request(request)
    return await async_generator(request)


@app.post("/models/internlm/generate")
async def async_internlm_full_generator(request: ClientRequest):
    update_internlm_request(request)
    return await async_full_generator(request)


@app.post("/models/internlm/generate_stream")
async def async_internlm_stream_generator(request: ClientRequest):
    update_internlm_request(request)
    return await async_stream_generator(request)


def init_server_app():
    global llm_server
    llm_server = LLMServer()
    print('init server app finish')


async def warmup(request: ClientRequest):
    request.parameters = Parameters(max_new_tokens=3)
    results = send_request(request)
    print('warmup get_stream_res...')

    async for item in get_stream_res(request, results):
        print(item)


def warmup_llama2():
    request = ClientRequest(inputs="test")
    asyncio.run(warmup(request))
    print('warmup llama2 finish')


def warmup_internlm():
    request = ClientRequest(inputs="test")
    update_internlm_request(request)
    asyncio.run(warmup(request))
    print('warmup internlm finish')


def run_server_app():
    print('server port is: ', SERVER_APP_PORT)
    uvicorn.run(app, host=SERVER_APP_HOST, port=SERVER_APP_PORT)


WARMUP_MODEL_MAP = {
    "llama": warmup_llama2,
    "internlm": warmup_internlm,
}


def warmup_model(model_name):
    model_prefix = model_name.split('_')[0]
    if model_prefix in WARMUP_MODEL_MAP.keys():
        func = WARMUP_MODEL_MAP[model_prefix]
        warmup_process = Process(target=func)
        warmup_process.start()
        warmup_process.join()
        print("mindspore serving is started.")
    else:
        print("model not support warmup : ", model_name)


async def _get_batch_size():
    global llm_server
    batch_size = llm_server.get_bs_current()
    ret = {'event': "message", "retry": 15000, "data": batch_size}
    yield json.dumps(ret)


async def _get_request_numbers():
    global llm_server
    queue_size = llm_server.get_queue_current()
    ret = {'event': "message", "retry": 15000, "data": queue_size}
    yield json.dumps(ret)


@app.get("/serving/get_bs")
async def async_full_generator():
    return EventSourceResponse(_get_batch_size())


@app.get("/serving/get_request_numbers")
async def async_full_generator():
    return EventSourceResponse(_get_request_numbers())


if __name__ == "__main__":
    init_server_app()
    # warmup_model(ModelName)
    run_server_app()
