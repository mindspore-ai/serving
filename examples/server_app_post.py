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
import argparse
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from mindspore_serving.client.client_utils import ClientRequest, Parameters, ValidatorUtil
from mindspore_serving.config.config import ServingConfig, check_valid_config
from mindspore_serving.server.llm_server_post import LLMServer
from mindspore_serving.serving_utils.constant import *

logging.basicConfig(level=logging.ERROR,
                    filename='./output/server_app.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

llm_server = None
config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # init LLMServer
    if not check_valid_config(config):
        yield
    global llm_server
    llm_server = LLMServer(config)
    yield
    llm_server.stop()
    print('---------------server app is done---------------')


app = FastAPI(lifespan=lifespan)


async def get_full_res(request, results):
    all_texts = ''
    tokens_list = []
    finish_reason = ""
    output_tokens_len = 0
    async for result in results:
        finish_reason = result.finish_reason
        output_tokens_len = result.output_tokens_len
        for index, output in enumerate(result.outputs):
            answer_texts = output.text
            if answer_texts is None:
                continue

            res_list = {
                "id": output.index,
                "logprob": output.logprob,
                "special": output.special,
                "text": answer_texts
            }

            tokens_list.append(res_list)
            all_texts += answer_texts

    ret = {
        "generated_text": all_texts,
        "finish_reason": finish_reason,
        "generated_tokens": output_tokens_len,
        "prefill": [tokens_list[0]],
        "seed": 0,
        "tokens": tokens_list,
        "top_tokens": [
            [tokens_list[0]]
        ],
        "details": None
    }
    yield (json.dumps(ret, ensure_ascii=False) + '\n').encode("utf-8")


async def get_full_res_sse(request, results):
    all_texts = ''
    tokens_list = []
    finish_reason = ""
    output_tokens_len = 0
    async for result in results:
        finish_reason = result.finish_reason
        output_tokens_len = result.output_tokens_len
        for index, output in enumerate(result.outputs):
            answer_texts = output.text
            if answer_texts is None:
                continue

            res_list = {
                "id": output.index,
                "logprob": output.logprob,
                "special": output.special,
                "text": answer_texts
            }
            tokens_list.append(res_list)
            all_texts += answer_texts

    ret = {
        "event": "message",
        "retry": 30000,
        "generated_text": all_texts,
        "finish_reason": finish_reason,
        "generated_tokens": output_tokens_len,
        "prefill": [tokens_list[0]],
        "seed": 0,
        "tokens": tokens_list,
        "top_tokens": [
            [tokens_list[0]]
        ],
        "details": None
    }
    yield (json.dumps(ret, ensure_ascii=False) + '\n').encode("utf-8")


async def get_stream_res(request, results):
    all_texts = ""
    tokens_list = []
    finish_reason = ""
    output_tokens_len = 0
    token_index = 0
    async for result in results:
        finish_reason = result.finish_reason
        output_tokens_len = result.output_tokens_len
        for index, output in enumerate(result.outputs):
            answer_texts = output.text
            if answer_texts is None:
                continue
            else:
                token_index += 1

            res_list = {
                "id": output.index,
                "logprob": output.logprob,
                "special": output.special,
                "text": answer_texts
            }
            tokens_list.append(res_list)
            all_texts += answer_texts

            ret = {
                "details": None,
                "generated_text": answer_texts,
                "tokens": res_list,
                "top_tokens": [
                    res_list
                ],
            }
            logging.debug("get_stream_res one token_index is {}".format(token_index))
            yield ("data:" + json.dumps(ret, ensure_ascii=False) + '\n').encode("utf-8")

    return_full_text = request.parameters.return_full_text
    if return_full_text:
        ret = {
            "generated_text": all_texts,
            "finish_reason": finish_reason,
            "generated_tokens": output_tokens_len,
            "prefill": [tokens_list[0]],
            "seed": 0,
            "tokens": tokens_list,
            "top_tokens": [
                [tokens_list[0]]
            ],
            "details": None
        }
        yield ("data:" + json.dumps(ret, ensure_ascii=False) + '\n').encode("utf-8")


async def get_stream_res_sse(request, results):
    all_texts = ""
    tokens_list = []
    finish_reason = ""
    output_tokens_len = 0
    token_index = 0
    async for result in results:
        finish_reason = result.finish_reason
        output_tokens_len = result.output_tokens_len
        for index, output in enumerate(result.outputs):
            answer_texts = output.text
            if answer_texts is None:
                continue
            else:
                token_index += 1

            res_list = {
                "id": output.index,
                "logprob": output.logprob,
                "special": output.special,
                "text": answer_texts
            }
            tokens_list.append(res_list)
            all_texts += answer_texts

            tmp_ret = {
                "details": None,
                "generated_text": answer_texts,
                "tokens": res_list,
                "top_tokens": [
                    res_list
                ]
            },
            ret = {
                "event": "message",
                "retry": 30000,
                "data": tmp_ret
            }
            logging.debug("get_stream_res one token_index is {}".format(token_index))
            yield (json.dumps(ret, ensure_ascii=False) + '\n').encode("utf-8")

    return_full_text = request.parameters.return_full_text
    if return_full_text:
        full_tmp_ret = {
            "details": None,
            "generated_text": all_texts,
            "finish_reason": finish_reason,
            "generated_tokens": output_tokens_len,
            "prefill": [tokens_list[0]],
            "seed": 0,
            "tokens": tokens_list,
            "top_tokens": [
                [tokens_list[0]]
            ]
        },
        ret = {
            "event": "message",
            "retry": 30000,
            "data": full_tmp_ret
        }
        yield (json.dumps(ret, ensure_ascii=False) + '\n').encode("utf-8")


def send_request(request: ClientRequest):
    print('request: ', request)

    if config.model_config.model_name == 'internlm_7b':
        request.inputs = INTERNLM_PROMPT_FORMAT.format(request.inputs)
    elif config.model_config.model_name == 'baichuan2pa':    
        request.inputs = BAICHUAN_PROMPT_FORMAT.format(request.inputs)

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

    if request.parameters.decoder_input_details is None:
        request.parameters.decoder_input_details = False
    if request.parameters.details is None:
        request.parameters.details = False
    if request.parameters.return_full_text is None:
        request.parameters.return_full_text = False
    if request.parameters.seed is None:
        request.parameters.seed = 0
    if request.parameters.stop is None:
        request.parameters.stop = []
    if request.parameters.top_n_tokens is None:
        request.parameters.top_n_tokens = 0
    if request.parameters.truncate is None:
        request.parameters.truncate = False
    if request.parameters.typical_p is None:
        request.parameters.typical_p = 0
    if request.parameters.watermark is None:
        request.parameters.watermark = False

    if not ValidatorUtil.validate_top_k(request.parameters.top_k, config.model_config.vocab_size):
        request.parameters.top_k = 1

    if not ValidatorUtil.validate_top_p(request.parameters.top_p) and request.parameters.top_p < 0.01:
        request.parameters.top_p = 0.01

    if not ValidatorUtil.validate_top_p(request.parameters.top_p) and request.parameters.top_p > 1.0:
        request.parameters.top_p = 1.0

    if not ValidatorUtil.validate_temperature(request.parameters.temperature):
        request.parameters.temperature = 1.0
        request.parameters.do_sample = False

    params = {
        "prompt": request.inputs,
        "do_sample": request.parameters.do_sample,
        "top_k": request.parameters.top_k,
        "top_p": request.parameters.top_p,
        "temperature": request.parameters.temperature,
        "repetition_penalty": request.parameters.repetition_penalty,
        "max_token_len": request.parameters.max_new_tokens,
    }

    print('params: ', params)
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
            return EventSourceResponse(get_stream_res_sse(request, results),
                                       media_type="text/event-stream",
                                       ping_message_factory=lambda: ServerSentEvent(
                                           **{"comment": "You can't see this ping"}),
                                       ping=600)
        else:
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
    if request.parameters.return_protocol == "sse":
        print('get_stream_res_sse...')
        return EventSourceResponse(get_stream_res_sse(request, results),
                                   media_type="text/event-stream",
                                   ping_message_factory=lambda: ServerSentEvent(
                                       **{"comment": "You can't see this ping"}),
                                   ping=600)
    else:
        print('get_stream_res...')
        return StreamingResponse(get_stream_res(request, results))


def update_internlm_request(request: ClientRequest):
    if request.inputs:
        request.inputs = "<s><|User|>:{}<eoh>\n<|Bot|>:".format(request.inputs)


@app.post("/models/internlm")
async def async_internlm_generator(request: ClientRequest):
    # update_internlm_request(request)
    return await async_generator(request)


@app.post("/models/internlm/generate")
async def async_internlm_full_generator(request: ClientRequest):
    # update_internlm_request(request)
    return await async_full_generator(request)


@app.post("/models/internlm/generate_stream")
async def async_internlm_stream_generator(request: ClientRequest):
    # update_internlm_request(request)
    return await async_stream_generator(request)


def run_server_app(config):
    print('server port is: ', config.server_port)
    uvicorn.run(app, host=config.server_ip, port=config.server_port)


async def _get_batch_size():
    global llm_server
    batch_size = llm_server.get_bs_current()
    ret = {'event': "message", "retry": 30000, "data": batch_size}
    yield json.dumps(ret, ensure_ascii=False)


async def _get_request_numbers():
    global llm_server
    queue_size = llm_server.get_queue_current()
    ret = {'event': "message", "retry": 30000, "data": queue_size}
    yield json.dumps(ret, ensure_ascii=False)


async def _get_serverd_model_info():
    global llm_server
    serverd_model_info = llm_server.get_serverd_model_info()
    ret = {
        "docker_label": serverd_model_info.docker_label,
        "max_batch_total_tokens": serverd_model_info.max_batch_total_tokens,
        "max_concurrent_requests": serverd_model_info.max_concurrent_requests,
        "max_input_length": serverd_model_info.max_input_length,
        "max_total_tokens": serverd_model_info.max_total_tokens,
        "model_device_type": "CANN",
        "model_dtype": serverd_model_info.model_dtype,
        "model_id": serverd_model_info.model_id,
        "model_pipeline_tag": "text-generation",
        "version": "2.3"
    }

    print(ret)
    yield json.dumps(ret, ensure_ascii=False)


@app.get("/serving/get_bs")
async def get_batch_size():
    return EventSourceResponse(_get_batch_size(),
                               media_type="text/event-stream",
                               ping_message_factory=lambda: ServerSentEvent(**{"comment": "You can't see this ping"}),
                               ping=600)


@app.get("/serving/get_request_numbers")
async def get_request_numbers():
    return EventSourceResponse(_get_request_numbers(),
                               media_type="text/event-stream",
                               ping_message_factory=lambda: ServerSentEvent(**{"comment": "You can't see this ping"}),
                               ping=600)


@app.get("/serving/get_serverd_model_info")
async def get_serverd_model_info():
    return EventSourceResponse(_get_serverd_model_info(),
                               media_type="text/event-stream",
                               ping_message_factory=lambda: ServerSentEvent(**{"comment": "You can't see this ping"}),
                               ping=600)


if __name__ == "__main__":
    work_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help='YAML config files')
    args = parser.parse_args()
    config = ServingConfig(args.config)
    run_server_app(config.serving_config)
