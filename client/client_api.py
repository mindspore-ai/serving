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
"""test clent post request"""

import json
import os
import requests
import logging

from typing import Dict, Optional, List, AsyncIterator, Iterator
from .client_utils import ClientRequest, Parameters, Response, StreamResponse, Token
from enum import Enum


class BaseClient:

    def __init__(
            self,
            base_url: str,
            headers: Optional[Dict[str, str]] = None,
            cookies: Optional[Dict[str, str]] = None,
            timeout: int = 30000,
    ):

        self.base_url = base_url
        self.headers = headers
        self.cookies = cookies
        self.timeout = timeout

    def generate(
            self,
            prompt: str,
            do_sample: bool = False,
            repetition_penalty: Optional[float] = 1.0,
            temperature: Optional[float] = 1.0,
            top_k: Optional[int] = 1,
            top_p: Optional[float] = 1,
            max_new_tokens: Optional[int] = 300,
    ) -> Response:

        parameters = Parameters(
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        request = ClientRequest(inputs=prompt, stream=False, parameters=parameters)

        resp = requests.post(
            self.base_url,
            json=request.dict(),
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.timeout,
        )
        print('resp.josn: ', resp.json)
        print(resp)

        payload = resp.json()
        print(payload)
        if resp.status_code != 200:
            message = resp.json()["error"]
            raise UnknownError(message)
        return Response(**payload)

    def generate_stream(
            self,
            prompt: str,
            do_sample: bool = True,
            repetition_penalty: Optional[float] = 1.0,
            temperature: Optional[float] = 1.0,
            top_k: Optional[int] = 1,
            top_p: Optional[float] = 1,
            max_new_tokens: Optional[int] = 300,
            return_full_text: Optional[bool] = True,
    ):

        parameters = Parameters(
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            return_full_text=return_full_text,
        )
        request = ClientRequest(inputs=prompt, stream=True, parameters=parameters)

        resp = requests.post(
            self.base_url,
            json=request.dict(),
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.timeout,
            stream=True,
        )

        print(resp)

        if resp.status_code != 200:
            message = resp.json()["error"]
            raise UnknownError(message)

        for byte_payload in resp.iter_lines():
            if byte_payload == b"\n":
                continue

            payload = byte_payload.decode("utf-8")

            if payload.startswith("data:"):
                json_payload = json.loads(payload.lstrip("data:").rstrip("/n"))
                print(json_payload)
                response = StreamResponse(**json_payload)
                yield response


class MindsporeInferenceClient(BaseClient):

    def __init__(self, model_type: str, server_url: str, token: Optional[str] = None, timeout: int = 30000):
        headers = {
            "user-agent": "mindspoer_serving/1.0"
        }
        if token is not None:
            headers["authorization"] = f"Bearer {token}"

        base_url = f"{server_url}/models/{model_type}"

        super(MindsporeInferenceClient, self).__init__(
            base_url, headers=headers, timeout=timeout
        )

