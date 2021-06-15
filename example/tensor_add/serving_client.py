# Copyright 2020 Huawei Technologies Co., Ltd
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
"""The client of example add"""

import numpy as np
from mindspore_serving.client import Client


def run_add_common():
    """invoke servable add method add_common"""
    client = Client("127.0.0.1:5500", "add", "add_common")
    instances = []

    # instance 1
    x1 = np.asarray([[1, 1], [1, 1]]).astype(np.float32)
    x2 = np.asarray([[1, 1], [1, 1]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2})

    # instance 2
    x1 = np.asarray([[2, 2], [2, 2]]).astype(np.float32)
    x2 = np.asarray([[2, 2], [2, 2]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2})

    # instance 3
    x1 = np.asarray([[3, 3], [3, 3]]).astype(np.float32)
    x2 = np.asarray([[3, 3], [3, 3]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2})

    result = client.infer(instances)
    print(result)


def run_add_cast():
    """invoke servable add method add_cast"""
    client = Client("127.0.0.1:5500", "add", "add_cast")
    instances = []
    x1 = np.ones((2, 2), np.int32)
    x2 = np.ones((2, 2), np.int32)
    instances.append({"x1": x1, "x2": x2})
    result = client.infer(instances)
    print(result)


def post_restful(address, servable_name, method_name, json_instances, version_number=None):
    """construct and post restful request"""
    import json
    import requests
    instances_map = {"instances": json_instances}
    post_payload = json.dumps(instances_map)
    print("request:", post_payload[:200])
    if version_number is not None:
        request_url = f"http://{address}/model/{servable_name}/version/{version_number}:{method_name}"
        result = requests.post(request_url, data=post_payload)
    else:
        request_url = f"http://{address}/model/{servable_name}:{method_name}"
        result = requests.post(request_url, data=post_payload)
    print("result", result.text[:200])
    result = json.loads(result.text)
    return result


def run_add_restful():
    """run restful request: invoke servable add method add_common"""
    # Client
    print("begin to run add restful.")
    instances = []
    x1 = np.asarray([[1.1, 2.2], [3.3, 4.4]]).astype(np.float32)
    x2 = np.asarray([[5.5, 6.6], [7.7, 8.8]]).astype(np.float32)
    instances.append({"x1": x1.tolist(), "x2": x2.tolist()})

    result = post_restful("localhost:1500", "add", "add_common", instances)
    print(result)


if __name__ == '__main__':
    run_add_common()
    run_add_cast()
    run_add_restful()
