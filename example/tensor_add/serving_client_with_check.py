# Copyright 2021 Huawei Technologies Co., Ltd
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
"""The client of example add with result check"""

import json
import requests
import numpy as np
from mindspore_serving.client import Client


def check_result(result, y_data_list):
    """check grpc output result"""
    assert len(result) == len(y_data_list)
    for result_item, y_data in zip(result, y_data_list):
        assert (np.abs(result_item["y"] - y_data) < 0.00001).all()


def run_add_common():
    """invoke servable add method add_common"""
    client = Client("localhost:5500", "add", "add_common")
    instances = []
    instance_count = 3
    y_data_list = []
    for i in range(instance_count):
        x1 = np.asarray([[1.1, 2.2], [3.3, 4.4]]).astype(np.float32) * (i + 1)
        x2 = np.asarray([[5.5, 6.6], [7.7, 8.8]]).astype(np.float32) * (i + 1)
        y_data_list.append(x1 + x2)
        instances.append({"x1": x1, "x2": x2})

    result = client.infer(instances)
    print(result)
    check_result(result, y_data_list)


def run_add_cast():
    """invoke servable add method add_cast"""
    client = Client("localhost:5500", "add", "add_cast")
    instances = []
    y_data_list = []
    x1 = np.ones((2, 2), np.int32)
    x2 = np.ones((2, 2), np.int32)
    instances.append({"x1": x1, "x2": x2})
    y_data_list.append((x1 + x2).astype(np.float32))
    result = client.infer(instances)
    print(result)
    check_result(result, y_data_list)


def post_restful(address, servable_name, method_name, json_instances, version_number=None):
    """construct post restful request"""
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


def check_number_result(result, y_data_list, output_name="y"):
    """check restful output result"""
    result = result["instances"]
    assert len(result) == len(y_data_list)
    for result_item, expected_item in zip(result, y_data_list):
        result_item = np.array(result_item[output_name])
        print("result", result_item)
        print("expect:", expected_item)
        assert result_item.shape == expected_item.shape
        assert (np.abs(result_item - expected_item) < 0.001).all()


def run_add_restful():
    """run restful request: invoke servable add method add_common"""
    # Client
    print("begin to run add restful.")
    y_data_list = []
    instances = []
    x1 = np.asarray([[1.1, 2.2], [3.3, 4.4]]).astype(np.float32)
    x2 = np.asarray([[5.5, 6.6], [7.7, 8.8]]).astype(np.float32)
    y_data_list.append((x1 + x2).astype(np.float32))
    instances.append({"x1": x1.tolist(), "x2": x2.tolist()})

    result = post_restful("localhost:1500", "add", "add_common", instances)
    check_number_result(result, y_data_list)


if __name__ == '__main__':
    run_add_common()
    run_add_cast()
    run_add_restful()
