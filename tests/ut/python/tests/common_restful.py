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
"""test Serving, Common"""

import json
import requests
import numpy as np

from common import init_str_servable, init_bytes_servable, init_bool_int_float_servable
from mindspore_serving import server


def compare_float_value(result, expect):
    if isinstance(expect, (float, int)):
        assert isinstance(result, float)
        assert abs(expect - result) < 0.001
        return
    expect = np.array(expect)
    result = np.array(result)
    assert (np.abs(expect - result) < 0.001).all()


def create_multi_instances_fp32(instance_count):
    instances = []
    # instance 1
    y_data_list = []
    for i in range(instance_count):
        x1 = np.asarray([[1.1, 2.2], [3.3, 4.4]]).astype(np.float32) * (i + 1)
        x2 = np.asarray([[5.5, 6.6], [7.7, 8.8]]).astype(np.float32) * (i + 1)
        y_data_list.append(x1 + x2)
        instances.append({"x1": x1.tolist(), "x2": x2.tolist()})
    return instances, y_data_list


def check_number_result(result, y_data_list, output_name="y"):
    result = result["instances"]
    assert len(result) == len(y_data_list)
    for result_item, expected_item in zip(result, y_data_list):
        result_item = np.array(result_item[output_name])
        print("result", result_item)
        print("expect:", expected_item)
        assert result_item.shape == expected_item.shape
        assert (np.abs(result_item - expected_item) < 0.001).all()


def post_restful(address, servable_name, method_name, json_instances, version_number=None):
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


def start_str_restful_server():
    base = init_str_servable()
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_restful_server("0.0.0.0:5500")
    return base


def start_bytes_restful_server():
    base = init_bytes_servable()
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_restful_server("0.0.0.0:5500")
    return base


def start_bool_int_float_restful_server():
    base = init_bool_int_float_servable()
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_restful_server("0.0.0.0:5500")
    return base
