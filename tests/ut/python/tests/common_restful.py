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


def compare_float_value(expect, result):
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


def create_multi_instances_int32_input_fp32_output(instance_count):
    instances = []
    # instance 1
    y_data_list = []
    for i in range(instance_count):
        x1 = np.asarray([[1.1, 2.2], [3.3, 4.4]]).astype(np.int32) * (i + 1)
        x2 = np.asarray([[5.5, 6.6], [7.7, 8.8]]).astype(np.int32) * (i + 1)
        y_data_list.append((x1 + x2).astype(np.float32))
        instances.append({"x1": x1.tolist(), "x2": x2.tolist()})
    return instances, y_data_list


def check_result(result, y_data_list, output_name="y"):
    result = result["instances"]
    assert len(result) == len(y_data_list)
    for result_item, expected_item in zip(result, y_data_list):
        result_item = np.array(result_item[output_name])
        print("result", result_item)
        print("expect:", expected_item)
        assert result_item.shape == expected_item.shape
        assert (np.abs(result_item - expected_item) < 0.001).all()


def post_restful(ip, restful_port, servable_name, method_name, json_instances, version_number=None):
    instances_map = {"instances": json_instances}
    post_payload = json.dumps(instances_map)
    print("request:", post_payload)
    if version_number is not None:
        request_url = f"http://{ip}:{restful_port}/model/{servable_name}/version/{version_number}:{method_name}"
        result = requests.post(request_url, data=post_payload)
    else:
        request_url = f"http://{ip}:{restful_port}/model/{servable_name}:{method_name}"
        result = requests.post(request_url, data=post_payload)
    print("result", result.text)
    result = json.loads(result.text)
    return result
