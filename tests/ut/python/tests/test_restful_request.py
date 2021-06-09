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
"""test Serving RESTful, with master, worker and client"""

import json

import requests
import numpy as np

from common import ServingTestBase, serving_test, generate_cert
from common import servable_config_import, servable_config_declare_servable
from common_restful import create_multi_instances_fp32, check_number_result, post_restful
from mindspore_serving import server


@serving_test
def test_restful_request_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_restful_server("0.0.0.0:5500")
    # Client
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)
    result = post_restful("localhost:5500", base.servable_name, "add_common", instances)
    check_number_result(result, y_data_list)


@serving_test
def test_restful_request_one_way_auth_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    generate_cert()
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    ssl_config = server.SSLConfig(certificate="server.crt", private_key="server.key", custom_ca="ca.crt",
                                  verify_client=False)
    server.start_restful_server("0.0.0.0:5500", ssl_config=ssl_config)
    # Client
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)
    result = post_restful("0.0.0.0:5500", base.servable_name, "add_common", instances, https=True)
    check_number_result(result, y_data_list)


@serving_test
def test_restful_request_mutual_auth_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    generate_cert(server_ip="127.0.0.1")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    ssl_config = server.SSLConfig(certificate="server.crt", private_key="server.key", custom_ca="ca.crt",
                                  verify_client=True)
    server.start_restful_server("0.0.0.0:5500", ssl_config=ssl_config)
    # Client
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)
    result = post_restful("127.0.0.1:5500", base.servable_name, "add_common", instances, https=True)
    check_number_result(result, y_data_list)


@serving_test
def test_restful_request_multi_times_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_restful_server("0.0.0.0:5500")
    for instance_count in range(1, 5):
        instances, y_data_list = create_multi_instances_fp32(instance_count)
        result = post_restful("localhost:5500", base.servable_name, "add_common", instances)
        check_number_result(result, y_data_list)


@serving_test
def test_restful_request_multi_times_int32_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_restful_server("0.0.0.0:5500")

    for instance_count in range(1, 5):
        instances = []
        # instance 1
        y_data_list = []
        for i in range(instance_count):
            x1 = np.asarray([[1.1, 2.2], [3.3, 4.4]]).astype(np.int32) * (i + 1)
            x2 = np.asarray([[5.5, 6.6], [7.7, 8.8]]).astype(np.int32) * (i + 1)
            y_data_list.append((x1 + x2).astype(np.float32))
            instances.append({"x1": x1.tolist(), "x2": x2.tolist()})
        result = post_restful("localhost:5500", base.servable_name, "add_cast", instances)
        check_number_result(result, y_data_list)


@serving_test
def test_restful_request_servable_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_restful_server("0.0.0.0:5500")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    result = post_restful("localhost:5500", base.servable_name + "_error", "add_common", instances)
    assert "servable is not available" in str(result["error_msg"])


@serving_test
def test_restful_request_method_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_restful_server("0.0.0.0:5500")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    result = post_restful("localhost:5500", base.servable_name, "add_common" + "_error", instances)
    assert "method is not available" in str(result["error_msg"])


@serving_test
def test_restful_request_with_version_number_0_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_restful_server("0.0.0.0:5500")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    # Client
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)
    result = post_restful("localhost:5500", base.servable_name, "add_common", instances, 0)
    check_number_result(result, y_data_list)


@serving_test
def test_restful_request_with_version_number_1_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_restful_server("0.0.0.0:5500")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    # Client
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)
    result = post_restful("localhost:5500", base.servable_name, "add_common", instances, 1)
    check_number_result(result, y_data_list)


@serving_test
def test_restful_request_with_version_number_2_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_restful_server("0.0.0.0:5500")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    result = post_restful("localhost:5500", base.servable_name, "add_common", instances, 2)
    assert "servable is not available" in str(result["error_msg"])


@serving_test
def test_restful_request_version_number_negative_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_restful_server("0.0.0.0:5500")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    result = post_restful("localhost:5500", base.servable_name, "add_common", instances, -1)
    assert "please check url, version number range failed" in str(result["error_msg"])


@serving_test
def test_restful_request_without_model_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_restful_server("0.0.0.0:5500")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)

    instances_map = {"instances": instances}
    post_payload = json.dumps(instances_map)
    print("request:", post_payload)
    request_url = "http://localhost:5500/x/:add_common"
    result = requests.post(request_url, data=post_payload)
    print("result", result.text)
    result = json.loads(result.text)
    assert "please check url, the keyword:[model] must contain" in str(result["error_msg"])


@serving_test
def test_restful_request_without_method_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_restful_server("0.0.0.0:5500")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)

    instances_map = {"instances": instances}
    post_payload = json.dumps(instances_map)
    print("request:", post_payload)
    request_url = f"http://localhost:5500/model/{base.servable_name}"
    result = requests.post(request_url, data=post_payload)
    print("result", result.text)
    result = json.loads(result.text)
    assert "please check url, the keyword:[service method] must contain." in str(result["error_msg"])


@serving_test
def test_restful_request_servable_version_reverse_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_restful_server("0.0.0.0:5500")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    # Client
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)

    instances_map = {"instances": instances}
    post_payload = json.dumps(instances_map)
    print("request:", post_payload)
    request_url = f"http://localhost:5500/version/0/model/{base.servable_name}:add_common"
    result = requests.post(request_url, data=post_payload)
    print("result", result.text)
    result = json.loads(result.text)
    check_number_result(result, y_data_list)


@serving_test
def test_restful_request_preprocess_outputs_count_not_match_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def add_trans_datatype(x1, x2):
    return x1.astype(np.float32)

@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.call_preprocess(add_trans_datatype, x1, x2)  # cast input to float32
    y = register.call_servable(x1, x2)    
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_restful_server("0.0.0.0:5500")
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    result = post_restful("localhost:5500", base.servable_name, "add_cast", instances)

    print(result)
    if "error_msg" in result:
        assert "Preprocess Failed" in str(result["error_msg"])
    else:
        assert len(result["instances"]) == 3
        assert "Preprocess Failed" in str(result["instances"][0]["error_msg"])


@serving_test
def test_restful_request_postprocess_outputs_count_not_match_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def add_trans_datatype(x1, x2):
    return x1.astype(np.float32)

@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.call_servable(x1, x2)    
    y, y2 = register.call_postprocess(add_trans_datatype, y, x2)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_restful_server("0.0.0.0:5500")
    # Client
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    result = post_restful("localhost:5500", base.servable_name, "add_cast", instances)
    if "error_msg" in result:
        assert "Postprocess Failed" in str(result["error_msg"])
    else:
        assert len(result["instances"]) == 3
        assert "Postprocess Failed" in str(result["instances"][0]["error_msg"])


@serving_test
def test_restful_request_larger_than_server_receive_max_size():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
    server.start_restful_server("0.0.0.0:5500", max_msg_mb_size=1)  # 1MB
    # Client
    instances = []
    x1 = np.ones([1024, 1024], np.float32)
    x2 = np.ones([1024, 1024], np.float32)
    instances.append({"x1": x1.tolist(), "x2": x2.tolist()})
    # more than 1MB msg
    result = post_restful("localhost:5500", base.servable_name + "_error", "add_common", instances)

    print(result)
    assert "http message is bigger than 1048576" in str(result["error_msg"])
