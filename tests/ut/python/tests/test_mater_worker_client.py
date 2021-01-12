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
"""test Serving with master, worker and client"""

import numpy as np
from mindspore_serving import master
from mindspore_serving import worker
from mindspore_serving.client import Client
from common import ServingTestBase, serving_test


def create_multi_instances_fp32(instance_count):
    instances = []
    # instance 1
    y_data_list = []
    for i in range(instance_count):
        x1 = np.asarray([[1.1, 2.2], [3.3, 4.4]]).astype(np.float32) * (i + 1)
        x2 = np.asarray([[5.5, 6.6], [7.7, 8.8]]).astype(np.float32) * (i + 1)
        y_data_list.append(x1 + x2)
        instances.append({"x1": x1, "x2": x2})
    return instances, y_data_list


def check_result(result, y_data_list):
    assert len(result) == len(y_data_list)
    for result_item, y_data in zip(result, y_data_list):
        assert (result_item["y"] == y_data).all()


@serving_test
def test_master_worker_client_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    worker.start_servable_in_master(base.servable_dir, base.servable_name, 0)
    master.start_grpc_server("0.0.0.0", 5500)
    # Client
    client = Client("localhost", 5500, base.servable_name, "add_common")
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)
    result = client.infer(instances)

    print(result)
    check_result(result, y_data_list)


# test servable_config.py with client
servable_config_import = r"""
import numpy as np
from mindspore_serving.worker import register
"""

servable_config_declare_servable = r"""
register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
"""

servable_config_preprocess_cast = r"""
def add_trans_datatype(x1, x2):
    return x1.astype(np.float32), x2.astype(np.float32)
"""

servable_config_method_add_common = r"""
@register.register_method(output_names=["y"])
def add_common(x1, x2):  # only support float32 inputs
    y = register.call_servable(x1, x2)
    return y
"""

servable_config_method_add_cast = r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.call_preprocess(add_trans_datatype, x1, x2)  # cast input to float32
    y = register.call_servable(x1, x2)
    return y
"""


@serving_test
def no_test_master_worker_client_servable_content_success():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += servable_config_method_add_common
    servable_content += servable_config_method_add_cast

    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_grpc_server("0.0.0.0", 5500)
    # Client
    client = Client("localhost", 5500, base.servable_name, "add_common")
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)
    result = client.infer(instances)

    print(result)
    check_result(result, y_data_list)


@serving_test
def no_test_master_worker_client_preprocess_outputs_count_not_match_failed():
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
    worker.start_servable_in_master(base.servable_dir, base.servable_name)
    master.start_grpc_server("0.0.0.0", 5500)
    # Client
    client = Client("localhost", 5500, base.servable_name, "add_cast")
    instance_count = 3
    instances, _ = create_multi_instances_fp32(instance_count)
    result = client.infer(instances)
    print(result)
    assert "Servable stopped" in str(result[0]["error"])
