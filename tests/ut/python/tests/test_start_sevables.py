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
import shutil
import os
import numpy as np

from common import ServingTestBase, serving_test, start_serving_server, create_client
from mindspore_serving import server
from mindspore_serving.server._servable_local import merge_config


@serving_test
def test_start_servable_servable_dir_invalid_failed():
    """
    Feature: test start servables
    Description: servable dir is not exist
    Expectation: failed to serving server.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        server.start_servables(
            server.ServableStartConfig(base.servable_dir + "_error", base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert "Check servable config failed, directory " in str(e)


# start_servable
@serving_test
def test_start_worker_no_servable_config_file_failed():
    """
    Feature: test start servables
    Description: servable_config.py is not exist
    Expectation: failed to serving server.
    """
    base = ServingTestBase()
    base.init_servable(1, "no_exist_servable_config.py")
    try:
        server.start_servables(
            server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0, version_number=0))
        assert False
    except RuntimeError as e:
        assert "Check servable config failed, file " in str(e)


@serving_test
def test_start_worker_no_model_file_failed():
    """
    Feature: test start servables
    Description: model file is not exist
    Expectation: failed to serving server.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py", model_file="tensor_add_error.mindir")
    try:
        server.start_servables(
            server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0, version_number=0))
        assert False
    except RuntimeError as e:
        assert "Load model failed, servable directory: " in str(e)


@serving_test
def test_start_servable_servable_dir_empty_invalid_failed():
    """
    Feature: test start servables
    Description: input parameter 'servable_directory' invalid
    Expectation: failed to serving server.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        server.start_servables(server.ServableStartConfig("", base.servable_name, device_ids=0, version_number=0))
        assert False
    except RuntimeError as e:
        assert "Parameter 'servable_directory' should not be empty str" in str(e)


@serving_test
def test_start_worker_type_servable_dir_invalid_failed():
    """
    Feature: test start servables
    Description: input parameter 'servable_directory' invalid
    Expectation: failed to serving server.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        server.start_servables(server.ServableStartConfig(1, base.servable_name, device_ids=0, version_number=0))
        assert False
    except RuntimeError as e:
        assert "Parameter 'servable_directory' should be str, but actually " in str(e)


@serving_test
def test_start_worker_type_servable_name_invalid_failed():
    """
    Feature: test start servables
    Description: input parameter 'servable_name' invalid
    Expectation: failed to serving server.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, False, device_ids=0, version_number=0))
        assert False
    except RuntimeError as e:
        assert "Parameter 'servable_name' should be str, but actually " in str(e)


@serving_test
def test_start_servable_version_number_invalid_failed():
    """
    Feature: test start servables
    Description: There is no specified version model
    Expectation: failed to serving server.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        server.start_servables(
            server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0, version_number=2))
        assert False
    except RuntimeError as e:
        assert "There is no specified version directory of models, specified version number: 2" in str(e)


@serving_test
def test_start_servable_version_number_invalid2_failed():
    """
    Feature: test start servables
    Description: There is no valid version directory
    Expectation: failed to serving server.
    """
    base = ServingTestBase()
    base.init_servable(0, "add_servable_config.py")
    try:
        server.start_servables(
            server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0, version_number=0))
        assert False
    except RuntimeError as e:
        assert "There is no valid version directory of models" in str(e)


@serving_test
def test_start_worker_type_version_number_invalid_failed():
    """
    Feature: test start servables
    Description: input parameter 'version_number' invalid
    Expectation: failed to serving server.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        server.start_servables(
            server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0, version_number=False))
        assert False
    except RuntimeError as e:
        assert "Parameter 'version_number' should be int, but actually " in str(e)

    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        server.start_servables(
            server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0, version_number=-1))
        assert False
    except RuntimeError as e:
        assert "Parameter 'version_number' should be >= 0" in str(e)


@serving_test
def test_start_worker_type_device_id_invalid_failed():
    """
    Feature: test start servables
    Description: input parameter 'device_ids' invalid
    Expectation: failed to serving server.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        server.start_servables(
            server.ServableStartConfig(base.servable_dir, base.servable_name, version_number=1, device_ids="1"))
        assert False
    except RuntimeError as e:
        assert "Parameter 'device_ids' should be int or tuple/list of int, but actually" in str(e)


@serving_test
def test_start_worker_device_id_range_invalid_failed():
    """
    Feature: test start servables
    Description: input parameter 'device_ids' invalid
    Expectation: failed to serving server.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        server.start_servables(
            server.ServableStartConfig(base.servable_dir, base.servable_name, version_number=1, device_ids=-1))
        assert False
    except RuntimeError as e:
        assert "The item value '-1' in parameter 'device_ids' should be >= 0" in str(e)


@serving_test
def test_start_worker_type_device_type_invalid_failed():
    """
    Feature: test start servables
    Description: input parameter 'device_type' invalid
    Expectation: failed to serving server.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        server.start_servables(
            server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0, version_number=1,
                                       device_type=123))
        assert False
    except RuntimeError as e:
        assert "Parameter 'device_type' should be str, but actually" in str(e)


@serving_test
def test_start_worker_device_type_value_invalid_failed():
    """
    Feature: test start servables
    Description: input parameter 'device_type' invalid
    Expectation: failed to serving server.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0,
                                                          device_type="InvalidDeviceType"))
        assert False
    except RuntimeError as e:
        assert "is inconsistent with current running environment, supported device type: 'None' or 'Ascend'" in str(e)


@serving_test
def test_start_worker_device_type_value_invalid2_failed():
    """
    Feature: test start servables
    Description: input parameter 'device_type' invalid
    Expectation: failed to serving server.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        server.start_servables(
            server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0, device_type=""))
        assert False
    except RuntimeError as e:
        assert "Parameter 'device_type' should not be empty str" in str(e)


@serving_test
def test_start_worker_type_device_type_none_success():
    """
    Feature: test start servables
    Description: input parameter 'device_type' invalid
    Expectation: failed to serving server.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_servables(
        server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0, device_type=None))


@serving_test
def test_start_worker_type_device_type_none2_success():
    """
    Feature: test start servables
    Description: input parameter 'device_type' invalid
    Expectation: failed to serving server.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_servables(
        server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0, device_type='None'))


@serving_test
def test_servable_start_config_merge_same_version_same_device_ids_success():
    """
    Feature: test merge servable config
    Description: specified version 1 and newest version 0 can merge to one config of version 1
    Expectation: success to merge config.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    config0 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=2, version_number=0)
    config1 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=2, version_number=1)
    config_ret = merge_config((config0, config1))
    assert len(config_ret) == 1
    assert config_ret[0].version_number == 1
    assert len(config_ret[0].device_ids) == 1
    assert config_ret[0].device_ids[0] == 2


@serving_test
def test_servable_start_config_merge_same_version_diff_device_ids_success():
    """
    Feature: test merge servable config
    Description: specified version 1 with diff device can merge to one config with device_ids merged
    Expectation: success to merge config.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    config0 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=1, version_number=1)
    config1 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=(0, 2), version_number=1)
    config_ret = merge_config((config0, config1))
    assert len(config_ret) == 1
    assert config_ret[0].version_number == 1
    assert len(config_ret[0].device_ids) == 3
    assert 0 in config_ret[0].device_ids
    assert 1 in config_ret[0].device_ids
    assert 2 in config_ret[0].device_ids


@serving_test
def test_servable_start_config_merge_diff_version_diff_device_ids_success():
    """
    Feature: test merge servable config
    Description: specified version 1 and newest version 0 with diff device can merge to one config of version 1 with
        device_ids merged
    Expectation: success to merge config.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    shutil.copytree(os.path.join(base.servable_dir, base.servable_name, "1"),
                    os.path.join(base.servable_dir, base.servable_name, "2"))
    config0 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=1, version_number=0)
    config1 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=2, version_number=1)
    config_ret = merge_config((config0, config1))
    assert len(config_ret) == 2
    assert config_ret[0].version_number == 2  # newest version
    assert len(config_ret[0].device_ids) == 1
    assert config_ret[0].device_ids[0] == 1

    assert config_ret[1].version_number == 1
    assert len(config_ret[1].device_ids) == 1
    assert config_ret[1].device_ids[0] == 2


@serving_test
def test_servable_start_config_merge_diff_version_same_device_ids_failed():
    """
    Feature: test merge servable config
    Description: specified version 1 and newest version 0 with same device is invalid
    Expectation: failed to merge config.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    shutil.copytree(os.path.join(base.servable_dir, base.servable_name, "1"),
                    os.path.join(base.servable_dir, base.servable_name, "2"))
    config0 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=2, version_number=0)
    config1 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=2, version_number=1)
    try:
        server.start_servables((config0, config1))
        assert False
    except RuntimeError as e:
        assert "Ascend 910 device id 2 is used repeatedly in servable" in str(e)


@serving_test
def test_servable_start_config_same_servable_name_diff_directory_failed():
    """
    Feature: test merge servable config
    Description: specified version 1 and newest version 0 with diff servable directory is invalid
    Expectation: failed to merge config.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    config0 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=2, version_number=0)
    config1 = server.ServableStartConfig(base.servable_dir + "2", base.servable_name, device_ids=2, version_number=1)
    try:
        server.start_servables((config0, config1))
        assert False
    except RuntimeError as e:
        assert f"The servable directory of servable name {base.servable_name} is different in multiple configurations" \
               in str(e)


@serving_test
def test_servable_start_config_multi_servable_same_device_id():
    """
    Feature: test merge servable config
    Description: diff servable same with same device id is invalid
    Expectation: failed to merge config.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")

    shutil.copytree(os.path.join(base.servable_dir, base.servable_name),
                    os.path.join(base.servable_dir, base.servable_name + "2"))

    config0 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=2, version_number=0)
    config1 = server.ServableStartConfig(base.servable_dir, base.servable_name + "2", device_ids=2, version_number=1)
    try:
        server.start_servables((config0, config1))
        assert False
    except RuntimeError as e:
        assert "Ascend 910 device id 2 is used repeatedly in servable" in str(e)


@serving_test
def test_servable_start_config_multi_servable_diff_device_id():
    """
    Feature: test merge servable config
    Description: servable name are same, some are diff
    Expectation: success to merge config.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")

    shutil.copytree(os.path.join(base.servable_dir, base.servable_name),
                    os.path.join(base.servable_dir, base.servable_name + "2"))

    config0 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=(1, 3), version_number=0)
    config1 = server.ServableStartConfig(base.servable_dir, base.servable_name + "2", device_ids=2, version_number=1)
    config3 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=(4, 5), version_number=0)
    config_ret = merge_config((config0, config1, config3))
    assert len(config_ret) == 2
    print(config_ret[0].servable_name)
    print(config_ret[1].servable_name)

    assert config_ret[0].version_number == 1  # newest version
    assert len(config_ret[0].device_ids) == 4
    assert tuple(config_ret[0].device_ids) == (1, 3, 4, 5)

    assert config_ret[1].version_number == 1
    assert len(config_ret[1].device_ids) == 1
    assert config_ret[1].device_ids[0] == 2


@serving_test
def test_servable_start_config_merge_diff_version_diff_dec_key_success():
    """
    Feature: test merge servable config
    Description: diff version with diff dec key
    Expectation: success to merge config.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    shutil.copytree(os.path.join(base.servable_dir, base.servable_name, "1"),
                    os.path.join(base.servable_dir, base.servable_name, "2"))
    config0 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=1, version_number=0,
                                         dec_key=("ABC" * 8).encode(), dec_mode='AES-GCM')
    config1 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=2, version_number=1,
                                         dec_key=("DEF" * 8).encode(), dec_mode='AES-CBC')
    config_ret = merge_config((config0, config1))
    assert len(config_ret) == 2
    assert config_ret[0].dec_key == ("ABC" * 8).encode()  # newest version
    assert config_ret[0].dec_mode == "AES-GCM"

    assert config_ret[1].dec_key == ("DEF" * 8).encode()  # newest version
    assert config_ret[1].dec_mode == "AES-CBC"


@serving_test
def test_servable_start_config_merge_same_version_diff_dec_key_failed():
    """
    Feature: test merge servable config
    Description: same version with diff dec key
    Expectation: failed to merge config.
    """
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    config0 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=1, version_number=0,
                                         dec_key=("ABC" * 8).encode(), dec_mode='AES-GCM')
    config1 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=2, version_number=1,
                                         dec_key=("DEF" * 8).encode(), dec_mode='AES-CBC')
    try:
        server.start_servables((config0, config1))
        assert False
    except RuntimeError as e:
        assert "The dec key or dec mode of servable name" in str(e)


@serving_test
def test_servable_start_config_with_dec_success():
    """
    Feature: test start servable with dec
    Description: test start servable with dec
    Expectation: success to start servable.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
tensor_add = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")

@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.add_stage(tensor_add, x1, x2, outputs_count=1)
    return y
"""
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content)
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0,
                                                      dec_key="ABCDEFGHABCDEFGH".encode(), dec_mode='AES-GCM'))


@serving_test
def test_start_servables_without_declared_model_none_device_ids_start_version0_success():
    """
    Feature: test start servables
    Description: no models, no device ids, with extra workers, no version directory, start version number 0
    Expectation: serving server running ok.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

def function_test(x1, x2):
    y = x1+x2
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(function_test, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=[], device_ids=None,
                                version_number=0, start_version_number=0)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    y = x1 + x2
    instances = [{"x1": x1, "x2": x2}]

    client = create_client("localhost:5500", base.servable_name, "predict", version_number=1)
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()


@serving_test
def test_start_servables_without_declared_model_none_device_ids_start_version1_success():
    """
    Feature: test start servables
    Description: no models, no device ids, with extra workers, no version directory, start version number 1
    Expectation: serving server running ok.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

def function_test(x1, x2):
    y = x1+x2
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(function_test, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=[], device_ids=None,
                                version_number=0, start_version_number=1)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    y = x1 + x2
    instances = [{"x1": x1, "x2": x2}] * 10

    client = create_client("localhost:5500", base.servable_name, "predict", version_number=1)
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()


@serving_test
def test_start_servables_without_declared_model_with_device_ids_start_version0_success():
    """
    Feature: test start servables
    Description: no models, with device ids, without extra workers, no version directory, start version number 0
    Expectation: serving server running ok.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

def function_test(x1, x2):
    y = x1+x2
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(function_test, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=[], device_ids=0,
                                version_number=0, start_version_number=0)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    y = x1 + x2
    instances = [{"x1": x1, "x2": x2}] * 10

    client = create_client("localhost:5500", base.servable_name, "predict", version_number=1)
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()


@serving_test
def test_start_servables_without_declared_model_with_device_ids_start_version0_with_extra_worker_success():
    """
    Feature: test start servables
    Description: no models, with device ids, without extra workers, no version directory, start version number 0
    Expectation: serving server running ok.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

def function_test(x1, x2):
    y = x1+x2
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(function_test, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=[], device_ids=0, num_parallel_workers=2,
                                version_number=0, start_version_number=0)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    y = x1 + x2
    instances = [{"x1": x1, "x2": x2}] * 10

    client = create_client("localhost:5500", base.servable_name, "predict", version_number=1)
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()


@serving_test
def test_start_servables_without_declared_model_with_device_ids_start_version1_with_extra_worker_success():
    """
    Feature: test start servables
    Description: no models, with device ids, with extra workers, no version directory, start version number 1
    Expectation: serving server running ok.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

def function_test(x1, x2):
    y = x1+x2
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(function_test, x1, x2, outputs_count=1)
    return y
    """
    base = start_serving_server(servable_content, model_file=[], device_ids=0, num_parallel_workers=2,
                                version_number=0, start_version_number=1)
    # Client
    x1 = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    x2 = np.array([[5.5, 6.6], [7.7, 8.8]], np.float32)
    y = x1 + x2
    instances = [{"x1": x1, "x2": x2}] * 10

    client = create_client("localhost:5500", base.servable_name, "predict", version_number=1)
    result = client.infer(instances)
    print("result", result)
    assert (result[0]["y"] == y).all()


@serving_test
def test_start_servables_with_declared_model_none_device_ids_start_version0_with_extra_worker_fail():
    """
    Feature: test start servables
    Description: with models, none device ids, with extra workers, no version directory, start version number 0
    Expectation: failed to serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")
def function_test(x1, x2):
    y = x1+x2
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(function_test, x1, x2, outputs_count=1)
    return y
    """
    try:
        start_serving_server(servable_content, model_file="tensor_add.mindir",
                             device_ids=None, num_parallel_workers=2,
                             version_number=None, start_version_number=0)
        assert False
    except RuntimeError as e:
        assert "There is no valid version directory of models" in str(e)


@serving_test
def test_start_servables_with_declared_model_none_device_ids_start_version1_with_extra_worker_fail():
    """
    Feature: test start servables
    Description: with models, none device ids, with extra workers, no version directory, start version number 1
    Expectation: failed to serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")
def function_test(x1, x2):
    y = x1+x2
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(function_test, x1, x2, outputs_count=1)
    return y
    """
    try:
        start_serving_server(servable_content, model_file="tensor_add.mindir",
                             device_ids=None, num_parallel_workers=2,
                             version_number=None, start_version_number=1)
        assert False
    except RuntimeError as e:
        # "Cannot find model {} version 1 registered"
        assert "There is no valid version directory of models" in str(e)


@serving_test
def test_start_servables_with_declared_model_none_device_ids_start_version0_with_version_dir_fail():
    """
    Feature: test start servables
    Description: with models, none device ids, with extra workers, with version directory, start version number 1
    Expectation: failed to serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")
def function_test(x1, x2):
    y = x1+x2
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(function_test, x1, x2, outputs_count=1)
    return y
    """
    try:
        start_serving_server(servable_content, model_file="tensor_add.mindir",
                             device_ids=None, num_parallel_workers=2,
                             version_number=1, start_version_number=0)
        assert False
    except RuntimeError as e:
        # "Cannot find model {} version 1 registered"
        assert "Cannot find model" in str(e)


@serving_test
def test_start_servables_with_declared_model_none_device_ids_start_version1_with_version_dir_fail():
    """
    Feature: test start servables
    Description: with models, none device ids, with extra workers, with version directory, start version number 1
    Expectation: failed to serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")
def function_test(x1, x2):
    y = x1+x2
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(function_test, x1, x2, outputs_count=1)
    return y
    """
    try:
        start_serving_server(servable_content, model_file="tensor_add.mindir",
                             device_ids=None, num_parallel_workers=2,
                             version_number=1, start_version_number=1)
        assert False
    except RuntimeError as e:
        # "Cannot find model {} version 1 registered"
        assert "Cannot find model" in str(e)


@serving_test
def test_start_servables_with_declared_model_with_device_ids_start_version0_without_version_dir_fail():
    """
    Feature: test start servables
    Description: with models, with device ids, with extra workers, without version directory, start version number 0
    Expectation: failed to serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")
def function_test(x1, x2):
    y = x1+x2
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(function_test, x1, x2, outputs_count=1)
    return y
    """
    try:
        start_serving_server(servable_content, model_file="tensor_add.mindir",
                             device_ids=1, num_parallel_workers=2,
                             version_number=None, start_version_number=0)
        assert False
    except RuntimeError as e:
        # "Cannot find model {} version 1 registered"
        assert "There is no valid version directory of models" in str(e)


@serving_test
def test_start_servables_with_declared_model_with_device_ids_start_version1_without_version_dir_fail():
    """
    Feature: test start servables
    Description: with models, with device ids, with extra workers, without version directory, start version number 1
    Expectation: failed to serving server.
    """
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register

model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")
def function_test(x1, x2):
    y = x1+x2
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.add_stage(function_test, x1, x2, outputs_count=1)
    return y
    """
    try:
        start_serving_server(servable_content, model_file="tensor_add.mindir",
                             device_ids=1, num_parallel_workers=2,
                             version_number=None, start_version_number=1)
        assert False
    except RuntimeError as e:
        # "Cannot find model {} version 1 registered"
        assert "There is no valid version directory of models" in str(e)
