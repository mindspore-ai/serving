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

from common import ServingTestBase, serving_test
from mindspore_serving import server
from mindspore_serving.server._servable_local import merge_config


@serving_test
def test_start_servable_servable_dir_invalid_failed():
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
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        server.start_servables(server.ServableStartConfig("", base.servable_name, device_ids=0, version_number=0))
        assert False
    except RuntimeError as e:
        assert "Parameter 'servable_directory' should not be empty str" in str(e)


@serving_test
def test_start_worker_type_servable_dir_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        server.start_servables(server.ServableStartConfig(1, base.servable_name, device_ids=0, version_number=0))
        assert False
    except RuntimeError as e:
        assert "Parameter 'servable_directory' should be str, but actually " in str(e)


@serving_test
def test_start_worker_type_servable_name_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, False, device_ids=0, version_number=0))
        assert False
    except RuntimeError as e:
        assert "Parameter 'servable_name' should be str, but actually " in str(e)


@serving_test
def test_start_servable_version_number_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        server.start_servables(
            server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0, version_number=2))
        assert False
    except RuntimeError as e:
        assert "There is no servable of the specified version number, " \
               "specified version number: " in str(e)


@serving_test
def test_start_servable_version_number_invalid2_failed():
    base = ServingTestBase()
    base.init_servable(0, "add_servable_config.py")
    try:
        server.start_servables(
            server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0, version_number=0))
        assert False
    except RuntimeError as e:
        assert "There is no valid version of servable, " in str(e)


@serving_test
def test_start_worker_type_version_number_invalid_failed():
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
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_servables(
        server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0, device_type=None))


@serving_test
def test_start_worker_type_device_type_none2_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    server.start_servables(
        server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0, device_type='None'))


@serving_test
def test_servable_start_config_merge_same_version_same_device_ids_success():
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
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    config0 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=1, version_number=0)
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
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    shutil.copytree(os.path.join(base.servable_dir, base.servable_name, "1"),
                    os.path.join(base.servable_dir, base.servable_name, "2"))
    config0 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=2, version_number=0)
    config1 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=2, version_number=1)
    try:
        server.start_servables((config0, config1))
    except RuntimeError as e:
        assert "Ascend 910 device id 2 is used repeatedly in servable" in str(e)


@serving_test
def test_servable_start_config_same_servable_name_diff_directory_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    config0 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=2, version_number=0)
    config1 = server.ServableStartConfig(base.servable_dir + "2", base.servable_name, device_ids=2, version_number=1)
    try:
        server.start_servables((config0, config1))
    except RuntimeError as e:
        assert f"The servable directory of servable name {base.servable_name} is different in multiple configurations" \
               in str(e)


@serving_test
def test_servable_start_config_multi_servable_same_device_id():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")

    shutil.copytree(os.path.join(base.servable_dir, base.servable_name),
                    os.path.join(base.servable_dir, base.servable_name + "2"))

    config0 = server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=2, version_number=0)
    config1 = server.ServableStartConfig(base.servable_dir, base.servable_name + "2", device_ids=2, version_number=1)
    try:
        server.start_servables((config0, config1))
    except RuntimeError as e:
        assert "Ascend 910 device id 2 is used repeatedly in servable" in str(e)


@serving_test
def test_servable_start_config_multi_servable_diff_device_id():
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
