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

from mindspore_serving import master
from mindspore_serving import worker
from common import ServingTestBase, serving_test


# start_servable_in_master
@serving_test
def test_start_servable_in_master_servable_dir_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        worker.start_servable_in_master(base.servable_dir + "_error", base.servable_name, 0)
        assert False
    except RuntimeError as e:
        assert "Load servable config failed, directory " in str(e)


@serving_test
def test_start_servable_in_master_servable_name_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name + "_error", 0)
        assert False
    except RuntimeError as e:
        assert "Load servable config failed, directory " in str(e)


@serving_test
def test_start_servable_in_master_version_number_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name, version_number=2)
        assert False
    except RuntimeError as e:
        assert "Start servable failed, there is no servable of the specified version number, " \
               "specified version number: " in str(e)


@serving_test
def test_start_servable_in_master_version_number_invalid2_failed():
    base = ServingTestBase()
    base.init_servable(0, "add_servable_config.py")
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name, version_number=0)
        assert False
    except RuntimeError as e:
        assert "Start servable failed, there is no servable of the specified version number, " \
               "specified version number: " in str(e)


@serving_test
def test_start_servable_in_master_no_servable_config_file_failed():
    base = ServingTestBase()
    base.init_servable(1, "no_exist_servable_config.py")
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name, version_number=0)
        assert False
    except RuntimeError as e:
        assert "Load servable config failed, file " in str(e)


@serving_test
def test_start_servable_in_master_no_model_file_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py", model_file="tensor_add_error.mindir")
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name, version_number=0)
        assert False
    except RuntimeError as e:
        assert "Load model failed, servable directory: " in str(e)


@serving_test
def test_start_servable_in_master_type_servable_dir_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        worker.start_servable_in_master(1, base.servable_name, version_number=0)
        assert False
    except RuntimeError as e:
        assert "Parameter 'servable_directory' should be str, but actually " in str(e)


@serving_test
def test_start_servable_in_master_servable_dir_empty_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        worker.start_servable_in_master("", base.servable_name, version_number=0)
        assert False
    except RuntimeError as e:
        assert "Parameter 'servable_directory' should not be empty str" in str(e)


@serving_test
def test_start_servable_in_master_type_servable_name_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        worker.start_servable_in_master(base.servable_dir, False, version_number=0)
        assert False
    except RuntimeError as e:
        assert "Parameter 'servable_name' should be str, but actually " in str(e)


@serving_test
def test_start_servable_in_master_type_version_number_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name, version_number=False)
        assert False
    except RuntimeError as e:
        assert "Parameter 'version_number' should be int, but actually " in str(e)


@serving_test
def test_start_servable_in_master_version_number_invalid_range_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name, version_number=-1)
        assert False
    except RuntimeError as e:
        assert "Parameter 'version_number' should be >= 0" in str(e)


@serving_test
def test_start_servable_in_master_type_device_id_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name, version_number=1, device_id="1")
        assert False
    except RuntimeError as e:
        assert "Parameter 'device_id' should be int, but actually" in str(e)


@serving_test
def test_start_servable_in_master_device_id_range_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name, version_number=1, device_id=-1)
        assert False
    except RuntimeError as e:
        assert "Parameter 'device_id' should be >= 0" in str(e)


@serving_test
def test_start_servable_in_master_type_device_type_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name, version_number=1, device_type=123)
        assert False
    except RuntimeError as e:
        assert "Parameter 'device_type' should be str, but actually" in str(e)


@serving_test
def test_start_servable_in_master_device_type_value_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name, device_type="InvalidDeviceType")
        assert False
    except RuntimeError as e:
        assert "Unsupport device type 'InvalidDeviceType'" in str(e)


@serving_test
def test_start_servable_in_master_device_type_value_invalid2_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name, device_type="")
        assert False
    except RuntimeError as e:
        assert "Parameter 'device_type' should not be empty str" in str(e)


@serving_test
def test_start_servable_in_master_type_device_type_none_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    worker.start_servable_in_master(base.servable_dir, base.servable_name, device_type=None)


@serving_test
def test_start_servable_in_master_type_device_type_none2_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    worker.start_servable_in_master(base.servable_dir, base.servable_name, device_type='None')


# start_servable
@serving_test
def test_start_worker_no_servable_config_file_failed():
    base = ServingTestBase()
    base.init_servable(1, "no_exist_servable_config.py")
    master.start_master_server()
    try:
        worker.start_servable(base.servable_dir, base.servable_name, version_number=0)
        assert False
    except RuntimeError as e:
        assert "Load servable config failed, file " in str(e)


@serving_test
def test_start_worker_no_model_file_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py", model_file="tensor_add_error.mindir")
    master.start_master_server()
    try:
        worker.start_servable(base.servable_dir, base.servable_name, version_number=0)
        assert False
    except RuntimeError as e:
        assert "Load model failed, servable directory: " in str(e)


@serving_test
def test_start_worker_type_servable_dir_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    master.start_master_server()
    try:
        worker.start_servable(1, base.servable_name, version_number=0)
        assert False
    except RuntimeError as e:
        assert "Parameter 'servable_directory' should be str, but actually " in str(e)


@serving_test
def test_start_worker_type_servable_name_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    master.start_master_server()
    try:
        worker.start_servable(base.servable_dir, False, version_number=0)
        assert False
    except RuntimeError as e:
        assert "Parameter 'servable_name' should be str, but actually " in str(e)


@serving_test
def test_start_worker_type_version_number_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    master.start_master_server()
    try:
        worker.start_servable(base.servable_dir, base.servable_name, version_number=False)
        assert False
    except RuntimeError as e:
        assert "Parameter 'version_number' should be int, but actually " in str(e)


@serving_test
def test_start_worker_version_number_invalid_range_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    master.start_master_server()
    try:
        worker.start_servable(base.servable_dir, base.servable_name, version_number=-1)
        assert False
    except RuntimeError as e:
        assert "Parameter 'version_number' should be >= 0" in str(e)


@serving_test
def test_start_worker_type_device_id_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    master.start_master_server()
    try:
        worker.start_servable(base.servable_dir, base.servable_name, version_number=1, device_id="1")
        assert False
    except RuntimeError as e:
        assert "Parameter 'device_id' should be int, but actually" in str(e)


@serving_test
def test_start_worker_device_id_range_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    master.start_master_server()
    try:
        worker.start_servable(base.servable_dir, base.servable_name, version_number=1, device_id=-1)
        assert False
    except RuntimeError as e:
        assert "Parameter 'device_id' should be >= 0" in str(e)


@serving_test
def test_start_worker_type_device_type_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    master.start_master_server()
    try:
        worker.start_servable(base.servable_dir, base.servable_name, version_number=1, device_type=123)
        assert False
    except RuntimeError as e:
        assert "Parameter 'device_type' should be str, but actually" in str(e)


@serving_test
def test_start_worker_device_type_value_invalid_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    master.start_master_server()
    try:
        worker.start_servable(base.servable_dir, base.servable_name, device_type="InvalidDeviceType")
        assert False
    except RuntimeError as e:
        assert "Unsupport device type 'InvalidDeviceType'" in str(e)


@serving_test
def test_start_worker_device_type_value_invalid2_failed():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    master.start_master_server()
    try:
        worker.start_servable(base.servable_dir, base.servable_name, device_type="")
        assert False
    except RuntimeError as e:
        assert "Parameter 'device_type' should not be empty str" in str(e)


@serving_test
def test_start_worker_type_device_type_none_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    master.start_master_server()
    worker.start_servable(base.servable_dir, base.servable_name, device_type=None)


@serving_test
def test_start_worker_type_device_type_none2_success():
    base = ServingTestBase()
    base.init_servable(1, "add_servable_config.py")
    master.start_master_server()
    worker.start_servable(base.servable_dir, base.servable_name, device_type='None')
