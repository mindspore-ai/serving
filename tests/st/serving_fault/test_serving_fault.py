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

import os
import pytest
import numpy as np


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_single
def test_serving_fault_kill_15_master():
    """test_serving"""
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(f"sh {sh_path}/kill_15_master.sh")
    assert np.allclose(ret, 0)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_single
def test_serving_fault_kill_9_master():
    """test_serving"""
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(f"sh {sh_path}/kill_9_master.sh")
    assert np.allclose(ret, 0)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_single
def test_serving_fault_kill_15_worker():
    """test_serving"""
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(f"sh {sh_path}/kill_15_worker.sh")
    assert np.allclose(ret, 0)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_single
def test_serving_fault_kill_9_worker():
    """test_serving"""
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(f"sh {sh_path}/kill_9_worker.sh")
    assert np.allclose(ret, 0)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_single
def serving_fault_restart():
    """test_serving"""
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(f"sh {sh_path}/restart.sh")
    assert np.allclose(ret, 0)


if __name__ == '__main__':
    test_serving_fault_kill_9_master()
    test_serving_fault_kill_15_master()
    test_serving_fault_kill_9_worker()
    test_serving_fault_kill_15_worker()
