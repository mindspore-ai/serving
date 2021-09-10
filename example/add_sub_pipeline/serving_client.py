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
"""The client of example add_sub pipeline"""

import numpy as np

from mindspore_serving.client import Client


def is_float_equal(left, right):
    """Check whether two float numbers are equal"""
    return (np.abs(left-right) < 0.0001).all()


def run_add_sub_only_model():
    """invoke servable add_sub method add_sub_only_model"""
    # x1+x2-x3
    client = Client("127.0.0.1:5500", "add_sub", "add_sub_only_model")
    instances = []

    # instance 1
    x1 = np.asarray([[30, 30], [20, 20]]).astype(np.float32)
    x2 = np.asarray([[20, 20], [20, 20]]).astype(np.float32)
    x3 = np.asarray([[10, 10], [10, 10]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2, "x3": x3})
    expect_y = x1 + x2 - x3

    result = client.infer(instances)
    print(result)
    assert len(result) == len(instances)
    assert is_float_equal(result[0]["y"], expect_y)


def run_add_sub_complex():
    """invoke servable add_sub method add_sub_complex"""
    # x1+x2+1-x3+1
    client = Client("127.0.0.1:5500", "add_sub", "add_sub_complex")
    instances = []

    # instance 1
    x1 = np.asarray([[30, 30], [20, 20]]).astype(np.float32)
    x2 = np.asarray([[20, 20], [20, 20]]).astype(np.float32)
    x3 = np.asarray([[10, 10], [10, 10]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2, "x3": x3})
    expect_y = x1 + x2 + 1 - x3 + 1

    result = client.infer(instances)
    print(result)
    assert len(result) == len(instances)
    assert is_float_equal(result[0]["y"], expect_y)


if __name__ == '__main__':
    run_add_sub_only_model()
    run_add_sub_complex()
