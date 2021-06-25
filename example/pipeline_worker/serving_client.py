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
"""The client of example add"""

import numpy as np
from mindspore_serving.client import Client

def run_predict():
    """invoke servable sub method predict"""
    client = Client("127.0.0.1:5500", "pipeline", "predict")
    instances = []

    # instance 1
    x1 = np.asarray([[1, 1], [1, 1]]).astype(np.float32)
    x2 = np.asarray([[1, 1], [1, 1]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2})

    # instance 2
    x1 = np.asarray([[2, 2], [2, 2]]).astype(np.float32)
    x2 = np.asarray([[2, 2], [2, 2]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2})

    # instance 3
    x1 = np.asarray([[3, 3], [3, 3]]).astype(np.float32)
    x2 = np.asarray([[3, 3], [3, 3]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2})

    result = client.infer(instances)
    print(result)

def run_predict_test():
    """invoke servable sub method predict"""
    client = Client("127.0.0.1:5500", "pipeline", "predict_test")
    instances = []

    # instance 1
    x1 = np.asarray([[1, 1], [1, 1]]).astype(np.float32)
    instances.append({"x": x1})

    # instance 2
    x1 = np.asarray([[2, 2], [2, 2]]).astype(np.float32)
    instances.append({"x": x1})

    # instance 3
    x1 = np.asarray([[3, 3], [3, 3]]).astype(np.float32)
    instances.append({"x": x1})

    result = client.infer(instances)
    print(result)

if __name__ == '__main__':
    run_predict()
    run_predict_test()
