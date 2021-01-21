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

import os
from functools import wraps
from shutil import rmtree
from mindspore_serving import worker
from mindspore_serving import master

servable_index = 0


class ServingTestBase:
    def __init__(self):
        servable_dir = "serving_python_ut_servables"
        self.servable_dir = os.path.join(os.getcwd(), servable_dir)
        rmtree(self.servable_dir, True)

    def init_servable(self, version_number, config_file, model_file="tensor_add.mindir"):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_abs = os.path.join(os.path.join(cur_dir, "../servable_config/"), config_file)
        try:
            with open(config_file_abs, "r") as fp:
                servable_config_content = fp.read()
        except FileNotFoundError:
            servable_config_content = None
        self.init_servable_with_servable_config(version_number, servable_config_content, model_file)

    def init_servable_with_servable_config(self, version_number, servable_config_content,
                                           model_file="tensor_add.mindir"):
        global servable_index
        self.servable_name = "add_" + str(servable_index)
        servable_index += 1

        self.version_number = version_number
        self.model_file_name = model_file
        self.servable_name_path = os.path.join(self.servable_dir, self.servable_name)
        self.version_number_path = os.path.join(self.servable_name_path, str(version_number))
        self.model_file_name_path = os.path.join(self.version_number_path, model_file)

        try:
            os.mkdir(self.servable_dir)
        except FileExistsError:
            pass
        try:
            os.mkdir(self.servable_name_path)
        except FileExistsError:
            pass
        try:
            os.mkdir(self.version_number_path)
        except FileExistsError:
            pass
        with open(self.model_file_name_path, "w") as fp:
            print("model content", file=fp)
        if servable_config_content is not None:
            config_file = os.path.join(self.servable_name_path, "servable_config.py")
            with open(config_file, "w") as fp:
                fp.write(servable_config_content)


def serving_test(func):
    @wraps(func)
    def wrap_test(*args, **kwargs):
        try:
            func(*args, **kwargs)
        finally:
            master.stop()
            worker.stop()
            servable_dir = os.path.join(os.getcwd(), "serving_python_ut_servables")
            rmtree(servable_dir, True)

    return wrap_test


def release_client(client):
    del client.stub
    client.stub = None


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
