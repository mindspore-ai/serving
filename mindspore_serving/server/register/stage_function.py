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
"""Postprocessing registration interface"""

from mindspore_serving._mindspore_serving import StageFunctionStorage_
from mindspore_serving import log as logger
from .utils import get_servable_dir, get_func_name


def check_stage_function(method_name, function_name, inputs_count, outputs_count):
    """Check whether inputs and outputs count is equal with last registered"""
    func_info = get_stage_info(function_name)
    if not func_info:
        return
    last_inputs_count, last_output_count = func_info
    if last_inputs_count != inputs_count:
        raise RuntimeError(f"Stage function '{function_name}' inputs count {inputs_count} not match "
                           f"last registered count {last_inputs_count}, method name '{method_name}'")
    if last_output_count != outputs_count:
        raise RuntimeError(f"Stage function '{function_name}' outputs count {outputs_count} not match "
                           f"last registered count {last_output_count}, method name '{method_name}'")


def get_stage_info(function_name):
    """Get cpp and python function inputs and outputs count"""
    func_info = StageFunctionStorage_.get_instance().get_pycpp_function_info(function_name)
    if not func_info:
        return None
    return func_info


class StageFunctionStorage:
    """Register and get stage function info: func, name, input and output count"""

    def __init__(self):
        self.function = {}
        self.storage = StageFunctionStorage_.get_instance()

    def register(self, method_name, fun, function_name, inputs_count, outputs_count, use_with_size):
        check_stage_function(method_name, function_name, inputs_count, outputs_count)
        if function_name in self.function:
            if self.function[function_name]["use_with_size"] != use_with_size:
                raise RuntimeError(f"Failed to add stage function {function_name}: parameter 'batch_size' in "
                                   f"multiple 'add_stage' should be enabled or disabled consistently")
        self.function[function_name] = {"fun": fun, "inputs_count": inputs_count, "outputs_count": outputs_count,
                                        "use_with_size": use_with_size}
        self.storage.register(function_name, inputs_count, outputs_count)

    def get(self, function_name):
        func = self.function.get(function_name, None)
        if func is None:
            raise RuntimeError(f"Stage function '{function_name}' not found")
        return func


stage_function_storage = StageFunctionStorage()


def register_stage_function(method_name, func, inputs_count, outputs_count, use_with_size):
    """register stage function"""
    servable_name = get_servable_dir()
    func_name = get_func_name(func)
    name = servable_name + "." + func_name

    logger.info(f"Register stage function {name} {inputs_count} {outputs_count}, use batch size: {use_with_size}")
    stage_function_storage.register(method_name, func, name, inputs_count, outputs_count, use_with_size)
