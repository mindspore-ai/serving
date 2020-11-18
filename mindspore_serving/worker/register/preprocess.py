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
"""Preprocessing registration interface"""

from mindspore_serving._mindspore_serving import PreprocessStorage_
from mindspore_serving.worker import check_type
from mindspore_serving.worker.common import get_servable_dir, get_func_name


def check_preprocess(preprocess_name, inputs_count, outputs_count):
    preprocess_info = PreprocessStorage_.get_instance().get_pycpp_preprocess_info(preprocess_name)
    if not preprocess_info:
        return
    last_inputs_count, last_output_count = preprocess_info
    if last_inputs_count != inputs_count:
        raise RuntimeError(f"Preprocess {preprocess_name} inputs count {inputs_count} not match "
                           f"last registered count {last_inputs_count}")
    if last_output_count != outputs_count:
        raise RuntimeError(f"retprocess {preprocess_name} outputs count {outputs_count} not match "
                           f"last registered count {last_output_count}")


class PreprocessStorage:
    def __init__(self):
        self.preprocess = {}
        self.storage = PreprocessStorage_.get_instance()

    def register(self, fun, preprocess_name, inputs_count, outputs_count):
        check_preprocess(preprocess_name, inputs_count, outputs_count)
        self.preprocess[preprocess_name] = {"fun": fun, "inputs_count": inputs_count, "outputs_count": outputs_count}
        self.storage.register(preprocess_name, inputs_count, outputs_count)

    def get(self, preprocess_name):
        preprocess = self.preprocess.get(preprocess_name, None)
        if preprocess is None:
            raise RuntimeError("Preprocess " + preprocess_name + " not found")
        return preprocess


preprocess_storage = PreprocessStorage()


def register_preprocess(inputs_count, outputs_count):
    """register preprocess, input_names and output_names can be str， tuple or list of str.
    For input_names and output_names, serving only consider the number of names contained in them,
    which should be consistent with the number of input and output used in register_method,
    and the specific names content are ignored."""
    check_type.check_int(inputs_count)
    check_type.check_int(outputs_count)

    def register(func):
        servable_name = get_servable_dir()
        func_name = get_func_name(func)
        name = servable_name + "." + func_name

        print("------------Register preprocess", name, inputs_count, outputs_count)
        preprocess_storage.register(func, name, inputs_count, outputs_count)
        return func

    return register
