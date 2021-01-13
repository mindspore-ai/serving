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

from mindspore_serving._mindspore_serving import PostprocessStorage_
from mindspore_serving.worker.common import get_servable_dir, get_func_name
from mindspore_serving import log as logger


def check_postprocess(postprocess_name, inputs_count, outputs_count):
    """Check whether inputs and outputs count is equal with last registered"""
    postprocess_info = PostprocessStorage_.get_instance().get_pycpp_postprocess_info(postprocess_name)
    if not postprocess_info:
        return
    last_inputs_count, last_output_count = postprocess_info
    if last_inputs_count != inputs_count:
        raise RuntimeError(f" Postprocess '{postprocess_name}' inputs count {inputs_count} not match "
                           f"last registered count {last_inputs_count}")
    if last_output_count != outputs_count:
        raise RuntimeError(f" Postprocess '{postprocess_name}' outputs count {outputs_count} not match "
                           f"last registered count {last_output_count}")


class PostprocessStorage:
    """Register and get postprocess info, postprocess info include: func, name, input and output count"""

    def __init__(self):
        self.postprocess = {}
        self.storage = PostprocessStorage_.get_instance()

    def clear(self):
        self.postprocess = {}

    def register(self, fun, postprocess_name, inputs_count, outputs_count):
        check_postprocess(postprocess_name, inputs_count, outputs_count)
        self.postprocess[postprocess_name] = {"fun": fun, "inputs_count": inputs_count, "outputs_count": outputs_count}
        self.storage.register(postprocess_name, inputs_count, outputs_count)

    def get(self, postprocess_name):
        postprocess = self.postprocess.get(postprocess_name, None)
        if postprocess is None:
            raise RuntimeError(f"Postprocess '{postprocess_name}' not found")
        return postprocess


postprocess_storage = PostprocessStorage()


def register_postprocess(func, inputs_count, outputs_count):
    """register postprocess"""
    servable_name = get_servable_dir()
    func_name = get_func_name(func)
    name = servable_name + "." + func_name

    logger.info(f"Register postprocess {name} {inputs_count} {outputs_count}")
    postprocess_storage.register(func, name, inputs_count, outputs_count)
