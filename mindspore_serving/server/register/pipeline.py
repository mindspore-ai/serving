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
"""Pipelineing registration interface"""

from mindspore_serving._mindspore_serving import PipelineStorage_, RequestSpec_
from mindspore_serving import log as logger
from .utils import get_servable_dir, get_func_name

class PipelineStorage:
    """Register and get pipeline info, pipeline info include: func, name, input and output count"""
    def __init__(self):
        self.pipeline = {}
        self.is_register = False

    def clear(self):
        self.pipeline = {}
        self.is_register = False

    def has_registered(self):
        return self.is_register

    def register(self, fun, pipeline_name, inputs_count, outputs_count):
        self.pipeline[pipeline_name] = {"fun": fun, "inputs_count": inputs_count, "outputs_count": outputs_count}
        self.is_register = True

    def get(self, pipeline_name):
        pipeline = self.pipeline.get(pipeline_name, None)
        if pipeline is None:
            raise RuntimeError(f"Pipeline '{pipeline_name}' not found")
        return pipeline

pipeline_storage = PipelineStorage()

def register_pipeline_args(func, inputs_count, outputs_count):
    """register pipeline"""
    servable_name = get_servable_dir()
    func_name = get_func_name(func)
    name = servable_name + "." + func_name

    logger.info(f"Register pipeline {name} {inputs_count} {outputs_count}")
    pipeline_storage.register(func, name, inputs_count, outputs_count)

class PipelineServable:
    """Create Pipeline Servable for Servable calls"""
    def __init__(self, servable_name, method, version_number=0):
        self.spec = RequestSpec_()
        self.storage = PipelineStorage_.get_instance()
        self.spec.servable_name = servable_name
        self.spec.method_name = method
        self.spec.version_number = version_number

    def run(self, *args):
        """
        Servable calls function in Pipeline register function.

        Args:
            servable_name (str): The name of servable.
            method (str): The name of method supplied by servable.
            version_number (str): The number of version supplied by servable.

        Raises:
            RuntimeError: The type or value of the parameters is invalid, or other errors happened.

        Examples:
            >>> from mindspore_serving.server import distributed
            >>> from mindspore_serving.server import register
            >>>
            >>> distributed.declare_servable(rank_size=8, stage_size=1, with_batch_dim=False)
            >>> @register.register_method(output_names=["y"])
            >>> def fun(x):
            ...     y = register.call_servable(x)
            ...     return y
            >>> servable = register.PipelineServable(servable_name="service", method="fun", version_number=0)
            >>> @register.register_pipeline(output_names=["y"])
            >>> def predict(x):
            ...     y = servable.call(x)
            ...     return y
        """
        return self.storage.run(self.spec, tuple(args))