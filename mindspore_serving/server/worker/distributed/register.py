# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Serving, distributed worker register"""

from mindspore_serving import log as logger
from mindspore_serving.server.common import check_type
from mindspore_serving.server.register.utils import get_servable_dir
from mindspore_serving.server.register.model import append_declared_model
from mindspore_serving._mindspore_serving import ModelMeta_, ServableRegister_


def declare_servable(rank_size, stage_size, with_batch_dim=True, without_batch_dim_inputs=None,
                     enable_pipeline_infer=False):
    """declare distributed servable in servable_config.py. For details, please refer to
    `MindSpore Serving-based Distributed Inference Service Deployment
    <https://www.mindspore.cn/serving/docs/en/r2.0/serving_distributed_example.html>`_.

    Args:
        rank_size (int): The rank size of the distributed model.
        stage_size (int): The stage size of the distributed model.
        with_batch_dim (bool, optional): Whether the first shape dim of the inputs and outputs of model is batch.
            Default: True.
        without_batch_dim_inputs (Union[int, tuple[int], list[int]], optional): Index of inputs that without batch dim
            when with_batch_dim is True. Default: None.
        enable_pipeline_infer (bool, optional): Whether to enable pipeline parallel inference. Pipeline parallelism can
            effectively improve inference performance. For details, see
            `Pipeline Parallelism
            <https://www.mindspore.cn/tutorials/experts/en/r2.0/parallel/pipeline_parallel.html>`_.
            Default: False.

    Return:
        Model, identification of this model, can be used for `Model.call` or as the inputs of `add_stage`.

    Raises:
        RuntimeError: The type or value of the parameters are invalid.

    Examples:
        >>> from mindspore_serving.server import distributed
        >>> model = distributed.declare_servable(rank_size=8, stage_size=1)
    """
    check_type.check_bool('with_batch_dim', with_batch_dim)
    check_type.check_bool('enable_pipeline_infer', enable_pipeline_infer)

    meta = ModelMeta_()
    meta.common_meta.servable_name = get_servable_dir()
    meta.common_meta.model_key = get_servable_dir()  # used to identify model
    meta.common_meta.with_batch_dim = with_batch_dim
    if without_batch_dim_inputs:
        without_batch_dim_inputs = check_type.check_and_as_int_tuple_list('without_batch_dim_inputs',
                                                                          without_batch_dim_inputs, 0)
        meta.common_meta.without_batch_dim_inputs = without_batch_dim_inputs

    # init distributed servable meta info
    check_type.check_int("rank_size", rank_size, 1)
    check_type.check_int("stage_size", stage_size, 1)
    meta.distributed_meta.rank_size = rank_size
    meta.distributed_meta.stage_size = stage_size
    meta.distributed_meta.enable_pipeline_infer = enable_pipeline_infer
    ServableRegister_.declare_distributed_model(meta)
    logger.info(f"Declare distributed servable, servable name: {meta.common_meta.model_key} "
                f", rank_size: {rank_size} , stage_size: {stage_size},  with_batch_dim: {with_batch_dim} "
                f", without_batch_dim_inputs: {without_batch_dim_inputs} "
                f", enable_pipeline_infer: {enable_pipeline_infer}")
    return append_declared_model(meta.common_meta.model_key)
