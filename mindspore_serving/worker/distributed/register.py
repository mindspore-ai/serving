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

from mindspore_serving._mindspore_serving import ServableMeta_, ServableStorage_
from mindspore_serving.worker import check_type
from mindspore_serving.worker.common import get_servable_dir
from mindspore_serving import log as logger


def declare_distributed_servable(rank_size, stage_size, with_batch_dim, without_batch_dim_inputs):
    """declare distributed servable in servable_config.py"""
    check_type.check_bool('with_batch_dim', with_batch_dim)

    meta = ServableMeta_()
    meta.common_meta.servable_name = get_servable_dir()
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
    ServableStorage_.declare_distributed_servable(meta)
    logger.info(f"Declare distributed servable, servable_name: {meta.common_meta.servable_name} "
                f", rank_size: {rank_size} , stage_size: {stage_size},  with_batch_dim: {with_batch_dim} "
                f", without_batch_dim_inputs: {without_batch_dim_inputs}")
