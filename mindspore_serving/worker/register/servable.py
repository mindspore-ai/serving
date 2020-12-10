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
"""Servable declaration interface"""

from mindspore_serving._mindspore_serving import ServableMeta_
from mindspore_serving.worker import check_type
from mindspore_serving.worker.common import get_servable_dir
from .method import _servable_storage


def declare_servable(servable_file, model_format, with_batch_dim=True):
    r"""
    declare the servable info.

    Args:
        servable_file (str): Model file name.
        model_format (str): Model format, "OM" or "MindIR", case ignored.
        with_batch_dim (bool): Whether the first shape dim of the inputs and outpus of model is batch dim, default True.

    Raises:
        RuntimeError: The type or value of the parameters is invalid.
    """

    check_type.check_str('servable_file', servable_file)
    check_type.check_str('model_format', model_format)
    check_type.check_bool('with_batch_dim', with_batch_dim)

    model_format = model_format.lower()
    if model_format not in ("om", "mindir"):
        raise RuntimeError("model format can only be OM or MindIR")

    meta = ServableMeta_()
    meta.servable_name = get_servable_dir()
    meta.servable_file = servable_file
    meta.set_model_format(model_format)
    meta.with_batch_dim = with_batch_dim
    _servable_storage.declare_servable(meta)
    print("------------Declare servable: servable_name", meta.servable_name,
          ", servable file", servable_file,
          ", model format", model_format, ", model config with batch dim", with_batch_dim)
