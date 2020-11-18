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


# with_batch_dim means the model first dim is batch, 'inputs mode' request can split into multi instances,
# otherwise, 'inputs mode' request will view as one instance
# model_format: OM, MindIR
def declare_servable(servable_file, model_format, with_batch_dim=True):
    """declare servable's model info, input_names and output_names can be str, tuple or list of str.
    For input_names and output_names, serving only consider the number of names contained in them,
    which should be consistent with the number of input and output used in register_method and
    the number of input and output of the model.
    The specific names content are ignored."""
    check_type.check_str(servable_file)
    check_type.check_str(model_format)
    check_type.check_bool(with_batch_dim)

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
