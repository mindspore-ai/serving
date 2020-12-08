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
"""Context setting interface"""
from mindspore_serving._mindspore_serving import Context_


class Context:
    def __init__(self):
        self.context_ = Context_.get_instance()

    def set_device_type(self, device_type):
        self.context_.set_device_type_str(device_type)

    def set_device_id(self, device_id):
        self.context_.set_device_id(device_id)


_k_context = None


def _context():
    """
    Get the global _context, if context is not created, create a new one.

    Returns:
        _Context, the global context in PyNative mode.
    """
    global _k_context
    if _k_context is None:
        _k_context = Context()
    return _k_context


def set_context(**kwargs):
    """The context setting interf. The acceptable parameters including:
    device_type: 'Ascend','Davinci', 'None'. Case ignored.
                 'Davinci' and 'Ascend' are the same.
                 'None' means depend on mindspore register 
    device_id: reasonable vice id
    """
    context = _context()
    for (k, w) in kwargs.items():
        if k == "device_type":
            context.set_device_type(w)
        elif k == "device_id":
            context.set_device_id(w)
        else:
            raise RuntimeError(f"Not support context key '{k}'")
