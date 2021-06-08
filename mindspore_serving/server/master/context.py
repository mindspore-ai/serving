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
"""Set context of serving"""
from mindspore_serving._mindspore_serving import MasterContext_
from mindspore_serving.server.common import check_type

__all__ = ["set_max_enqueued_requests"]

_context = MasterContext_.get_instance()


def set_max_enqueued_requests(max_enqueued_requests):
    r"""
    Set the maximum number of requests waiting to be processed.

    Args:
        max_enqueued_requests (int): The maximum acceptable infer message size in number, default 10000,
            Max infer number should be a positive integer.

    Raises:
        RuntimeError: The type or value of the parameters are invalid, or other error happened.
    """
    check_type.check_int("max_enqueued_requests", max_enqueued_requests, 1)
    _context.set_max_enqueued_requests(max_enqueued_requests)
