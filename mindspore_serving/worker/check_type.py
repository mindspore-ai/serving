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
"""T check for worker"""


def check_and_as_str_tuple_list(strs):
    """Check whether the input parameters are reasonable multiple str inputs,
    which can be single str, tuple or list of str.
    finally, return tuple of str"""
    if isinstance(strs, str):
        strs = (strs,)

    if not isinstance(strs, (tuple, list)):
        raise RuntimeError("Check failed, expecting str or tuple/list of str, actually", type(strs))

    if isinstance(strs, (tuple, list)):
        for item in strs:
            if not isinstance(item, str):
                raise RuntimeError("Check failed, expecting tuple/st to be str, actually", type(item))

    return tuple(strs)


def check_str(str_val):
    """Check whether the input parameters are reasonable str input"""
    if not isinstance(str_val, str):
        raise RuntimeError("Check str failed, expecting str, actually", type(str_val))


def check_bool(bool_val):
    """Check whether the input parameters are reasonable bool input"""
    if not isinstance(bool_val, bool):
        raise RuntimeError("Check bool failed, expecting bool, actually", type(bool_val))


def check_int(int_val):
    """Check whether the input parameters are reasonable int input"""
    if not isinstance(int_val, int):
        raise RuntimeError("Check failed, expecting int, actually", {type(int_val)})
