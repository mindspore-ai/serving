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


def check_and_as_str_tuple_list(arg_name, strs):
    """Check whether the input parameters are reasonable multiple str inputs,
    which can be single str, tuple or list of str.
    finally, return tuple of str"""
    if isinstance(strs, str):
        strs = (strs,)

    if not isinstance(strs, (tuple, list)):
        raise RuntimeError(f"Parameter '{arg_name}' should be str or tuple/list of str, but actually {type(strs)}")

    if isinstance(strs, (tuple, list)):
        for item in strs:
            if not isinstance(item, str):
                raise RuntimeError(f"The item of parameter '{arg_name}' should be str, but actually {type(item)}")
            if not item:
                raise RuntimeError(f"The item of parameter '{arg_name}' should not be empty str")

    return tuple(strs)


def check_str(arg_name, str_val):
    """Check whether the input parameters are reasonable str input"""
    if not isinstance(str_val, str):
        raise RuntimeError(f"Parameter '{arg_name}' should be str, but actually {type(str_val)}")
    if not str_val:
        raise RuntimeError(f"Parameter '{arg_name}' should not be empty str")


def check_bool(arg_name, bool_val):
    """Check whether the input parameters are reasonable bool input"""
    if not isinstance(bool_val, bool):
        raise RuntimeError(f"Parameter '{arg_name}' should be bool, but actually {type(bool_val)}")


def check_int(arg_name, int_val, mininum=None, maximum=None):
    """Check whether the input parameters are reasonable int input"""
    if not isinstance(int_val, int):
        raise RuntimeError(f"Parameter '{arg_name}' should be int, but actually {type(int_val)}")
    if mininum is not None and int_val < mininum:
        if maximum is not None:
            raise RuntimeError(f"Parameter '{arg_name}' should be in range [{mininum},{maximum}]")
        raise RuntimeError(f"Parameter '{arg_name}' should be >= {mininum}")
    if maximum is not None and int_val > maximum:
        if mininum is not None:
            raise RuntimeError(f"Parameter '{arg_name}' should be in range [{mininum},{maximum}]")
        raise RuntimeError(f"Parameter '{arg_name}' should be <= {maximum}")


def check_ip_port(arg_name, port):
    """Check whether the input parameters are reasonable ip port"""
    check_int(arg_name, port, 1, 65535)
