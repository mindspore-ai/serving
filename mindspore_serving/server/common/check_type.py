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


def check_and_as_tuple_with_str_list(arg_name, strs):
    """Check whether the input parameters are reasonable multiple str inputs,
    which can be single str, tuple or list of str, tuple with list of str.
    Finally, return tuple with list of str.
    """
    if isinstance(strs, str):
        strs = (list(strs),)
        return tuple(strs)

    if not isinstance(strs, (tuple, list)):
        raise RuntimeError(f"Parameter '{arg_name}' should be str or tuple/list of str, but actually {type(strs)}")

    str_list = []
    for item in strs:
        it_list = []
        if isinstance(item, list):
            for inner in item:
                if not isinstance(inner, str):
                    raise RuntimeError(f"The inner of parameter '{arg_name}' should be str, "
                                       f"but actually {type(inner)}")
                if not inner:
                    raise RuntimeError(f"The inner of parameter '{arg_name}' should not be empty str")
                if item in it_list:
                    raise RuntimeError(f"The inner value '{inner}' in parameter '{arg_name}' "
                                       f"should not be repeated")
                it_list.append(inner)
        else:
            if not isinstance(item, str):
                raise RuntimeError(f"The item of parameter '{arg_name}' should be str, but actually {type(item)}")
            if not item:
                raise RuntimeError(f"The item of parameter '{arg_name}' should not be empty str")
            if item in str_list:
                raise RuntimeError(f"The item value '{item}' in parameter '{arg_name}' should not be repeated")
            it_list.append(item)
        str_list.append(it_list)

    return tuple(str_list)


def check_and_as_str_tuple_list(arg_name, strs):
    """Check whether the input parameters are reasonable multiple str inputs,
    which can be single str, tuple or list of str.
    Finally, return tuple of str.
    """
    if isinstance(strs, str):
        strs = (strs,)

    if not isinstance(strs, (tuple, list)):
        raise RuntimeError(f"Parameter '{arg_name}' should be str or tuple/list of str, but actually {type(strs)}")

    str_list = []
    for item in strs:
        if not isinstance(item, str):
            raise RuntimeError(f"The item of parameter '{arg_name}' should be str, but actually {type(item)}")
        if not item:
            raise RuntimeError(f"The item of parameter '{arg_name}' should not be empty str")
        if item in str_list:
            raise RuntimeError(f"The item value '{item}' in parameter '{arg_name}' should not be repeated")
        str_list.append(item)

    return tuple(str_list)


def check_str(arg_name, str_val):
    """Check whether the input parameters are reasonable str input"""
    if not isinstance(str_val, str):
        raise RuntimeError(f"Parameter '{arg_name}' should be str, but actually {type(str_val)}")
    if not str_val:
        raise RuntimeError(f"Parameter '{arg_name}' should not be empty str")


def check_bytes(arg_name, bytes_val):
    """Check whether the input parameters are reasonable bytes input"""
    if not isinstance(bytes_val, bytes):
        raise RuntimeError(f"Parameter '{arg_name}' should be bytes, but actually {type(bytes_val)}")
    if not bytes_val:
        raise RuntimeError(f"Parameter '{arg_name}' should not be empty bytes")


def check_bool(arg_name, bool_val):
    """Check whether the input parameters are reasonable bool input"""
    if not isinstance(bool_val, bool):
        raise RuntimeError(f"Parameter '{arg_name}' should be bool, but actually {type(bool_val)}")


def check_int(arg_name, int_val, minimum=None, maximum=None, is_tuple_item=False):
    """Check whether the input parameters are reasonable int input"""
    if not is_tuple_item:
        prefix = f"Parameter '{arg_name}'"
    else:
        prefix = f"The item value '{int_val}' in parameter '{arg_name}'"

    if isinstance(int_val, bool):
        raise RuntimeError(f"{prefix} should be int, but actually {type(int_val)}")
    if not isinstance(int_val, int):
        raise RuntimeError(f"{prefix} should be int, but actually {type(int_val)}")
    if minimum is not None and int_val < minimum:
        if maximum is not None:
            raise RuntimeError(f"{prefix} should be in range [{minimum},{maximum}]")
        raise RuntimeError(f"{prefix} should be >= {minimum}")
    if maximum is not None and int_val > maximum:
        if minimum is not None:
            raise RuntimeError(f"{prefix} should be in range [{minimum},{maximum}]")
        raise RuntimeError(f"{prefix} should be <= {maximum}")


def check_ip_port(arg_name, port):
    """Check whether the input parameters are reasonable ip port"""
    check_int(arg_name, port, 1, 65535)


def check_and_as_int_tuple_list(arg_name, ints, minimum=None, maximum=None):
    """Check whether the input parameters are reasonable multiple int inputs,
    which can be single int, tuple or list of int.
    Finally, return tuple of int.
    """
    if isinstance(ints, int):
        ints = (ints,)

    if not isinstance(ints, (tuple, list)):
        raise RuntimeError(f"Parameter '{arg_name}' should be int or tuple/list of int, but actually {type(ints)}")

    int_list = []
    for item in ints:
        if item in int_list:
            raise RuntimeError(f"The item value '{item}' in parameter '{arg_name}' should not be repeated")
        check_int(arg_name, item, minimum, maximum, True)
        int_list.append(item)

    return tuple(int_list)


def check_int_tuple_list(arg_name, ints, minimum=None, maximum=None):
    """Check whether the input parameters are reasonable multiple int inputs,
    which can be single tuple or list of int.
    Finally, return tuple of int.
    """
    if not isinstance(ints, (tuple, list)):
        raise RuntimeError(f"Parameter '{arg_name}' should be tuple/list of int, but actually {type(ints)}")

    int_list = []
    for item in ints:
        if item in int_list:
            raise RuntimeError(f"The item value '{item}' in parameter '{arg_name}' should not be repeated")
        check_int(arg_name, item, minimum, maximum, True)
        int_list.append(item)
