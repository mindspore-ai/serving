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
"""Common implement for worker"""
import inspect
import os


def get_servable_dir():
    """Get the directory where servable is located. The name of the directory is the name of servable"""
    stack = inspect.stack()
    for item in stack:
        if item.filename.endswith("servable_config.py"):
            abs_path = os.path.realpath(item.filename)
            last_dir = os.path.split(abs_path)[0]
            last_dir = os.path.split(last_dir)[1]
            if not last_dir:
                continue
            return last_dir
    raise RuntimeError("Failed to obtain the directory of servable_config.py")


def get_func_name(func):
    """Get function name for preprocess and postprocess, as the identification name"""
    return func.__name__
