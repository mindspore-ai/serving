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
"""Init MindSpore Cxx"""
import os
import importlib
from mindspore_serving import log as logger
from ._check_version import check_version_and_env_config

_flag_set_mindspore_cxx_env = False


def _set_mindspore_cxx_env():
    """Append MindSpore CXX lib path to LD_LIBRARY_PATH"""
    model_spec = importlib.util.find_spec("mindspore")
    if not model_spec or not model_spec.submodule_search_locations:
        logger.info(f"find mindspore failed, LD_LIBRARY_PATH will not add MindSpore lib path")
        return
    if not isinstance(model_spec.submodule_search_locations, list):
        logger.info(f"find mindspore failed, LD_LIBRARY_PATH will not add MindSpore lib path, "
                    f"locations: {model_spec.submodule_search_locations}")
        return
    ms_dir = model_spec.submodule_search_locations[0]
    if not ms_dir:
        logger.info(f"find mindspore failed, LD_LIBRARY_PATH will not add MindSpore lib path")
        return
    ms_dir = os.path.join(ms_dir, "lib")
    ld_lib_path = os.getenv('LD_LIBRARY_PATH', "")
    if ld_lib_path:
        os.environ['LD_LIBRARY_PATH'] = ld_lib_path + ":" + ms_dir
    else:
        os.environ['LD_LIBRARY_PATH'] = ms_dir
    logger.info(f"Update env LD_LIBRARY_PATH from '{ld_lib_path}' to '{os.getenv('LD_LIBRARY_PATH')}'")


def init_mindspore_cxx_env():
    """Init env for load libmindspore.so"""
    global _flag_set_mindspore_cxx_env
    if _flag_set_mindspore_cxx_env:
        return
    _flag_set_mindspore_cxx_env = True
    check_version_and_env_config()
    _set_mindspore_cxx_env()
