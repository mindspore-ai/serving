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
import importlib.util

from mindspore_serving import log as logger
from mindspore_serving._mindspore_serving import Worker_
from .check_version import check_version_and_env_config, check_version_and_try_set_env_lib

_flag_set_mindspore_cxx_env = False


def get_mindspore_whl_path():
    """Get MindSpore whl install path"""
    model_spec = importlib.util.find_spec("mindspore")
    if not model_spec or not model_spec.submodule_search_locations:
        return ""
    if not isinstance(model_spec.submodule_search_locations, list):
        return ""
    ms_dir = model_spec.submodule_search_locations[0]
    return ms_dir


def check_mindspore_version(ms_dir):
    """check MindSpore version number"""
    try:
        from mindspore_serving.version import __version__
    except ModuleNotFoundError:
        logger.warning(f"Get MindSpore Serving version failed")
        return
    try:
        with open(os.path.join(ms_dir, "version.py"), "r") as fp:
            version_str = fp.readline().replace("\n", "").replace("\r", "").replace(" ", "") \
                .replace("'", "").replace("\"", "")
            prefix = "__version__="
            if version_str[:len(prefix)] != prefix:
                logger.warning(f"Get MindSpore version failed")
                return
            ms_version = version_str[len(prefix):]
    except FileNotFoundError:
        logger.warning(f"Get MindSpore version failed")
        return
    serving_versions = __version__.split(".")
    ms_versions = ms_version.split(".")
    if serving_versions[:2] != ms_versions[:2]:
        logger.warning(f"MindSpore version {ms_version} and MindSpore Serving version {__version__} are expected "
                       f"to be consistent. If not, there may be compatibility problems.")
        return


def set_mindspore_cxx_env():
    """Append MindSpore CXX lib path to LD_LIBRARY_PATH"""
    global _flag_set_mindspore_cxx_env
    if _flag_set_mindspore_cxx_env:
        return
    _flag_set_mindspore_cxx_env = True

    ld_lib_path = os.getenv('LD_LIBRARY_PATH', "")
    check_version_and_try_set_env_lib()  # try set env LD_LIBRARY_PATH
    logger.info(f"Update env LD_LIBRARY_PATH from '{ld_lib_path}' to '{os.getenv('LD_LIBRARY_PATH')}'")

    ld_lib_path = os.getenv('LD_LIBRARY_PATH', "")
    ms_dir = get_mindspore_whl_path()
    if not ms_dir:
        logger.info(f"find mindspore failed, LD_LIBRARY_PATH will not add MindSpore lib path")
        return
    check_mindspore_version(ms_dir)
    ms_dir = os.path.join(ms_dir, "lib")

    if ld_lib_path:
        if ms_dir not in ld_lib_path.split(":"):
            os.environ['LD_LIBRARY_PATH'] = ld_lib_path + ":" + ms_dir
    else:
        os.environ['LD_LIBRARY_PATH'] = ms_dir
    logger.info(f"Update env LD_LIBRARY_PATH from '{ld_lib_path}' to '{os.getenv('LD_LIBRARY_PATH')}'")


def init_mindspore_cxx_env(enable_lite):
    """Init env for load libmindspore.so"""
    set_mindspore_cxx_env()
    device_type = Worker_.get_device_type("none", enable_lite)
    if not device_type:
        logger.warning("Failed to get device type")
        return
    check_version_and_env_config(device_type)
