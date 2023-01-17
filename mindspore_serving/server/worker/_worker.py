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
"""Interface for start up servable"""

import os
import sys
from functools import wraps
from mindspore_serving import log as logger
from mindspore_serving.server.common import check_type, get_abs_path
from mindspore_serving.server.worker import init_mindspore

from mindspore_serving._mindspore_serving import ExitSignalHandle_
from mindspore_serving._mindspore_serving import Worker_
from mindspore_serving._mindspore_serving import ServableContext_
from .task import _start_py_task

_wait_and_clear_thread = None


def _set_enable_lite(enable_lite):
    """Set device id, default 0"""
    ServableContext_.get_instance().set_enable_lite(enable_lite)


def _set_device_id(device_id):
    """Set device id, default 0"""
    ServableContext_.get_instance().set_device_id(device_id)


def _set_device_type(device_type):
    """Set device type, now can be 'None'(default), 'GPU' and 'Ascend', 'Davinci'(same as 'Ascend'), case ignored. """
    if device_type is not None:
        check_type.check_str('device_type', device_type)
        ServableContext_.get_instance().set_device_type_str(device_type)
    else:
        ServableContext_.get_instance().set_device_type_str('None')  # depend on MindSpore build target


def get_newest_version_number(servable_directory, servable_name):
    """Get newest version number of servable"""
    max_version = 0
    servable_directory = get_abs_path(servable_directory)
    version_root_dir = os.path.join(servable_directory, servable_name)
    try:
        files = os.listdir(version_root_dir)
    except FileNotFoundError:
        return 0
    for file in files:
        if not os.path.isdir(os.path.join(version_root_dir, file)):
            continue
        if not file.isdigit() or file == "0" and str(int(file)) != file:
            continue
        version = int(file)
        if max_version < version:
            max_version = version
    return max_version


def stop():
    r"""
    Stop the running of worker.

    Examples:
        >>> import os
        >>> from mindspore_serving import server
        >>>
        >>> servable_dir = os.path.abspath(".")
        >>> config = server.ServableConfig(servable_dir, "lenet", device_ids=0)
        >>> server.start_servables(servable_configs=config)
        >>> server.start_grpc_server("0.0.0.0:5500")
        >>> ...
        >>> server.stop()
    """

    Worker_.stop_and_clear()


def stop_on_except(func):
    """Wrap of clear environment and exit on Serving exception"""

    @wraps(func)
    def handle_except(*args, **kwargs):
        try:
            ExitSignalHandle_.start()  # Set flag to running and receive Ctrl+C message
            func(*args, **kwargs)
        except:
            stop()
            raise

    return handle_except


def _load_servable_config(servable_directory, servable_name):
    """Load servable config named servable_config.py in directory `servable_directory`/`servable_name` """
    config_dir = os.path.join(servable_directory, servable_name)
    if not os.path.isdir(config_dir):
        raise RuntimeError(f"Load servable config failed, directory '{config_dir}' not exist, "
                           f"servable directory '{servable_directory}', servable name '{servable_name}'")
    config_file = os.path.join(config_dir, "servable_config.py")
    if not os.path.isfile(config_file):
        raise RuntimeError(f"Load servable config failed, file '{config_file}' not exist, "
                           f"servable directory '{servable_directory}', servable name '{servable_name}'")
    sys.path.append(servable_directory)
    try:
        __import__(servable_name + ".servable_config")
    except Exception as e:
        logger.error(f"import {servable_name}.servable_config failed, {str(e)}")
        raise RuntimeError(f"import {servable_name}.servable_config failed, {str(e)}")


@stop_on_except
def start_servable(servable_directory, servable_name, version_number,
                   device_type, device_id, master_address, worker_address, dec_key, dec_mode, enable_lite):
    r"""
    Start up the servable named 'servable_name' defined in 'servable_directory', and link the worker to the master
    through gRPC master_address and worker_address.
    """
    check_type.check_str('servable_directory', servable_directory)
    check_type.check_str('servable_name', servable_name)
    check_type.check_int('version_number', version_number, 0)
    check_type.check_int('device_id', device_id, 0)
    check_type.check_str('master_address', master_address)
    check_type.check_str('worker_address', worker_address)
    if dec_key is not None:
        check_type.check_bytes('dec_key', dec_key)
    else:
        dec_key = ''
    check_type.check_str('dec_mode', dec_mode)
    check_type.check_bool('enable_lite', enable_lite)
    _set_enable_lite(enable_lite)

    _load_servable_config(servable_directory, servable_name)
    model_names = Worker_.get_declared_model_names()
    if model_names:
        init_mindspore.init_mindspore_cxx_env(enable_lite)
        newest_version_number = get_newest_version_number(servable_directory, servable_name)
        if not newest_version_number:
            raise RuntimeError(
                f"There is no valid version directory of models while there are models declared in servable_config.py, "
                f"servable directory: {servable_directory}, servable name: {servable_name}")
    if version_number == 0:
        version_number = 1

    _set_device_type(device_type)
    _set_device_id(device_id)
    Worker_.start_servable(servable_directory, servable_name, version_number, master_address, worker_address,
                           dec_key, dec_mode)
    _start_py_task()


@stop_on_except
def start_extra_servable(servable_directory, servable_name, version_number, device_type, device_ids_empty,
                         dec_key, dec_mode, master_address, worker_address, enable_lite):
    r"""
    Start up the servable named 'servable_name' defined in 'servable_directory', and link the worker to the master
    through gRPC master_address and worker_address.
    """
    check_type.check_str('servable_directory', servable_directory)
    check_type.check_str('servable_name', servable_name)
    check_type.check_int('version_number', version_number, 0)
    check_type.check_str('device_type', device_type)
    check_type.check_bool('device_ids_empty', device_ids_empty)
    check_type.check_str('master_address', master_address)
    check_type.check_str('worker_address', worker_address)
    if dec_key is not None:
        check_type.check_bytes('dec_key', dec_key)
    else:
        dec_key = ''
    check_type.check_str('dec_mode', dec_mode)
    check_type.check_bool('enable_lite', enable_lite)
    _set_enable_lite(enable_lite)

    _load_servable_config(servable_directory, servable_name)
    model_names = Worker_.get_declared_model_names()
    if model_names:
        init_mindspore.init_mindspore_cxx_env(enable_lite)
        newest_version_number = get_newest_version_number(servable_directory, servable_name)
        if not newest_version_number:
            raise RuntimeError(
                f"There is no valid version directory of models while there are models declared in servable_config.py, "
                f"servable directory: {servable_directory}, servable name: {servable_name}")
    if version_number == 0:
        version_number = 1

    _set_device_type(device_type)
    Worker_.start_extra_servable(servable_directory, servable_name, version_number, device_ids_empty,
                                 dec_key, dec_mode, master_address, worker_address)
    _start_py_task()
