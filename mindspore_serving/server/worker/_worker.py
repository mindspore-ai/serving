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

import threading
from functools import wraps
from mindspore_serving import log as logger
from mindspore_serving.server.common import check_type
from mindspore_serving.server.worker import init_mindspore
from mindspore_serving._mindspore_serving import ExitSignalHandle_
from mindspore_serving._mindspore_serving import Worker_
from mindspore_serving._mindspore_serving import ServableContext_
from .register.preprocess import preprocess_storage
from .register.postprocess import postprocess_storage
from .task import _start_py_task, _join_py_task

_wait_and_clear_thread = None


def _clear_python():
    """Clear python storage data"""
    preprocess_storage.clear()
    postprocess_storage.clear()


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


def _start_wait_and_clear():
    """Waiting for Ctrl+C, and clear up environment"""

    def thread_func():
        Worker_.wait_and_clear()
        _join_py_task()
        _clear_python()
        logger.info("Serving worker: exited ------------------------------------")
        print("Serving worker: exited ------------------------------------")

    global _wait_and_clear_thread
    if not _wait_and_clear_thread:
        _wait_and_clear_thread = threading.Thread(target=thread_func)
        _wait_and_clear_thread.start()


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
    _join_py_task()
    _clear_python()


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

    import sys
    import os
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
                   device_type, device_id,
                   master_address, worker_address):
    r"""
    Start up the servable named 'servable_name' defined in 'servable_directory', and link the worker to the master
    through gRPC (master_ip, master_port).

    Serving has two running modes. One is running in a single process, providing the Serving service of a single model.
    The other includes a master and multiple workers. This interface is for the second scenario.

    The master is responsible for providing the Serving access interface for clients,
    while the worker is responsible for providing the inference service of the specific model. The communications
    between the master and workers through gPRC are defined as (master_ip, master_port) and (worker_ip, worker_port).

    Args:
        servable_directory (str): The directory where the servable is located in. There expects to has a directory
            named `servable_name`. For more detail:
            `How to config Servable <https://www.mindspore.cn/tutorial/inference/zh-CN/master/serving_model.html>`_ .

        servable_name (str): The servable name.
        version_number (int): Servable version number to be loaded. The version number should be a positive integer,
            starting from 1, and 0 means to load the latest version. Default: 0.
        device_type (str): Currently only supports "Ascend", "Davinci" and None, Default: None.
            "Ascend" means the device type can be Ascend910 or Ascend310, etc.
            "Davinci" has the same meaning as "Ascend".
            None means the device type is determined by the MindSpore environment.
        device_id (int): The id of the device the model loads into and runs in.
        master_address (str): The master socket address the worker linked to.
        worker_address (str): The worker socket address the master linked to.
    """
    check_type.check_str('servable_directory', servable_directory)
    check_type.check_str('servable_name', servable_name)
    check_type.check_int('version_number', version_number, 0)

    check_type.check_int('device_id', device_id, 0)

    check_type.check_str('master_address', master_address)
    check_type.check_str('worker_address', worker_address)

    init_mindspore.init_mindspore_cxx_env()
    _load_servable_config(servable_directory, servable_name)

    _set_device_type(device_type)
    _set_device_id(device_id)
    Worker_.start_servable(servable_directory, servable_name, version_number, master_address, worker_address)
    _start_py_task(Worker_.get_batch_size())
    _start_wait_and_clear()
