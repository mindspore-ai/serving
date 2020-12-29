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
"""Inferface for start up servable"""

import threading
from functools import wraps
from mindspore_serving import log as logger
from mindspore_serving._mindspore_serving import Worker_
from . import context
from .task import _start_py_task
from . import check_type

_wait_and_clear_thread = None


def _start_wait_and_clear():
    """Waiting for Ctrl+C, and clear up environment"""

    def thread_func():
        logger.info("Serving worker: wait for Ctrl+C to exit ------------------------------------")
        print("Serving worker: wait for Ctrl+C to exit ------------------------------------")
        Worker_.wait_and_clear()

    global _wait_and_clear_thread
    if not _wait_and_clear_thread:
        _wait_and_clear_thread = threading.Thread(target=thread_func)
        _wait_and_clear_thread.start()


def stop():
    r"""
    Stop the running of worker.

    Examples:
        >>> import os
        >>> from mindspore_serving import worker
        >>>
        >>> servable_dir = os.path.abspath(".")
        >>> worker.start_servable(servable_dir, "lenet", device_id=0, \
        ...                       master_ip="127.0.0.1", master_port=6500, \
        ...                       host_ip="127.0.0.1", host_port=6600)
        >>> ...
        >>> worker.stop()
    """

    Worker_.stop()


def stop_on_except(func):
    """Wrap of clear environment and exit on Serving exception"""

    @wraps(func)
    def handle_except(*args, **kwargs):
        try:
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
    __import__(servable_name + ".servable_config")


@stop_on_except
def start_servable(servable_directory, servable_name, version_number=0,
                   device_type=None, device_id=0,
                   master_ip="0.0.0.0", master_port=6100, worker_ip="0.0.0.0", worker_port=6200):
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
        master_ip (str): The master ip the worker linked to.
        master_port (int): The master port the worker linked to.
        worker_ip (str): The worker ip the master linked to.
        worker_port (int): The worker port the master linked to.

    Examples:
        >>> import os
        >>> from mindspore_serving import worker
        >>>
        >>> servable_dir = os.path.abspath(".")
        >>> worker.start_servable(servable_dir, "lenet", device_id=0, \
        ...                       master_ip="127.0.0.1", master_port=6500, \
        ...                       host_ip="127.0.0.1", host_port=6600)
    """
    check_type.check_str('servable_directory', servable_directory)
    check_type.check_str('servable_name', servable_name)
    check_type.check_int('version_number', version_number, 0)

    check_type.check_int('device_id', device_id, 0)

    check_type.check_str('master_ip', master_ip)
    check_type.check_ip_port('master_port', master_port)

    check_type.check_str('worker_ip', worker_ip)
    check_type.check_ip_port('worker_port', worker_port)

    _load_servable_config(servable_directory, servable_name)

    if device_type is not None:
        check_type.check_str('device_type', device_type)
        context.set_context(device_type=device_type)
    else:
        context.set_context(device_type='None')  # depend on register implement

    context.set_context(device_id=device_id)

    Worker_.start_servable(servable_directory, servable_name, version_number, master_ip, master_port,
                           worker_ip, worker_port)
    _start_py_task(Worker_.get_batch_size())
    _start_wait_and_clear()


@stop_on_except
def start_servable_in_master(servable_directory, servable_name, version_number=0, device_type=None,
                             device_id=0):
    r"""
    Start up the servable named 'servable_name' defined in 'svable_directory', and the worker will run in
    the process of the master.

    Serving has two running modes. One is running in a single process, providing the Serving service of a single model.
    The other includes a master and multiple workers. This interface is for the first scenario.

    Args:
        servable_directory (str): The directory where the servable is located in. There expects to has a directory named
            `servable_name`. For more detail:
            `How to config Servable <https://www.mindspore.cn/tutorial/inference/zh-CN/master/serving_model.html>`_ .

        servable_name (str): The servable name.
        version_number (int): Servable version number to be loaded. The version number should be a positive integer,
            starting from 1, and 0 means to load the latest version. Default: 0.
        device_type (str): Currently only supports "Ascend", "Davinci" and None, Default: None.
            "Ascend" means the device type can be Ascend910 or Ascend310, etc.
            "Davinci" has the same meaning as "Ascend".
            None means the device type is determined by the MindSpore environment.

    Examples:
        >>> import os
        >>> from mindspore_serving import worker
        >>> from mindspore_serving import master
        >>>
        >>> servable_dir = os.path.abspath(".")
        >>> worker.start_servable_in_master(servable_dir, "lenet", device_id=0)
        >>>
        >>> master.start_grpc_server("0.0.0.0", 5500)
        >>> master.start_restful_server("0.0.0.0", 1500)
    """
    check_type.check_str('servable_directory', servable_directory)
    check_type.check_str('servable_name', servable_name)
    check_type.check_int('version_number', version_number, 0)

    check_type.check_int('device_id', device_id, 0)

    _load_servable_config(servable_directory, servable_name)

    if device_type is not None:
        check_type.check_str('device_type', device_type)
        context.set_context(device_type=device_type)
    else:
        context.set_context(device_type='None')  # depend on register implement

    context.set_context(device_id=device_id)
    Worker_.start_servable_in_master(servable_directory, servable_name, version_number)
    _start_py_task(Worker_.get_batch_size())
    _start_wait_and_clear()
