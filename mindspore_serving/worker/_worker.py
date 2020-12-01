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
from mindspore_serving._mindspore_serving import Worker_
from . import context
from .task import start_py_task
from . import check_type

_wait_and_clear_thread = None


# waiting for Ctrl+C, and clear
def _start_wait_and_clear():
    def thread_func():
        print("Serving worker: wait for Ctrl+C to exit ------------------------------------")
        Worker_.wait_and_clear()

    global _wait_and_clear_thread
    if not _wait_and_clear_thread:
        _wait_and_clear_thread = threading.Thread(target=thread_func)
        _wait_and_clear_thread.start()


def stop():
    Worker_.stop()


def stop_on_except(func):
    @wraps(func)
    def handle_except(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except:
            stop()
            raise

    return handle_except


def _load_servable_config(servable_directory, servable_name):
    import sys
    sys.path.append(servable_directory)
    __import__(servable_name + ".servable_config")


# define version_strategy : "lastest" "specific" "multi"
# device_type: 'Ascend910', 'Ascend310'
@stop_on_except
def start_servable(servable_directory, servable_name, version_number=0,
                   device_type=None, device_id=0,
                   master_ip="0.0.0.0", master_port=6100, host_ip="0.0.0.0", host_port=6200):
    r"""
    Start up the servable named 'servable_name' defined in 'servable_directory', and the servable linked to the master
    through gRPC (master_ip, master_port).

    Serving has two running modes. One is running in a single process, providing the Serving service of a single model.
    The other includes a master and multiple workers. The master is responsible for providing the Serving access
    interface for client, the worker is responsible for providing the service of the specific model, and the master
    and worker communicate through gPRC defined as (master_ip, master_port) and (host_ip, host_port).

    Args:
        servable_directory (str): The directory where the servable located in, there expected to has a directory named
                                  `servable_name`. For example:
                                  servables_dir/  # servable_directory
                                    └─ bert/   # servable_name
                                        ├───1/  # version 1
                                        │   └───  bert.mindir
                                        ├───2/  # version 2
                                        │    └───  bert.mindir
                                        └──servable_config.py

        servable_name (str): The servable name.
        version_number (int): Model version number to be loaded. odel version ould be a positive integer.
                             0 means to load the latest version.
                             In other cases, such as 1, it means to load the specified version. Default: 0.
        device_type (str): Current only support "Ascend" and None.
                           "Ascend" which means device type can  Ascend910 or Ascend310, etc.
                           None: The device type is determined by the mindspore environment.
                           Default: None.
        device_id (int): The id of the device the model loads into and runs in.
        master_ip (str): The master ip the worker linked to.
        master_port (int): The master port the worker linked to.
        host_ip (str): The worker ip the master linked to.
        host_port (int): The worker port the master linked to.
    """
    check_type.check_str(servable_directory)
    check_type.check_str(servable_name)
    check_type.check_int(version_number)

    if device_type:
        check_type.check_str(device_type)
    check_type.check_int(device_id)

    check_type.check_str(master_ip)
    check_type.check_int(master_port)

    check_type.check_str(host_ip)
    check_type.check_int(host_port)

    _load_servable_config(servable_directory, servable_name)

    if device_type is not None:
        context.set_context(device_type=device_type)
    else:
        context.set_context(device_type='None')  # depend on register implement

    context.set_context(device_id=device_id)

    Worker_.start_servable(servable_directory, servable_name, version_number, master_ip, master_port,
                           host_ip, host_port)
    start_py_task(Worker_.get_batch_size())
    _start_wait_and_clear()


# define version_strategy : "lastest" "specific" "multi"
@stop_on_except
def start_servable_in_master(servable_directory, servable_name, version_number=0, device_type=None,
                             device_id=0):
    r"""
    Start up the servable named 'servable_name' defined in 'svable_directory', and the servable will running in
    the process of the master.

    Serving has two running modes. One is running in a single process, providing the Serving service of a single model.
    The other includes a master and multiple workers. The master is responsible for providing the Serving access
    interface for client, the worker is responsible for providing the service of the specific model, and the master
    and worker communicate through gPRC.

    Args:
        servable_directory (str): The directory where the servable located in, there expected to has a directory named
                                  `servable_name`. For example:
                                  servables_dir/  # servable_directory
                                    └─ bert/   # servable_name
                                        ├───1/  # version 1
                                        │   └───  bert.mindir
                                        ├───2/  # version 2
                                        │    └───  bert.mindir
                                        └──servable_config.py

        servable_name (str): The servable name.
        version_number (int): Model version number to be loaded. odel version ould be a positive integer.
                             0 means to load the latest version.
                             In other cases, such as 1, it means to load the specified version. Default: 0.
        device_type (str): Current only support "Ascend" and None.
                           "Ascend" which means device type can be Ascend910 or Ascend310, etc.
                           None: The device type is determined by the mindspore environment.
                           Default: None.
        device_id (int): The id of the device the model loads into and runs in.
    """
    check_type.check_str(servable_directory)
    check_type.check_str(servable_name)
    check_type.check_int(version_number)

    if device_type:
        check_type.check_int(device_type)
    check_type.check_int(device_id)

    _load_servable_config(servable_directory, servable_name)

    if device_type is not None:
        context.set_context(device_type=device_type)
    else:
        context.set_context(device_type='None')  # depend on register implement

    context.set_context(device_id=device_id)
    Worker_.start_servable_in_master(servable_directory, servable_name, version_number)
    start_py_task(Worker_.get_batch_size())
    _start_wait_and_clear()
