# Copyright 2021 Huawei Technologies Co., Ltd
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
"""method of server supplied for master"""

import threading
from functools import wraps

from mindspore_serving._mindspore_serving import ExitSignalHandle_
from mindspore_serving._mindspore_serving import Master_

from mindspore_serving import log as logger
from mindspore_serving.server.common import check_type

_wait_and_clear_thread = None


# waiting for Ctrl+C, and clear
def _start_wait_and_clear():
    """Start thread waiting for catch ctrl+c, and clear env"""

    def thread_func():
        logger.info("Serving master: wait for Ctrl+C to exit ------------------------------------")
        print("Serving master: wait for Ctrl+C to exit ------------------------------------")
        Master_.wait_and_clear()

    global _wait_and_clear_thread
    if not _wait_and_clear_thread:
        _wait_and_clear_thread = threading.Thread(target=thread_func)
        _wait_and_clear_thread.start()


at_stop_list = []


def add_atstop_proc(func):
    """At serving server stop, execute function"""
    global at_stop_list
    at_stop_list.append(func)


def stop():
    r"""
    Stop the running of serving server.

    Examples:
        >>> from mindspore_serving import server
        >>>
        >>> server.start_grpc_server("0.0.0.0:5500")
        >>> server.start_restful_server("0.0.0.0:1500")
        >>> ...
        >>> server.stop()
    """
    Master_.stop_and_clear()
    global at_stop_list
    for func in at_stop_list:
        result = func()
        if result is None or result is True:
            at_stop_list.remove(func)


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


@stop_on_except
def start_grpc_server(address="0.0.0.0:5500", max_msg_mb_size=100):
    r"""
    Start gRPC server for the communication between serving client and server.

    Args:
        address (str): gRPC server address, the address can be {ip}:{port} or unix:{unix_domain_file_path}.
            - {ip}:{port} - Internet domain socket address.
            - unix:{unix_domain_file_path} - Unix domain socket address, which is used to communicate with multiple
                processes on the same machine. {unix_domain_file_path} can be relative or absolute file path,
                but the directory where the file is located must already exist.
        max_msg_mb_size (int): The maximum acceptable RESTful message size in megabytes(MB), default 100,
            value range [1, 512].
    Raises:
        RuntimeError: Failed to start the RESTful server.

    Examples:
        >>> from mindspore_serving import server
        >>>
        >>> server.start_grpc_server("0.0.0.0:1500")
    """
    check_type.check_str('address', address)
    check_type.check_int('max_msg_mb_size', max_msg_mb_size, 1, 512)

    Master_.start_grpc_server(address, max_msg_mb_size)


@stop_on_except
def start_restful_server(address="0.0.0.0:5900", max_msg_mb_size=100):
    r"""
    Start RESTful server for the communication between serving client and server.

    Args:
        address (str): RESTful server address, the address can be {ip}:{port} or unix:{unix_domain_file_path}.
            - {ip}:{port} - Internet domain socket address.
            - unix:{unix_domain_file_path} - Unix domain socket address, which is used to communicate with multiple
                processes on the same machine. {unix_domain_file_path} can be relative or absolute file path,
                but the directory where the file is located must already exist.
        max_msg_mb_size (int): The maximum acceptable RESTful message size in megabytes(MB), default 100,
            value range [1, 512].
    Raises:
        RuntimeError: Failed to start the RESTful server.

    Examples:
        >>> from mindspore_serving import server
        >>>
        >>> server.start_restful_server("0.0.0.0:1500")
    """
    check_type.check_str('address', address)
    check_type.check_int('max_msg_mb_size', max_msg_mb_size, 1, 512)

    Master_.start_restful_server(address, max_msg_mb_size)


def start_master_server(address):
    """Start the gRPC server for the communication between workers and the master of serving server"""
    check_type.check_str('address', address)

    Master_.start_grpc_master_server(address)
