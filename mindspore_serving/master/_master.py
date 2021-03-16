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
"""method of server supplied for master"""

import threading
from functools import wraps
from mindspore_serving.common import check_type
from mindspore_serving import log as logger
from mindspore_serving._mindspore_serving import ExitSignalHandle_
from mindspore_serving._mindspore_serving import Master_

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


def stop():
    r"""
    Stop the running of master.

    Examples:
        >>> from mindspore_serving import master
        >>>
        >>> master.start_grpc_server("0.0.0.0", 5500)
        >>> master.start_restful_server("0.0.0.0", 1500)
        >>> ...
        >>> master.stop()
    """
    Master_.stop_and_clear()


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
def start_grpc_server(ip="0.0.0.0", grpc_port=5500, max_msg_mb_size=100):
    r"""
    Start gRPC server for the communication between client and serving.

    Args:
        ip (str): gRPC server ip.
        grpc_port (int): gRPC port ip, default 5500, ip port range [1, 65535].
        max_msg_mb_size (int): The maximum acceptable gRPC message size in megabytes(MB), default 100,
            value range [1, 512].
    Raises:
        RuntimeError: Fail to start the gRPC server.

    Examples:
        >>> from mindspore_serving import master
        >>>
        >>> master.start_grpc_server("0.0.0.0", 5500)
        >>> master.start_restful_server("0.0.0.0", 1500)
    """
    check_type.check_str('ip', ip)
    check_type.check_ip_port('grpc_port', grpc_port)
    check_type.check_int('max_msg_mb_size', max_msg_mb_size, 1, 512)

    Master_.start_grpc_server(ip, grpc_port, max_msg_mb_size)
    _start_wait_and_clear()


@stop_on_except
def start_master_server(ip="127.0.0.1", master_port=6100):
    r"""
    Start the gRPC server for the communication between workers and the master.

    Note:
        The ip is expected to be accessed only by workers, not clients.

    Args:
        ip (str): gRPC ip for workers to communicate with, default '127.0.0.1'.
        master_port (int): gRPC port ip, default 6100, ip port range [1, 65535].

    Raises:
        RuntimeError: Fail to start the master server.

    Examples:
        >>> from mindspore_serving import master
        >>>
        >>> master.start_grpc_server("0.0.0.0", 5500)
        >>> master.start_restful_server("0.0.0.0", 1500)
        >>> master.start_master_server("127.0.0.1", 6100)
    """
    check_type.check_str('ip', ip)
    check_type.check_ip_port('master_port', master_port)

    Master_.start_grpc_master_server(ip, master_port)
    _start_wait_and_clear()


@stop_on_except
def start_restful_server(ip="0.0.0.0", restful_port=5900, max_msg_mb_size=100):
    r"""
    Start RESTful server for the communication between client and serving.

    Args:
        ip (str): RESTful server ip.
        restful_port (int): gRPC port ip, default 5900, ip port range [1, 65535].
        max_msg_mb_size (int): The maximum acceptable RESTful message size in megabytes(MB), default 100,
            value range [1, 512].
    Raises:
        RuntimeError: Fail to start the RESTful server.

    Examples:
        >>> from mindspore_serving import master
        >>>
        >>> master.start_restful_server("0.0.0.0", 1500)
    """
    check_type.check_str('ip', ip)
    check_type.check_ip_port('restful_port', restful_port)
    check_type.check_int('max_msg_mb_size', max_msg_mb_size, 1, 512)

    Master_.start_restful_server(ip, restful_port, max_msg_mb_size)
    _start_wait_and_clear()
