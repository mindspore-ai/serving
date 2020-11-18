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
from mindspore_serving._mindspore_serving import Master_

_wait_and_clear_thread = None


# waiting for Ctrl+C, and clear
def _start_wait_and_clear():
    def thread_func():
        print("Serving master: wait for Ctrl+C to exit ------------------------------------")
        Master_.wait_and_clear()

    global _wait_and_clear_thread
    if not _wait_and_clear_thread:
        _wait_and_clear_thread = threading.Thread(target=thread_func)
        _wait_and_clear_thread.start()


def stop():
    Master_.stop()


def stop_on_except(func):
    @wraps(func)
    def handle_except(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except:
            stop()
            raise

    return handle_except


@stop_on_except
def start_grpc_server(ip="0.0.0.0", grpc_port=5500, max_msg_mb_size=100):
    """start grpc server for the communication between client and serving.
    the ip should be accessible to the client."""
    Master_.start_grpc_server(ip, grpc_port, max_msg_mb_size)
    _start_wait_and_clear()


@stop_on_except
def start_master_server(ip="0.0.0.0", grpc_port=6100):
    """start grpc server for the communication between workers and the master.
    the ip is expected to be accessed only by workers."""
    Master_.start_grpc_master_server(ip, grpc_port)


@stop_on_except
def start_restful_server(ip="0.0.0.0", restful_port=5900, max_msg_mb_size=100):
    """start restful server for the communication between client and serving.
    the ip should be accessible to the client."""
    Master_.start_restful_server(ip, restful_port, max_msg_mb_size)
    _start_wait_and_clear()
