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

from mindspore_serving.server.common import check_type
from mindspore_serving.server.common.decorator import deprecated
from mindspore_serving import server


@deprecated("1.3.0", "mindspore_serving.server.start_grpc_server")
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

    server.start_grpc_server(address=f"{ip}:{grpc_port}", max_msg_mb_size=max_msg_mb_size)


@deprecated("1.3.0", "mindspore_serving.server.start_restful_server")
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

    server.start_restful_server(address=f"{ip}:{restful_port}", max_msg_mb_size=max_msg_mb_size)


@deprecated("1.3.0", "mindspore_serving.server.stop")
def stop():
    r"""
    Stop the running of master.

    Examples:
        >>> from mindspore_serving import master
        >>>
        >>> master.start_grpc_server("0.0.0.0:5500")
        >>> master.start_restful_server("0.0.0.0:1500")
        >>> ...
        >>> master.stop()
    """
    server.stop()
