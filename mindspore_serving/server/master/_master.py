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

from functools import wraps

from mindspore_serving._mindspore_serving import ExitSignalHandle_
from mindspore_serving._mindspore_serving import Master_
from mindspore_serving._mindspore_serving import SSLConfig_

from mindspore_serving.server.common import check_type

_wait_and_clear_thread = None

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


class SSLConfig:
    r"""
    The server's ssl_config encapsulates necessary parameters for SSL-enabled connections.

    Args:
        certificate (str): File holding the PEM-encoded certificate chain as a byte string to use or None if no
            certificate chain should be used.
        private_key (str): File holding the PEM-encoded private key as a byte string, or None if no private key should
            be used.
        custom_ca (str, optional): File holding the PEM-encoded root certificates as a byte string. When verify_client
            is True, custom_ca must be provided. When verify_client is False, this parameter will be ignored.
            Default: None.
        verify_client (bool, optional): If verify_client is true, use mutual authentication. If false, use one-way
            authentication. Default: False.

    Raises:
        RuntimeError: The type or value of the parameters are invalid.
    """

    def __init__(self, certificate, private_key, custom_ca=None, verify_client=False):
        check_type.check_str("certificate", certificate)
        check_type.check_str("private_key", private_key)
        check_type.check_bool("verify_client", verify_client)

        self.custom_ca = custom_ca
        self.certificate = certificate
        self.private_key = private_key
        self.verify_client = verify_client
        if self.verify_client:
            check_type.check_str("custom_ca", custom_ca)


@stop_on_except
def start_grpc_server(address, max_msg_mb_size=100, ssl_config=None):
    r"""
    Start gRPC server for the communication between serving client and server.

    Args:
        address (str): gRPC server address, the address can be `{ip}:{port}` or `unix:{unix_domain_file_path}`.

            - `{ip}:{port}` - Internet domain socket address.
            - `unix:{unix_domain_file_path}` - Unix domain socket address, which is used to communicate with multiple
              processes on the same machine. `{unix_domain_file_path}` can be relative or absolute file path,
              but the directory where the file is located must already exist.

        max_msg_mb_size (int, optional): The maximum acceptable gRPC message size in megabytes(MB), value range
            [1, 512]. Default: 100.
        ssl_config (mindspore_serving.server.SSLConfig, optional): The server's ssl_config, if None, disabled ssl.
            Default: None.

    Raises:
        RuntimeError: Failed to start the gRPC server: parameter verification failed, the gRPC address is wrong or
            the port is duplicate.

    Examples:
        >>> from mindspore_serving import server
        >>>
        >>> server.start_grpc_server("0.0.0.0:5500")
    """
    check_type.check_str('address', address)
    check_type.check_int('max_msg_mb_size', max_msg_mb_size, 1, 512)

    config = SSLConfig_()
    if ssl_config is not None:
        if not isinstance(ssl_config, SSLConfig):
            raise RuntimeError("The type of ssl_config should be type of SSLConfig")
        with open(ssl_config.certificate, 'rb') as c_fs:
            c_bytes = c_fs.read()
        with open(ssl_config.private_key, 'rb') as pk_fs:
            pk_bytes = pk_fs.read()
        if ssl_config.verify_client:
            with open(ssl_config.custom_ca, 'rb') as rc_fs:
                rc_bytes = rc_fs.read()
            config.custom_ca = rc_bytes
        config.certificate = c_bytes
        config.private_key = pk_bytes
        config.verify_client = ssl_config.verify_client
        config.use_ssl = True
    Master_.start_grpc_server(address, config, max_msg_mb_size)


@stop_on_except
def start_restful_server(address, max_msg_mb_size=100, ssl_config=None):
    r"""
    Start RESTful server for the communication between serving client and server.

    Args:
        address (str): RESTful server address, the address should be Internet domain socket address.
        max_msg_mb_size (int, optional): The maximum acceptable RESTful message size in megabytes(MB), value range
            [1, 512]. Default: 100.
        ssl_config (mindspore_serving.server.SSLConfig, optional): The server's ssl_config, if None, disabled ssl.
            Default: None.

    Raises:
        RuntimeError: Failed to start the RESTful server: parameter verification failed, the RESTful address is wrong
            or the port is duplicate.

    Examples:
        >>> from mindspore_serving import server
        >>>
        >>> server.start_restful_server("0.0.0.0:5900")
    """
    check_type.check_str('address', address)
    check_type.check_int('max_msg_mb_size', max_msg_mb_size, 1, 512)

    config = SSLConfig_()
    if ssl_config is not None:
        if not isinstance(ssl_config, SSLConfig):
            raise RuntimeError("The type of ssl_config should be class of SSLConfig")
        if ssl_config.verify_client:
            config.custom_ca = ssl_config.custom_ca
        config.certificate = ssl_config.certificate
        config.private_key = ssl_config.private_key
        config.verify_client = ssl_config.verify_client
        config.use_ssl = True
    Master_.start_restful_server(address, config, max_msg_mb_size)


def start_master_server(address):
    """Start the gRPC server for the communication between workers and the master of serving server"""
    check_type.check_str('address', address)

    Master_.start_grpc_master_server(address)


def only_model_stage(servable_name):
    """Whether only the model stages exist"""
    return Master_.only_model_stage(servable_name)
