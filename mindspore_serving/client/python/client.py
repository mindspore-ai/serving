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
"""MindSpore Serving Client"""

import grpc
import numpy as np
import mindspore_serving.proto.ms_service_pb2 as ms_service_pb2
import mindspore_serving.proto.ms_service_pb2_grpc as ms_service_pb2_grpc


def _create_tensor(data, tensor=None):
    """Create tensor from numpy data"""
    if tensor is None:
        tensor = ms_service_pb2.Tensor()

    tensor.shape.dims.extend(data.shape)
    dtype_map = {
        np.bool: ms_service_pb2.MS_BOOL,
        np.int8: ms_service_pb2.MS_INT8,
        np.uint8: ms_service_pb2.MS_UINT8,
        np.int16: ms_service_pb2.MS_INT16,
        np.uint16: ms_service_pb2.MS_UINT16,
        np.int32: ms_service_pb2.MS_INT32,
        np.uint32: ms_service_pb2.MS_UINT32,

        np.int64: ms_service_pb2.MS_INT64,
        np.uint64: ms_service_pb2.MS_UINT64,
        np.float16: ms_service_pb2.MS_FLOAT16,
        np.float32: ms_service_pb2.MS_FLOAT32,
        np.float64: ms_service_pb2.MS_FLOAT64,
    }
    for k, v in dtype_map.items():
        if k == data.dtype:
            tensor.dtype = v
            break
    if tensor.dtype == ms_service_pb2.MS_UNKNOWN:
        raise RuntimeError("Unknown data type " + str(data.dtype))
    tensor.data = data.tobytes()
    return tensor


def _create_scalar_tensor(vals, tensor=None):
    """Create tensor from scalar data"""
    if not isinstance(vals, (tuple, list)):
        vals = (vals,)
    return _create_tensor(np.array(vals), tensor)


def _create_bytes_tensor(bytes_vals, tensor=None):
    """Create tensor from bytes data"""
    if tensor is None:
        tensor = ms_service_pb2.Tensor()

    if not isinstance(bytes_vals, (tuple, list)):
        bytes_vals = (bytes_vals,)
    tensor.shape.dims.extend([len(bytes_vals)])
    tensor.dtype = ms_service_pb2.MS_BYTES
    for item in bytes_vals:
        tensor.bytes_val.append(item)
    return tensor


def _create_str_tensor(str_vals, tensor=None):
    """Create tensor from str data"""
    if tensor is None:
        tensor = ms_service_pb2.Tensor()

    if not isinstance(str_vals, (tuple, list)):
        str_vals = (str_vals,)
    tensor.shape.dims.extend([len(str_vals)])
    tensor.dtype = ms_service_pb2.MS_STRING
    for item in str_vals:
        tensor.bytes_val.append(bytes(item, encoding="utf8"))
    return tensor


def _create_numpy_from_tensor(tensor):
    """Create numpy from protobuf tensor"""
    dtype_map = {
        ms_service_pb2.MS_BOOL: np.bool,
        ms_service_pb2.MS_INT8: np.int8,
        ms_service_pb2.MS_UINT8: np.uint8,
        ms_service_pb2.MS_INT16: np.int16,
        ms_service_pb2.MS_UINT16: np.uint16,
        ms_service_pb2.MS_INT32: np.int32,
        ms_service_pb2.MS_UINT32: np.uint32,

        ms_service_pb2.MS_INT64: np.int64,
        ms_service_pb2.MS_UINT64: np.uint64,
        ms_service_pb2.MS_FLOAT16: np.float16,
        ms_service_pb2.MS_FLOAT32: np.float32,
        ms_service_pb2.MS_FLOAT64: np.float64,
    }
    if tensor.dtype == ms_service_pb2.MS_STRING or tensor.dtype == ms_service_pb2.MS_BYTES:
        result = []
        for item in tensor.bytes_val:
            if tensor.dtype == ms_service_pb2.MS_STRING:
                result.append(bytes.decode(item))
            else:
                result.append(item)
        if len(result) == 1:
            return result[0]
        return result

    result = np.frombuffer(tensor.data, dtype_map[tensor.dtype]).reshape(tensor.shape.dims)
    return result


def _check_str(arg_name, str_val):
    """Check whether the input parameters are reasonable str input"""
    if not isinstance(str_val, str):
        raise RuntimeError(f"Parameter '{arg_name}' should be str, but actually {type(str_val)}")
    if not str_val:
        raise RuntimeError(f"Parameter '{arg_name}' should not be empty str")


def _check_int(arg_name, int_val, minimum=None, maximum=None):
    """Check whether the input parameters are reasonable int input"""
    if not isinstance(int_val, int):
        raise RuntimeError(f"Parameter '{arg_name}' should be int, but actually {type(int_val)}")
    if minimum is not None and int_val < minimum:
        if maximum is not None:
            raise RuntimeError(f"Parameter '{arg_name}' should be in range [{minimum},{maximum}]")
        raise RuntimeError(f"Parameter '{arg_name}' should be >= {minimum}")
    if maximum is not None and int_val > maximum:
        if minimum is not None:
            raise RuntimeError(f"Parameter '{arg_name}' should be in range [{minimum},{maximum}]")
        raise RuntimeError(f"Parameter '{arg_name}' should be <= {maximum}")


class SSLConfig:
    """
    The client's ssl_config encapsulates grpc's ssl channel credentials for SSL-enabled connections.

    Args:
        certificate (str, optional): File holding the PEM-encoded certificate chain as a byte string to use or None if
            no certificate chain should be used. Default: None.
        private_key (str, optional): File holding the PEM-encoded private key as a byte string, or None if no private
            key should be used. Default: None.
        custom_ca (str, optional): File holding the PEM-encoded root certificates as a byte string, or None to retrieve
            them from a default location chosen by gRPC runtime. Default: None.

    Raises:
        RuntimeError: The type or value of the parameters is invalid.

    """

    def __init__(self, certificate=None, private_key=None, custom_ca=None):
        if certificate is not None:
            _check_str("certificate", certificate)
        if private_key is not None:
            _check_str("private_key", private_key)
        if custom_ca is not None:
            _check_str("custom_ca", custom_ca)

        self.certificate = certificate
        self.private_key = private_key
        self.custom_ca = custom_ca


class Client:
    """
    The Client encapsulates the serving gRPC API, which can be used to create requests,
    access serving, and parse results.

    Note:
        The maximum amount of data that the client can send in one request is 512MB, and the maximum amount of data that
        the server can accept can be configured as 1~512MB, 100MB by default.

    Args:
        address (str): Serving address.
        servable_name (str): The name of servable supplied by Serving.
        method_name (str): The name of method supplied by servable.
        version_number (int, optional): The version number of servable, 0 means the maximum version number in all
            running versions. Default: 0.
        ssl_config (mindspore_serving.client.SSLConfig, optional): The server's ssl_config, if None, disabled ssl.
            Default: None.

    Raises:
        RuntimeError: The type or value of the parameters are invalid, or other errors happened.

    Examples:
        >>> from mindspore_serving.client import Client
        >>> import numpy as np
        >>> client = Client("localhost:5500", "add", "add_cast")
        >>> instances = []
        >>> x1 = np.ones((2, 2), np.int32)
        >>> x2 = np.ones((2, 2), np.int32)
        >>> instances.append({"x1": x1, "x2": x2})
        >>> result = client.infer(instances)
        >>> print(result)
    """

    def __init__(self, address, servable_name, method_name, version_number=0, ssl_config=None):
        _check_str("address", address)
        _check_str("servable_name", servable_name)
        _check_str("method_name", method_name)
        _check_int("version_number", version_number, 0)

        self.address = address
        self.servable_name = servable_name
        self.method_name = method_name
        self.version_number = version_number

        msg_bytes_size = 512 * 1024 * 1024  # 512MB
        options = [
            ('grpc.max_send_message_length', msg_bytes_size),
            ('grpc.max_receive_message_length', msg_bytes_size),
        ]
        if ssl_config is not None:
            if not isinstance(ssl_config, SSLConfig):
                raise RuntimeError("The type of ssl_config should be type of SSLConfig")
            rc_bytes = pk_bytes = c_bytes = None
            if ssl_config.certificate is not None:
                with open(ssl_config.certificate, 'rb') as c_fs:
                    c_bytes = c_fs.read()
            if ssl_config.private_key is not None:
                with open(ssl_config.private_key, 'rb') as pk_fs:
                    pk_bytes = pk_fs.read()
            if ssl_config.custom_ca is not None:
                with open(ssl_config.custom_ca, 'rb') as rc_fs:
                    rc_bytes = rc_fs.read()
            if (c_bytes is None and pk_bytes is not None) or (c_bytes is not None and pk_bytes is None):
                raise RuntimeError("The certificate and private_key should be passed at the same time")
            creds = grpc.ssl_channel_credentials(root_certificates=rc_bytes,
                                                 private_key=pk_bytes,
                                                 certificate_chain=c_bytes)
            self.channel = grpc.secure_channel(address, creds, options=options)
        else:
            self.channel = grpc.insecure_channel(address, options=options)

        self.stub = ms_service_pb2_grpc.MSServiceStub(self.channel)

    def infer(self, instances):
        """
        Used to create requests, access serving service, and parse and return results.

        Args:
            instances (Union[dict, tuple[dict]]): Instance or tuple of instances,
                every instance item is the inputs dict. The key is the input name,
                and the value is the input value, the type of value can be python int,
                float, bool, str, bytes, numpy number, or numpy array object.

        Raises:
            RuntimeError: The type or value of the parameters is invalid, or other errors happened.

        Examples:
            >>> from mindspore_serving.client import Client
            >>> import numpy as np
            >>> client = Client("localhost:5500", "add", "add_cast")
            >>> instances = []
            >>> x1 = np.ones((2, 2), np.int32)
            >>> x2 = np.ones((2, 2), np.int32)
            >>> instances.append({"x1": x1, "x2": x2})
            >>> result = client.infer(instances)
            >>> print(result)
        """
        request = self._create_request(instances)
        try:
            result = self.stub.Predict(request)
            return self._paser_result(result)

        except grpc.RpcError as e:
            print(e.details())
            status_code = e.code()
            print(status_code.name)
            print(status_code.value)
            return {"error": f"Grpc Error, {status_code.value}, {e.details()}"}

    def infer_async(self, instances):
        """
        Used to create requests, async access serving.

        Args:
            instances (Union[dict, tuple[dict]]): Instance or tuple of instances, every instance item
                is the inputs dict. The key is the input name, and the value is the input value, the
                type of value can be python int, float, bool, str, bytes, numpy number,
                or numpy array object.

        Raises:
            RuntimeError: The type or value of the parameters is invalid, or other errors happened.

        Examples:
            >>> from mindspore_serving.client import Client
            >>> import numpy as np
            >>> client = Client("localhost:5500", "add", "add_cast")
            >>> instances = []
            >>> x1 = np.ones((2, 2), np.int32)
            >>> x2 = np.ones((2, 2), np.int32)
            >>> instances.append({"x1": x1, "x2": x2})
            >>> result_future = client.infer_async(instances)
            >>> result = result_future.result()
            >>> print(result)
        """
        request = self._create_request(instances)
        try:
            result_future = self.stub.Predict.future(request)
            return ClientGrpcAsyncResult(result_future)

        except grpc.RpcError as e:
            print(e.details())
            status_code = e.code()
            print(status_code.name)
            print(status_code.value)
            return ClientGrpcAsyncError({"error": f"Grpc Error, {status_code.value}, {e.details()}"})

    def _create_request(self, instances):
        """Used to create request spec."""
        if not isinstance(instances, (tuple, list)):
            instances = (instances,)

        request = ms_service_pb2.PredictRequest()
        request.servable_spec.name = self.servable_name
        request.servable_spec.method_name = self.method_name
        request.servable_spec.version_number = self.version_number

        for item in instances:
            if isinstance(item, dict):
                request.instances.append(self._create_instance(**item))
            else:
                raise RuntimeError("instance should be a map")
        return request

    @staticmethod
    def _create_instance(**kwargs):
        """Used to create gRPC instance."""
        instance = ms_service_pb2.Instance()
        for k, w in kwargs.items():
            tensor = instance.items[k]
            if isinstance(w, (np.ndarray, np.number)):
                _create_tensor(w, tensor)
            elif isinstance(w, str):
                _create_str_tensor(w, tensor)
            elif isinstance(w, (bool, int, float)):
                _create_scalar_tensor(w, tensor)
            elif isinstance(w, bytes):
                _create_bytes_tensor(w, tensor)
            else:
                raise RuntimeError("Not support value type " + str(type(w)))
        return instance

    @staticmethod
    def _paser_result(result):
        """Used to parse result."""
        error_msg_len = len(result.error_msg)
        if error_msg_len == 1 and result.error_msg[0].error_code != 0:
            return {"error": bytes.decode(result.error_msg[0].error_msg)}
        ret_val = []
        instance_len = len(result.instances)
        if error_msg_len not in (0, instance_len):
            raise RuntimeError(f"error msg result size {error_msg_len} not be 0, 1 or "
                               f"length of instances {instance_len}")
        for i in range(instance_len):
            instance = result.instances[i]
            if error_msg_len == 0 or result.error_msg[i].error_code == 0:
                instance_map = {}
                for k, w in instance.items.items():
                    instance_map[k] = _create_numpy_from_tensor(w)
                ret_val.append(instance_map)
            else:
                ret_val.append({"error": bytes.decode(result.error_msg[i].error_msg)})
        return ret_val


class ClientGrpcAsyncResult:
    """
    When Client.infer_async invoke successfully, a ClientGrpcAsyncResult object is returned.

    Examples:
        >>> from mindspore_serving.client import Client
        >>> import numpy as np
        >>> client = Client("localhost:5500", "add", "add_cast")
        >>> instances = []
        >>> x1 = np.ones((2, 2), np.int32)
        >>> x2 = np.ones((2, 2), np.int32)
        >>> instances.append({"x1": x1, "x2": x2})
        >>> result_future = client.infer_async(instances)
        >>> result = result_future.result()
        >>> print(result)
    """

    def __init__(self, result_future):
        self.result_future = result_future

    def result(self):
        """Wait and get result of inference result, the gRPC message will be parse to tuple of instances result.
        Every instance result is dict, and value could be numpy array/number, str or bytes according gRPC Tensor
        data type.
        """
        result = self.result_future.result()
        # pylint: disable=protected-access
        result = Client._paser_result(result)
        return result


class ClientGrpcAsyncError:
    """When gRPC failed happened when calling Client.infer_async, a ClientGrpcAsyncError object is returned.
    """

    def __init__(self, result_error):
        self.result_error = result_error

    def result(self):
        """Get gRPC error message.
        """
        return self.result_error
