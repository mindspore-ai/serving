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

import sys
import grpc
import numpy as np
import ms_service_pb2
import ms_service_pb2_grpc


def create_tensor(data):
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
        np.str: ms_service_pb2.MS_STRING,
        np.bytes: ms_service_pb2.MS_BYTES,
    }
    tensor.dtype = dtype_map[data.dtype]
    tensor.data = data.tobytes()
    return tensor


def run_inputs():
    if len(sys.argv) > 2:
        sys.exit("input error")
    channel_str = ""
    if len(sys.argv) == 2:
        split_args = sys.argv[1].split('=')
        if len(split_args) > 1:
            channel_str = split_args[1]
        else:
            channel_str = 'localhost:5500'
    else:
        channel_str = 'localhost:5500'

    channel = grpc.insecure_channel(channel_str)
    stub = ms_service_pb2_grpc.MSServiceStub(channel)
    request = ms_service_pb2.PredictRequest()
    request.servable_spec.name = "add_test"
    request.servable_spec.method_name = "method_test"

    request.inputs["x0"] = create_tensor(np.ones([2, 2]).astype(np.float32))
    request.inputs["x1"] = create_tensor(np.ones([2, 2]).astype(np.float32))
    request.inputs["x2"] = create_tensor(np.ones([2, 2]).astype(np.float32))

    try:
        result = stub.Predict(request)
        print("ms client received: ")
        if not result.error_msg:
            result_np = np.frombuffer(result.result[0].data, dtype=np.float32).reshape(
                result.result[0].tensor_shape.dims)
            print(result_np)
        else:
            print(result.error_msg)
    except grpc.RpcError as e:
        print(e.details())
        status_code = e.code()
        print(status_code.name)
        print(status_code.value)


def run_instances():
    if len(sys.argv) > 2:
        sys.exit("input error")
    if len(sys.argv) == 2:
        split_args = sys.argv[1].split('=')
        if len(split_args) > 1:
            channel_str = split_args[1]
        else:
            channel_str = 'localhost:5500'
    else:
        channel_str = 'localhost:5500'

    channel = grpc.insecure_channel(channel_str)
    stub = ms_service_pb2_grpc.MSServiceStub(channel)
    request = ms_service_pb2.PredictRequest()
    request.servable_spec.name = "add_test"
    request.servable_spec.method_name = "method_test"

    for i in range(5):
        instance = request.instances.add()
        instance.items["x0"] = create_tensor(np.ones([2, ]).astype(np.float32))
        instance.items["x1"] = create_tensor(np.ones([2, ]).astype(np.float32))
        instance.items["x2"] = create_tensor(np.ones([2, ]).astype(np.float32))
    try:
        result = stub.Predict(request)
        print("ms client received: ")
        if not result.error_msg:
            result_np = np.frombuffer(result.result[0].data, dtype=np.float32).reshape(
                result.result[0].tensor_shape.dims)
            print(result_np)
        elif len(result.error_msg) == 1:
            print(result.error_msg[0].error_msg)
        else:
            for i in range(len(result.error_msg)):
                print("result of ", i)
                if result.error_msg[i].error_code != 0:
                    print(result.error_msg[i].error_msg)
                else:
                    print(result.instances[i])

    except grpc.RpcError as e:
        print(e.details())
        status_code = e.code()
        print(status_code.name)
        print(status_code.value)


if __name__ == '__main__':
    run_inputs()
    run_instances()
