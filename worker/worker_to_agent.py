import numpy as np

import proto.work_agent_pb2 as agent_pb2
import shared_memory

dtype_map = {
    np.bool: agent_pb2.MS_BOOL,
    np.int8: agent_pb2.MS_INT8,
    np.uint8: agent_pb2.MS_UINT8,
    np.int16: agent_pb2.MS_INT16,
    np.uint16: agent_pb2.MS_UINT16,
    np.int32: agent_pb2.MS_INT32,
    np.uint32: agent_pb2.MS_UINT32,

    np.int64: agent_pb2.MS_INT64,
    np.uint64: agent_pb2.MS_UINT64,
    np.float16: agent_pb2.MS_FLOAT16,
    np.float32: agent_pb2.MS_FLOAT32,
    np.float64: agent_pb2.MS_FLOAT64,
}

dtype_map_rev = {
    agent_pb2.MS_BOOL: np.bool,
    agent_pb2.MS_INT8: np.int8,
    agent_pb2.MS_UINT8: np.uint8,
    agent_pb2.MS_INT16: np.int16,
    agent_pb2.MS_UINT16: np.uint16,
    agent_pb2.MS_INT32: np.int32,
    agent_pb2.MS_UINT32: np.uint32,

    agent_pb2.MS_INT64: np.int64,
    agent_pb2.MS_UINT64: np.uint64,
    agent_pb2.MS_FLOAT16: np.float16,
    agent_pb2.MS_FLOAT32: np.float32,
    agent_pb2.MS_FLOAT64: np.float64,
}


class AgentReply:
    def __init__(self):
        self.error_code = 0  # 0 for success
        self.tensors = []  # list[mslite.Tensor]


def create_proto_tensor(lite_tensor):
    """暂时只支持np.array和np.number"""
    tensor = agent_pb2.Tensor()
    if not isinstance(lite_tensor, (np.ndarray, np.number)):
        raise RuntimeError("unsupported tensor data type" + str(type(lite_tensor)))
    tensor.shape.dims.extend(lite_tensor.shape)
    for k, v in dtype_map.items():
        if k == lite_tensor.dtype:
            tensor.dtype = v
            break
    tensor.data = lite_tensor.tobytes()
    return tensor


def create_proto_request(lite_inputs, is_first_iteration):
    """
    used to create gRPC request -- worker send to agents
    inputs:
        lite_inputs: list(np.array)
        is_first_iteration: bool, for choosing prefill or decode model
    """
    request = agent_pb2.AgentRequest()
    for lite in lite_inputs:
        request.inputs.append(create_proto_tensor(lite))
    request.if_prefill = is_first_iteration
    # TODO 这个subgraph是否会用到
    request.subgraph = 0
    return request


def create_shm_tensor(name, shape, type):
    tensor = agent_pb2.Tensor()
    tensor.shm_data.memory_key = name
    tensor.shape.dims.extend(shape)
    for k, v in dtype_map.items():
        if k == type:
            tensor.dtype = v
            break
    return tensor


def parse_proto_request(request):
    """
    解析agent从worker收到的请求
    input:
        request: work_agent.proto:AgentRequest.
    output:
        result: list[np.array]
        if_inc: bool
    """
    result = []
    for item in request.inputs:
        tensor = np.frombuffer(item.data, dtype_map_rev[item.dtype]).reshape(item.shape.dims)
        result.append(tensor)
    prefill_flag = request.if_prefill
    return result, prefill_flag


def parse_proto_request_new(request):
    """
    解析agent从worker收到的请求
    input:
        request: work_agent.proto:AgentRequest.
    output:
        result: list[np.array]
        if_inc: bool
    """
    result = []
    # 第一个input_ids为shm
    print("inputs from serving {}".format(request.inputs))
    shm_tensor = request.inputs[0]
    existing_shm = shared_memory.SharedMemory(name=shm_tensor.shm_data.memory_key)
    input_ids = np.ndarray((shm_tensor.shape.dims), dtype=dtype_map_rev[shm_tensor.dtype], buffer=existing_shm.buf)
    print("inputs from shm is {}".format(input_ids))
    result.append(input_ids)
    for item in request.inputs[1:]:
        tensor = np.frombuffer(item.data, dtype_map_rev[item.dtype]).reshape(item.shape.dims)
        print("other tensor {}".format(tensor))
        result.append(tensor)
    prefill_flag = request.if_prefill
    print("final inputs {}".format(result))
    return result, prefill_flag


def parse_proto_reply(reply):
    result = AgentReply()
    result.error_code = reply.error_msg.error_code

    for out in reply.outputs:
        tensor = np.frombuffer(out.data, dtype_map_rev[out.dtype]).reshape(out.shape.dims)
        result.tensors.append(tensor)

    return result.tensors[0]

def parse_proto_reply_new(reply):
    result = AgentReply()
    result.error_code = reply.error_msg.error_code
    # 返回shm tensor，在获取所有卡结果后，再进行解析
    shm_tensor = reply.outputs[0]
    return shm_tensor


def create_proto_reply(outputs, error_code=0):
    """
    construct a proto reply from agent to worker
    inputs:
        outputs: list[tensor]
        error_code: int64, 0 for success, 1 for failed
    """
    reply = agent_pb2.AgentReply()
    reply.error_msg.error_code = error_code
    for out in outputs:
        reply.outputs.append(create_proto_tensor(out))
    return reply


def create_proto_request_new(input_shape, input_type, name, other_inputs, is_prefill, mask=None, freq_cos=None, freq_sin=None):
    request = agent_pb2.AgentRequest()
    request.inputs.append(create_shm_tensor(name, input_shape, input_type))
    for lite in other_inputs:
        request.inputs.append(create_proto_tensor(lite))
    request.if_prefill = is_prefill
    request.subgraph = 0
    return request


def create_proto_reply_new(output_shape, output_type, name, error_code=0):
    """
    construct a proto reply from agent to worker
    inputs:
        outputs: list[tensor]
        error_code: int64, 0 for success, 1 for failed
    """
    reply = agent_pb2.AgentReply()
    reply.error_msg.error_code = error_code
    reply.outputs.append(create_shm_tensor(name, output_shape, output_type))
    print("reply is {}".format(reply.outputs))
    return reply
