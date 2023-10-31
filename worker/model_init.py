import time
from abc import ABC
from abc import abstractmethod
from mindspore_lite import Model
import grpc
import logging
import numpy as np
from config.serving_config import Baseconfig
from typing import List
import socket

import proto.work_agent_pb2_grpc as work_agent_pb2_grpc
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from worker.worker_to_agent import *
import shared_memory
import asyncio

dtype_to_int = {
    np.bool: 0,
    np.int8: 1,
    np.uint8: 2,
    np.int16: 3,
    np.uint16: 4,
    np.int32: 5,
    np.uint32: 6,

    np.int64: 7,
    np.uint64: 8,
    np.float16: 9,
    np.float32: 10,
    np.float64: 11,
}

int_to_dtype = {
    0: np.bool,
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,

    7: np.int64,
    8: np.uint64,
    9: np.float16,
    10: np.float32,
    11: np.float64,
}


def send_request_to_agent(item, agent_request):
    logging.info("start test run")
    send_req_to_agent_time = time.time()
    proto_reply = item.Predict(agent_request)
    logging.info("get proto_reply time is {} ".format((time.time() - send_req_to_agent_time) * 1000))
    outputs = parse_proto_reply_new(proto_reply)
    logging.info("parse proto reply time is {} ".format((time.time() - send_req_to_agent_time) * 1000))
    return outputs


class BaseInputsOfInfer:
    """
    BaseInputsOfInfer interface.
    """

    def get_inputs(self, model: Model, **kwargs):
        pass

    @staticmethod
    def get_lite_tensor_list(inputs, model):
        input_list = []
        for item in inputs:
            if item is None:
                continue
            input_list.append(item)
        lite_inputs = model.get_inputs()
        for input_np, tensor in zip(input_list, lite_inputs):
            tensor.set_data_from_numpy(input_np)
        return lite_inputs


class CommonInputsOfInfer(BaseInputsOfInfer):
    """
    common infer inputs of llm models.
    """

    def __init__(self):
        pass

    # pylint: disable=W0221
    def get_inputs(self, input_ids=None, current_index=None, valid_length=None,
                   init_reset=None, is_first_iteration=True, **kwargs):
        if not is_first_iteration:
            inputs_tmp = []
            for i in range(len(current_index)):
                current_index_tmp = int(current_index[i]) - i * input_ids.shape[1]  # multibatch
                # use numpy to slice array to avoid complie ascend slice op
                inputs_tmp.append(input_ids[i][current_index_tmp:current_index_tmp + 1])
            input_ids = np.array(inputs_tmp, dtype=np.int32)
        inputs = [input_ids, current_index, init_reset, valid_length]
        return inputs


class CommonInputsOfInferDyn(BaseInputsOfInfer):
    """
    common infer inputs of llm models.
    """

    def __init__(self):
        pass

    # pylint: disable=W0221
    def get_inputs(self, input_ids=None, current_index=None, valid_length=None,
                   init_reset=None, is_first_iteration=True, mask=None, freq_cos=None, freq_sin=None, **kwargs):
        if not is_first_iteration:
            inputs_tmp = []
            for i in range(len(current_index)):
                current_index_tmp = int(current_index[i]) - i * input_ids.shape[1]  # multibatch
                # use numpy to slice array to avoid complie ascend slice op
                print('inptus_tmp: {}'.format(input_ids[i][current_index_tmp:current_index_tmp + 1]))
                inputs_tmp.append(input_ids[i][current_index_tmp:current_index_tmp + 1])
            input_ids = np.array(inputs_tmp, dtype=np.int32)
        # print("inputs_ids:{}".format(input_ids))
        if is_first_iteration:
            # mask, freq_cos, fre_sin
            inputs = [input_ids, current_index, init_reset, valid_length, mask, freq_cos, freq_sin]
        else:
            inputs = [input_ids, current_index, init_reset, valid_length]
        return inputs


class InputOfInfer:
    """
    Input of llm model.
    """
    MAPPING = {
        "bloom": CommonInputsOfInfer,
        "llama": CommonInputsOfInfer,
        "glm2": CommonInputsOfInfer,
        "common": CommonInputsOfInfer,
        "llama_dyn": CommonInputsOfInferDyn
    }

    @classmethod
    def get_inputs(cls, model_name: str, **kwargs):
        """
        Get input tensor list of mslite.

        Args:
            model_name: str, model name.

        Returns:
            tensor list of mslite.
        """
        name = ""
        if model_name not in InputOfInfer.MAPPING:
            for k in InputOfInfer.MAPPING:
                if model_name.startswith(k):
                    name = k
                    break
            if not name:
                logging.warning("Model name not in support maps.Common input format will be used to do inference.")
                name = "common"
        else:
            name = model_name
        return InputOfInfer.MAPPING[name]().get_inputs(**kwargs)


class Singleton(object):
    def __init__(self, cls):
        self._cls = cls
        self.uniqueInstance = None

    def __call__(self):
        if self.uniqueInstance is None:
            self.uniqueInstance = self._cls()
        return self.uniqueInstance


"""
全局定义一个DisModel, 保存和agents的通信管道
"""


@Singleton
class DisModel:
    def __init__(self):
        self.agent_stubs = []
        self.model_name = None
        self.pool = None

    def init(self, agent_ports, agent_ip, model_name, shm_names: List[str] = None):
        # 初始化agents grpc stub
        msg_bytes_size = 1024 * 1024 * 1024  # 512MB
        options = [
            ('grpc.max_send_message_length', msg_bytes_size),
            ('grpc.max_receive_message_length', msg_bytes_size),
            ('grpc.enable_http_proxy', 0)
        ]
        for port in agent_ports:
            print("port ip is {}".format(port))

            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect((agent_ip, port))
            send_str = '#' + ",".join(str(element) for element in shm_names)
            client.sendall(send_str.encode())
            self.agent_stubs.append(client)

            # send shm_names
        self.model_name = model_name
        self.pool = ThreadPoolExecutor(max_workers=len(agent_ports), thread_name_prefix='worker_to_agent_pool')

    def get_predict_inputs(self, input_ids, current_index=None,
                           valid_length=None, init_reset=None, is_first_iteration=True, mask=None, freq_cos=None,
                           freq_sin=None, **kwargs):
        """Get inputs of llm model for mslite."""
        return InputOfInfer.get_inputs(self.model_name, input_ids=input_ids, current_index=current_index,
                                       valid_length=valid_length, init_reset=init_reset,
                                       is_first_iteration=is_first_iteration, mask=mask, freq_cos=freq_cos,
                                       freq_sin=freq_sin, **kwargs)

    def get_model_inputs(self, input_ids, current_index=None,
                         valid_length=None, init_reset=None, is_first_iteration=True, mask=None, freq_cos=None,
                         freq_sin=None, **kwargs) -> np.array:
        if is_first_iteration:
            init_reset = np.array([False])
            lite_inputs = self.get_predict_inputs(input_ids, current_index,
                                                  valid_length, init_reset, is_first_iteration, mask=mask,
                                                  freq_cos=freq_cos, freq_sin=freq_sin, **kwargs)

        else:
            init_reset = np.array([True])
            lite_inputs = self.get_predict_inputs(input_ids, current_index,
                                                  valid_length, init_reset, is_first_iteration, **kwargs)

        return lite_inputs

    def call(self, shm, input_ids, current_index, valid_length, init_reset, is_first_iteration, mask=None,
             freq_cos=None, freq_sin=None, **kwargs):
        """kvcache infer"""
        time_start = time.time()
        logging.info("length of input ids {}".format(len(input_ids)))
        logging.info("is prefill {}".format(is_first_iteration))
        lite_inputs = self.get_model_inputs(input_ids, current_index,
                                            valid_length, init_reset, is_first_iteration, mask=mask, freq_cos=freq_cos,
                                            freq_sin=freq_sin, **kwargs)

        result = []
        first_input = lite_inputs[0]
        tmp = np.ndarray(first_input.shape, dtype=first_input.dtype, buffer=shm.buf)
        tmp[:] = first_input[:]

        other_inputs = lite_inputs[1:]
        agent_request = create_proto_request_new(tmp.shape, tmp.dtype, shm.name, other_inputs, is_first_iteration)
        logging.info("get_model_inputs time is {} ".format((time.time() - time_start) * 1000))
        thread_time = time.time()
        with ThreadPoolExecutor(max_workers=len(self.agent_stubs)) as t:
            tasks = []

            for item in self.agent_stubs:
                tasks.append(t.submit(lambda w: send_request_to_agent(*w), (item, agent_request)))
            wait(tasks)
            for x in tasks:
                result.append(x.result())
        logging.info("thread_time is {} ".format((time.time() - thread_time) * 1000))
        time_sharemem = time.time()
        shm_tensor = result[0]

        existing_shm = shared_memory.SharedMemory(name=shm_tensor.shm_data.memory_key)
        outputs = np.ndarray((shm_tensor.shape.dims), dtype=dtype_map_rev[shm_tensor.dtype], buffer=existing_shm.buf)
        probs = outputs[0]
        prag_s = outputs[1].astype(np.int32)
        logging.info("share_mem is {} ".format((time.time() - time_sharemem) * 1000))
        return probs, prag_s, existing_shm

    def get_gen_parms_np(self, batch_size, dtype=np.float16, **kwargs):
        do_sample_list = kwargs.pop("do_sample_list")
        top_k_list = kwargs.pop("top_k_list")
        top_p_list = kwargs.pop("top_p_list"),
        temperature_list = kwargs.pop("temperature_list"),
        repetition_penalty = kwargs.pop("repetition_penalty"),

        do_sample_np = np.array(do_sample_list).reshape(batch_size, 1).astype(dtype)
        top_p_np = np.array(top_p_list).reshape(batch_size, 1).astype(dtype)
        top_k_np = np.array(top_k_list).reshape(batch_size, 1).astype(dtype)
        temperature_np = np.array(temperature_list).reshape(batch_size, 1).astype(dtype)
        repetition_np = np.array(repetition_penalty).reshape(batch_size, 1).astype(dtype)
        parms_np = np.concatenate((do_sample_np, top_p_np, top_k_np, temperature_np, repetition_np), axis=-1)

        return parms_np

    def callV3(self, shms: List, input_ids, current_index, valid_length, init_reset, is_first_iteration, mask=None,
               freq_cos=None, freq_sin=None, **kwargs):
        """kvcache infer"""
        time_start = time.time()
        logging.info("length of input ids {}".format(len(input_ids)))
        logging.info("is prefill {}".format(is_first_iteration))
        shapes_str = ''
        if is_first_iteration:
            lite_inputs = self.get_model_inputs(input_ids, current_index,
                                                valid_length, init_reset, is_first_iteration, mask=mask,
                                                freq_cos=freq_cos,
                                                freq_sin=freq_sin, **kwargs)
            init = 0 if is_first_iteration else 1
            init_reset = [init for _ in range(Baseconfig.batch_size)]

            lite_inputs[2] = np.array(init_reset).reshape(Baseconfig.batch_size, 1).astype(np.int32)

            shape_list = []
            type_list = []

            first_group = np.concatenate((lite_inputs[0], lite_inputs[1].reshape(Baseconfig.batch_size, 1),
                                          lite_inputs[2], lite_inputs[3].reshape(Baseconfig.batch_size, 1)), axis=1)
            first = np.ndarray(first_group.shape, dtype=first_group.dtype, buffer=shms[0].buf)
            first[:] = first_group[:]
            shape_list.append(first_group.shape)

            # 如果是prefill的话，需要将另外三个array也写到共享内存中
            # mask, freq_cos, freq_sin
            second = np.ndarray(lite_inputs[4].shape, dtype=lite_inputs[4].dtype, buffer=shms[1].buf)
            second[:] = lite_inputs[4][:]

            third = np.ndarray(lite_inputs[5].shape, dtype=lite_inputs[5].dtype, buffer=shms[2].buf)
            third[:] = lite_inputs[5][:]

            fourth = np.ndarray(lite_inputs[6].shape, dtype=lite_inputs[6].dtype, buffer=shms[3].buf)
            fourth[:] = lite_inputs[6][:]

            parms_np_dtype = np.float16
            parms_np = self.get_gen_parms_np(Baseconfig.batch_size, parms_np_dtype, **kwargs)
            gen_parms = np.ndarray(parms_np.shape, dtype=parms_np_dtype, buffer=shms[4].buf)
            gen_parms[:] = parms_np[:]

            shape_list.append(lite_inputs[4].shape)
            shape_list.append(lite_inputs[5].shape)
            shape_list.append(lite_inputs[6].shape)
            shape_list.append(parms_np.shape)

            shape_strs = []
            for shape in shape_list:
                shape_str = " ".join(str(element) for element in shape)
                shape_strs.append(shape_str)
            shapes_str = "*" + ",".join(element for element in shape_strs)
        else:
            shapes_str = "a"
        logging.info("get input lite is {} ".format((time.time() - time_start) * 1000))

        # socket
        shapes_str = shapes_str.encode()

        for item in self.agent_stubs:
            item.sendall(shapes_str)
        _ = self.agent_stubs[0].recv(1, socket.MSG_WAITALL).decode()

        result = [int(np.ndarray((1,), dtype=np.int32, buffer=shms[0].buf)[0])]

        logging.info("model.call time is {} ".format((time.time() - time_start) * 1000))
        return [int(result[0])], 1
