import sys
import time
from abc import ABC
from abc import abstractmethod
from mindspore_lite import Model
import logging
import numpy as np
import mindspore as ms

from config.serving_config import get_inputs_custom
from config.serving_config import Baseconfig, AgentConfig
from typing import List
import socket
from concurrent.futures import ThreadPoolExecutor, wait
from multiprocessing import shared_memory
import asyncio


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
                   init_reset=None, is_first_iteration=True, InputExtraList=[], **kwargs):
        mask = InputExtraList[0]
        freq_cos = InputExtraList[1]
        freq_sin = InputExtraList[2]
        if not is_first_iteration:
            inputs_tmp = []
            for i in range(len(current_index)):
                current_index_tmp = int(current_index[i]) - i * input_ids.shape[1]  # multibatch
                # use numpy to slice array to avoid complie ascend slice op
                
                inputs_tmp.append(input_ids[i][current_index_tmp:current_index_tmp + 1])
            input_ids = np.array(inputs_tmp, dtype=np.int32)
        if is_first_iteration:
            # mask, freq_cos, fre_sin 
            inputs = [input_ids, current_index, init_reset, valid_length, mask, freq_cos, freq_sin]
        else:
            inputs = [input_ids, current_index, init_reset, valid_length]
        return inputs

class CustomInputsOfInfer(BaseInputsOfInfer):
    """
    common infer inputs of llm models.
    """

    def __init__(self):
        self.get_input_from_config = get_inputs_custom

    # pylint: disable=W0221
    def get_inputs(self, **kwargs):


        return self.get_input_from_config(**kwargs)

        # print("inputs after get_inputs:{}".format(inputs))
        #lite_inputs = BaseInputsOfInfer.get_lite_tensor_list(inputs, model)
        # return lite_inputs
        inputs_custom = self.get_input_from_config(**kwargs)
        if inputs_custom is None:
            logging.error('custom inputs definited by customer is None,please check it in server config!')
        return inputs_custom
    
class InputOfInfer:
    """
    Input of llm model.
    """
    MAPPING = {
        "bloom": CommonInputsOfInfer,
        "llama": CommonInputsOfInfer,
        "glm2": CommonInputsOfInfer,
        "common": CommonInputsOfInfer,
        "llama_dyn": CommonInputsOfInferDyn,
        "internlm" : CommonInputsOfInfer,
        "baichuan2" : CommonInputsOfInfer,
        "custom" : CustomInputsOfInfer
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
        if Baseconfig['input_function'] == 'custom':
            model_name = "custom"
            logging.debug('model name {}'.format(model_name))
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

class CommonWarp:
    """
    common infer inputs of llm models.
    """

    def __init__(self):
        pass

    # pylint: disable=W0221
    def get_warp_inputs(self, lite_inputs=None,**kwargs):
        init = 0
        init_reset = [init for _ in range(Baseconfig.prefill_batch_size)]

        lite_inputs[2] = np.array(init_reset).reshape(Baseconfig.prefill_batch_size, 1).astype(np.int32)

        first_group = np.concatenate((lite_inputs[0], lite_inputs[1].reshape(Baseconfig.prefill_batch_size, 1),
                                        lite_inputs[2], lite_inputs[3].reshape(Baseconfig.prefill_batch_size, 1)), axis=1)
        second_group = []
        return first_group, second_group


class CommonWarpDyn:
    """
    common infer inputs of llm models.
    """

    def __init__(self):
        pass

    # pylint: disable=W0221
    def get_warp_inputs(self, lite_inputs=None, **kwargs):
       
        init = 0
        init_reset = [init for _ in range(Baseconfig.prefill_batch_size)]
        lite_inputs[2] = np.array(init_reset).reshape(Baseconfig.prefill_batch_size, 1).astype(np.int32)

        first_group = np.concatenate((lite_inputs[0], lite_inputs[1].reshape(Baseconfig.prefill_batch_size, 1),
                                        lite_inputs[2], lite_inputs[3].reshape(Baseconfig.prefill_batch_size, 1)), axis=1)
       
        second_group = []
        for i in range(4, len(lite_inputs)):
            second_group.append(lite_inputs[i])
        return first_group, second_group
        
    
class WarpInputOfInfer:
    """
    Input of llm model.
    """
    MAPPING = {
        "bloom": CommonWarp,
        "llama": CommonWarp,
        "glm2": CommonWarp,
        "common": CommonWarp,
        "llama_dyn": CommonWarpDyn,
        "internlm" : CommonWarp,
        "baichuan2" : CommonWarp,
    }

    @classmethod
    def get_warp_inputs(cls, model_name: str, **kwargs):
        """
        Get warpping input tensor list of mslite.

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
        return WarpInputOfInfer.MAPPING[name]().get_warp_inputs(**kwargs)


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

    def init(self, agent_ports, agent_ip, model_name, shm_names: List[str] = None):
        print(f"agent_ports is {agent_ports}")
        for port in agent_ports:
            print("port ip is {}".format(port))
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # socket是1对1的，设置超时机制，防止多个serving连接同一个LLM
            client.settimeout(5)
            client.connect((agent_ip, port))
            send_str = '#' + ",".join(str(element) for element in shm_names)
            client.sendall(send_str.encode())
            data = client.recv(6, socket.MSG_WAITALL)
            print(data)
            if data.decode() == "failed":
                print("there exists another connected serving now, stop the previous serving at first")
                sys.exit()
            self.agent_stubs.append(client)
            client.settimeout(None)
            # send shm_names
        self.model_name = model_name

    # def __del__(self):
    #     self.stop()

    def stop(self):
        print("waiting worker to exit")
        for item in self.agent_stubs:
            cnt = 0
            while True or cnt < 1000:
                item.sendall("e".encode())
                data = item.recv(4096).decode()
                print(data)
                if data == "free":
                    print("close socket")
                    item.close()
                    break
                cnt += 1
            if cnt >= 1000:
                print("agent is running now, failed to stop serving, try to stop later")
        print("exit!")


    def get_predict_inputs(self, input_ids, current_index=None,
                           valid_length=None, init_reset=None, is_first_iteration=True, **kwargs):
        """Get inputs of llm model for mslite."""
        return InputOfInfer.get_inputs(self.model_name, input_ids=input_ids, current_index=current_index,
                                       valid_length=valid_length, init_reset=init_reset,
                                       is_first_iteration=is_first_iteration, **kwargs)

    def get_model_inputs(self, input_ids, current_index=None,
                         valid_length=None, init_reset=None, is_first_iteration=True, **kwargs) -> np.array:
        if is_first_iteration:
            init_reset = np.array([False])
            lite_inputs = self.get_predict_inputs(input_ids, current_index,
                                                  valid_length, init_reset, is_first_iteration, **kwargs)

        else:
            init_reset = np.array([True])
            lite_inputs = self.get_predict_inputs(input_ids, current_index,
                                                  valid_length, init_reset, is_first_iteration, **kwargs)

        return lite_inputs
    
    def get_warp_inputs(self, lite_inputs=None, **kwargs):
        """Get inputs of llm model for mslite."""
        return WarpInputOfInfer.get_warp_inputs(self.model_name, lite_inputs=lite_inputs, **kwargs)

    def get_gen_parms_np(self, batch_size, dtype=np.float16, **kwargs):
        do_sample_list = kwargs.pop("do_sample_list")
        top_k_list = kwargs.pop("top_k_list")
        top_p_list = kwargs.pop("top_p_list"),
        temperature_list = kwargs.pop("temperature_list"),
        repetition_penalty = kwargs.pop("repetition_penalty")
        decode_index_list = kwargs.pop("decode_index_list")

        do_sample_np = np.array(do_sample_list).reshape(batch_size, 1).astype(dtype)
        top_p_np = np.array(top_p_list).reshape(batch_size, 1).astype(dtype)
        top_k_np = np.array(top_k_list).reshape(batch_size, 1).astype(dtype)
        temperature_np = np.array(temperature_list).reshape(batch_size, 1).astype(dtype)
        repetition_np = np.array(repetition_penalty).reshape(batch_size, 1).astype(dtype)
        decode_index_np = np.array(decode_index_list).reshape(batch_size, 1).astype(dtype)
        parms_np = np.concatenate((do_sample_np, top_p_np, top_k_np, temperature_np, repetition_np, decode_index_np),
                                  axis=-1)
        return parms_np

    def callV3(self, shms: List, input_ids, current_index,
               valid_length, init_reset, is_first_iteration, valid_batch_flag, InputExtraList=[],
               current_batch_size=None, **kwargs):
        """kvcache infer"""
        time_start = time.time()
        logging.debug("is prefill {}".format(is_first_iteration))
        decode_index_list = kwargs.get("decode_index_list")
        if is_first_iteration:
            lite_inputs = self.get_model_inputs(input_ids, current_index, valid_length,
                                                init_reset, is_first_iteration, InputExtraList=InputExtraList, **kwargs)
            # 前4个array拼接成一个
            # init_reset变成[batch_size, 1]
            logging.debug("prefill input_ids {} \ncurrent_index {} \n valid_length {} \n".format(input_ids,
                                                                                                 current_index,
                                                                                                 valid_length))
            first_group, second_group = self.get_warp_inputs(lite_inputs=lite_inputs, **kwargs)
            
            shape_list = []
            first = np.ndarray(first_group.shape, dtype=first_group.dtype, buffer=shms[0].buf)
            first[:] = first_group[:]
            shape_list.append(first_group.shape)

            # 如果是prefill的话，需要将另外三个array也写到共享内存中

            if len(second_group) != 0:
                for j in range(len(second_group)):
                    
                    item = np.ndarray(second_group[j].shape, dtype=second_group[j].dtype, buffer=shms[1 + j].buf)
                    
                    item[:] = second_group[j][:]
                   
                    shape_list.append(second_group[j].shape)
            mem_index = len(second_group)
            parms_np_dtype = np.float16
            parms_np = self.get_gen_parms_np(Baseconfig.prefill_batch_size, parms_np_dtype, **kwargs)
            gen_index = max(3, mem_index)
            gen_parms = np.ndarray(parms_np.shape, dtype=parms_np_dtype, buffer=shms[gen_index + 1].buf)
            gen_parms[:] = parms_np[:]

            shape_list.append(parms_np.shape)

            shape_strs = []
            for shape in shape_list:
                shape_str = " ".join(str(element) for element in shape)
                shape_strs.append(shape_str)
            shapes_str = "*" + ",".join(element for element in shape_strs)
        else:
            logging.debug("valid_batch_flag in decode is {}".format(valid_batch_flag))
            batch_flag_str = " ".join(str(element) for element in valid_batch_flag)
            shapes_str = "a" + '_' + str(current_batch_size) + '_' + batch_flag_str
        logging.info("get input lite is {} ".format((time.time() - time_start) * 1000))
        logging.info("server decode batch size is {} ".format(current_batch_size))
        shapes_str = shapes_str.encode()

        for item in self.agent_stubs:
            item.sendall(shapes_str)
        recv_data = self.agent_stubs[0].recv(1, socket.MSG_WAITALL).decode()

        result = []
        if recv_data == "2":
            for _ in decode_index_list:
                # result.append(int(Baseconfig.end_token))
                result.append(int(-1))
            print("--------------------predict failed, abandon current prompt, please try again----------------")
            return result, 1
        for decode_index in decode_index_list:
            tmp = np.ndarray((decode_index + 1,), dtype=np.int32, buffer=shms[5].buf)
            result.append(int(tmp[decode_index:decode_index + 1]))
        logging.info("model.call time is {} ".format((time.time() - time_start) * 1000))
        return result, 1


