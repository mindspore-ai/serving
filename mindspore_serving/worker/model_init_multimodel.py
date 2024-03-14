import time
import logging
import numpy as np
from typing import List
import socket
from mindspore_serving.config.config import ServingConfig
from mindspore_serving.models.build_inputs import build_inputs


class BaseInputsOfInfer:
    """
    BaseInputsOfInfer interface.
    """

    def get_inputs(self, model, **kwargs):
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
        # lite_inputs = BaseInputsOfInfer.get_lite_tensor_list(inputs, model)
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
        "wizard_coder": CommonInputsOfInferDyn,
        "internlm": CommonInputsOfInfer,
        "baichuan2": CommonInputsOfInfer,
        "custom": CustomInputsOfInfer
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
        # name = ""
        # if Baseconfig['input_function'] == 'custom':
        #     model_name = "custom"
        #     logging.debug('model name {}'.format(model_name))
        # if model_name not in InputOfInfer.MAPPING:
        #     for k in InputOfInfer.MAPPING:
        #         if model_name.startswith(k):
        #             name = k
        #             break
        #     if not name:
        #         logging.warning("Model name not in support maps.Common input format will be used to do inference.")
        #         name = "common"
        # else:
        #     name = model_name
        return InputOfInfer.MAPPING['common']().get_inputs(**kwargs)


class CommonWarp:
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
        "wizard_coder": CommonWarpDyn,
        "internlm": CommonWarp,
        "baichuan2": CommonWarp,
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
        self.config = None

    def init(self, config, shm_names: List[str] = None):
        self.config = config
        agent_ip = config.serving_config.agent_ip
        agent_ports = config.serving_config.agent_ports
        model_name = config.model_config.model_name
        print(f"agent_ports is {agent_ports}")
        for port in agent_ports:
            print("port ip is {}".format(port))
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # socket是1对1的，设置超时机制，防止多个serving连接同一个LLM
            client.settimeout(5)
            client.connect((agent_ip, port))
            send_str = '#' + ",".join(str(element) for element in shm_names)
            client.sendall(send_str.encode())
            data = client.recv(6, socket.MSG_WAITALL).decode()
            print(data)
            if data == "failed":
                client.close()
                for agent in self.agent_stubs:
                    agent.close()
                raise ConnectionError("there exists another connected serving now, stop the previous serving at first")
            self.agent_stubs.append(client)
            client.settimeout(None)
            # send shm_names
        self.model_name = model_name

    @staticmethod
    def reset_agent_status(config):
        print("waiting to reset agents status")
        agent_ip = config.serving_config.agent_ip
        agent_ports = config.serving_config.agent_ports
        for port in agent_ports:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # socket是1对1的，设置超时机制，防止多个serving连接同一个LLM
            client.settimeout(5)
            client.connect((agent_ip, port))
            client.sendall("r".encode())
            data = client.recv(6, socket.MSG_WAITALL).decode()
            print(data)
            if data == "succes":
                print("reset")
        print("reset all agents!")

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
                         valid_length=None, is_first_iteration=True, **kwargs) -> np.array:
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

    @staticmethod
    def get_gen_parms_np(batch_size, dtype=np.float16, **kwargs):
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

    def _assemble_pa_inputs(self, is_first_iteration, batch_valid_length: np.array, cache_engine_list, seq_length,
                            valid_batch_flag):
        if is_first_iteration:
            return self._assemble_pa_full_inputs(batch_valid_length, cache_engine_list, seq_length, valid_batch_flag)
        else:
            return self._assemble_pa_inc_inputs(batch_valid_length, cache_engine_list, seq_length, valid_batch_flag)

    def _assemble_pa_full_inputs(self, batch_valid_length: np.array, cache_engine_list, seq_length, valid_batch_flag):
        block_size = cache_engine_list[0].block_size
        max_num_blocks_per_seq = seq_length // block_size

        bs = len(valid_batch_flag)
        block_tables = []
        slot_mapping = []
        for i in range(bs):
            if valid_batch_flag[i]:
                cache_engine_list[i].prepare_cache(batch_valid_length[i])
            # 预留出首个块，给冗余写用，全量需要这个 TODO:后续优化ReshapeAndCache逻辑，跳过冗余位置
            block_table = cache_engine_list[i].block_table
            # padded_table = block_table + [ -1 for _ in range(max_num_blocks_per_seq - len(cache_engine_list[i].block_table) + 1)]
            padded_table = block_table + [-1 for _ in
                                          range(max_num_blocks_per_seq - len(cache_engine_list[i].block_table))]
            block_tables.append(padded_table)

            slots = [block_table[k // block_size] * block_size + k % block_size for k in range(batch_valid_length[i])]
            null_slot_idx = 0
            slots = slots + [null_slot_idx for _ in range(seq_length - batch_valid_length[i])]
            slot_mapping = slot_mapping + slots
        block_tables = np.array(block_tables, dtype=np.int32)
        slot_mapping = np.array(slot_mapping, dtype=np.int32)
        return block_tables, slot_mapping

    def _assemble_pa_inc_inputs(self, batch_valid_length: np.array, cache_engine_list, seq_length, valid_batch_flag):
        block_size = cache_engine_list[0].block_size
        max_num_blocks_per_seq = seq_length // block_size
        bs = len(valid_batch_flag)
        block_tables = []
        slot_mapping = []
        for i in range(bs):
            if valid_batch_flag[i]:
                cache_engine_list[i].prepare_cache(1)  # 增量推理时，每个序列新增一个token。
                valid_length = cache_engine_list[i].num_token  # - block_size
            else:
                valid_length = 1
            block_table = cache_engine_list[i].block_table
            padded_table = block_table + [-1 for _ in
                                          range(max_num_blocks_per_seq - len(cache_engine_list[i].block_table))]
            block_tables.append(padded_table)
            curent_idx = valid_length - 1

            slots = [block_table[curent_idx // block_size] * block_size + curent_idx % block_size]
            slot_mapping = slot_mapping + slots
        block_tables = np.array(block_tables, dtype=np.int32)
        slot_mapping = np.array(slot_mapping, dtype=np.int32)
        return block_tables, slot_mapping

    def call(self, shms: List, input_ids, current_index,
             valid_length, init_reset, is_first_iteration, valid_batch_flag, extra_inputs=None,
             current_batch_size=None, **kwargs):
        """kvcache infer"""
        time_start = time.time()
        logging.debug("is prefill {}".format(is_first_iteration))
        decode_index_list = kwargs.get("decode_index_list")
        # 加入pa
        if self.config.model_config.page_attention:
            cache_engine_list = kwargs.get("cache_engine_list")
            seq_length = kwargs.get("seq_length")
        if is_first_iteration:
            lite_inputs = self.get_model_inputs(input_ids, current_index, valid_length,
                                                is_first_iteration, extra_inputs=extra_inputs, **kwargs)
            # 前4个array拼接成一个
            # init_reset变成[batch_size, 1]
            # first_group, second_group = self.get_warp_inputs(lite_inputs=lite_inputs, **kwargs)
            init = 0
            prefill_bs = len(input_ids)
            init_reset = [init for _ in range(prefill_bs)]
            lite_inputs[2] = np.array(init_reset).reshape(prefill_bs, 1).astype(np.int32)

            first_group = np.concatenate((lite_inputs[0], lite_inputs[1].reshape(prefill_bs, 1),
                                          lite_inputs[2], lite_inputs[3].reshape(prefill_bs, 1)), axis=1)
            shape_list = []
            first = np.ndarray(first_group.shape, dtype=first_group.dtype, buffer=shms[0].buf)
            first[:] = first_group[:]
            shape_list.append(first_group.shape)

            # 如果是prefill的话，需要将另外三个array也写到共享内存中
            second_group = []
            for i in range(4, len(lite_inputs)):
                second_group.append(lite_inputs[i])
            logging.debug("second_group {}".format(second_group))
            if len(second_group) != 0:
                for j in range(len(second_group)):
                    logging.debug("second_group index {}".format(j))
                    item = np.ndarray(second_group[j].shape, dtype=second_group[j].dtype, buffer=shms[1 + j].buf)
                    item[:] = second_group[j][:]
                    shape_list.append(second_group[j].shape)
            # mem_index = len(second_group)
            params_np_dtype = np.float16
            params_np = self.get_gen_parms_np(prefill_bs, params_np_dtype, **kwargs)
            # gen_index = max(3, mem_index)

            gen_params = np.ndarray(params_np.shape, dtype=params_np_dtype, buffer=shms[4].buf)
            gen_params[:] = params_np[:]

            shape_list.append(params_np.shape)

            shape_strs = []
            for shape in shape_list:
                shape_str = " ".join(str(element) for element in shape)
                shape_strs.append(shape_str)
            shapes_str = "*" + ",".join(element for element in shape_strs)
        else:
            logging.debug("valid_batch_flag in decode is {}".format(valid_batch_flag))
            batch_flag_str = " ".join(str(element) for element in valid_batch_flag)
            shapes_str = "a" + '_' + str(current_batch_size) + '_' + batch_flag_str

        # 加入pa
        if self.config.model_config.page_attention:
            block_tables, slot_mapping = self._assemble_pa_inputs(is_first_iteration, valid_length, cache_engine_list,
                                                                  seq_length, valid_batch_flag)
            block_tables_np = np.array(block_tables, dtype=np.int32)
            block_tables_shm = np.ndarray(block_tables_np.shape, dtype=block_tables_np.dtype, buffer=shms[7].buf)
            block_tables_shm[:] = block_tables_np[:]
            slot_mapping_np = np.array(slot_mapping, dtype=np.int32)
            slot_mapping_shm = np.ndarray(slot_mapping_np.shape, dtype=slot_mapping_np.dtype, buffer=shms[8].buf)
            slot_mapping_shm[:] = slot_mapping_np[:]

            shape_strs = []
            for shape in [block_tables_np.shape, slot_mapping_np.shape]:
                shape_str = " ".join(str(element) for element in shape)
                shape_strs.append(shape_str)
            if is_first_iteration:
                shapes_str += "," + ",".join(element for element in shape_strs)
            else:
                shapes_str += "_" + "_".join(element for element in shape_strs)
        logging.debug("get input lite is {} ".format((time.time() - time_start) * 1000))
        logging.debug("server decode batch size is {} ".format(current_batch_size))
        shapes_str = shapes_str.encode()

        for item in self.agent_stubs:
            item.sendall(shapes_str)
        recv_data = self.agent_stubs[0].recv(1, socket.MSG_WAITALL).decode()

        result = []
        if recv_data == "2":
            for _ in decode_index_list:
                # result.append(int(Baseconfig.end_token))
                result.append((int(-1),0))
            print("--------------------predict failed, abandon current prompt, please try again----------------")
            logging.error("predict failed, abandon current prompt, please try again")
            return result, 1
        for decode_index in decode_index_list:
            tmp = np.ndarray((decode_index + 1,), dtype=np.int32, buffer=shms[5].buf)
            tmp_logprob = np.ndarray((decode_index + 1,), dtype=np.float64, buffer=shms[6].buf)
            result.append((int(tmp[decode_index:decode_index + 1]), float(tmp_logprob[decode_index:decode_index + 1])))

        logging.info("--------------------callV3 result value is {} ".format(result))
        logging.info("model.call time is {} ".format((time.time() - time_start) * 1000))
        return result, 1
