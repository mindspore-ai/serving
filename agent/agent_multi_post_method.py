"""agent"""

import argparse
import logging
import signal
import threading
import time
from concurrent import futures
from multiprocessing import Process
import socket
import copy

import mindspore_lite as mslite
import mindspore as ms
import psutil
import os
import numpy as np
import numpy
from sub_process.sub_process import listen_agents_after_startup
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

from multiprocessing import shared_memory
import logging
import time
from config.serving_config import Baseconfig, AgentConfig, AgentIP, get_warmup_inputs, ExtraInput
from models.post_sampling.topk import post_sampling, softmax_np
from mindspore_lite import Tensor, DataType
from serving_utils.err_code import AgentStatus

pool = ThreadPoolExecutor(max_workers=20, thread_name_prefix='test_thread')

VOCAB_LEN = Baseconfig.vocab_size
batch_size = Baseconfig.batch_size
PORTS = AgentConfig.AgentPorts


class Config:
    def __init__(self, device_id, rank_id, config_file, config_inc_file, config_post_sampling,
                 model0_path, model1_path, post_sampling_model_path, post_sampling_model_path2, agent_address, index):
        self.device_id = device_id
        self.rank_id = rank_id
        self.config_file = config_file
        self.config_inc_file = config_inc_file
        self.config_post_sampling = config_post_sampling
        self.model0_path = model0_path
        self.model1_path = model1_path
        self.post_sampling_model_path = post_sampling_model_path,
        self.post_sampling_model_path2 = post_sampling_model_path2,
        self.agent_address = agent_address
        self.index = index


def load_model(model0_path, model1_path, config_file, config_inc_file_list, rank_id, device_id):
    # 加载模型
    context = mslite.Context()
    print('device_id: ', device_id)
    print('rank_id: ', rank_id)
    context.ascend.device_id = device_id
    context.ascend.rank_id = rank_id
    context.ascend.provider = "ge"
    context.target = ["Ascend"]
    # 单模型
    if len(model1_path) == 0:
        model0 = mslite.Model()
        model0.build_from_file(model0_path, mslite.ModelType.MINDIR, context, config_file)
        model1 = None
        return model0, model1

    # rank_table_file放在config_file中
    all_models = [mslite.Model()]  # prefill
    # decode
    for _ in config_inc_file_list:
        all_models.append(mslite.Model())
    model_group = mslite.ModelGroup(mslite.ModelGroupFlag.SHARE_WEIGHT)
    model_group.add_model(all_models)

    all_models[0].build_from_file(model0_path, mslite.ModelType.MINDIR, context, config_file)
    # warm up prefill model
    logging.info('warmup prefill model ...')
    prefill_inputs_list = get_warmup_inputs(batch_size=1, full_model=True)
    prefill_lite_inputs = [mslite.Tensor(item) for item in prefill_inputs_list]
    for item in prefill_lite_inputs:
        print("prefill item ", item.shape, item.dtype)
    all_models[0].predict(prefill_lite_inputs)
    logging.info('warmup prefill model finish')
    logging.info('warmup decode model ...')
    if 'zactivate_len' not in Baseconfig or len(config_inc_file_list) != len(Baseconfig.zactivate_len):
        logging.error('zactivate len config is not consistent with config_inc_file_list')
    for i in range(len(config_inc_file_list)):
        act_len = Baseconfig.zactivate_len[i]
        print(f"starting warm up {act_len} decode model")
        if 'dyn_batch_size' in Baseconfig:
            warm_batch_size = Baseconfig.dyn_batch_size[0]
        else:
            warm_batch_size = Baseconfig.batch_size
        if 'seq_type' in Baseconfig and Baseconfig.seq_type == 'dyn':
            warm_seq_length = 1
        else:
            warm_seq_length = Baseconfig.seq_length[0]

        decode_inputs_list = get_warmup_inputs(seq_length=warm_seq_length, batch_size=warm_batch_size, full_model=False, valid_length=[act_len - 1])

        decode_lite_inputs = [mslite.Tensor(item) for item in decode_inputs_list]

        for item in decode_lite_inputs:
            print("decode item ", item.shape, item.dtype, flush=True)
            print(item, flush=True)

        all_models[i + 1].build_from_file(model1_path, mslite.ModelType.MINDIR, context, config_inc_file_list[i])
        all_models[i + 1].predict(decode_lite_inputs)
        print(f"finish warm up {act_len} decode model")

    print('warmup all decode models finish', flush=True)
    print(f"load model {model0_path} and {model1_path} in rank {rank_id} successful", flush=True)
    return all_models[0], all_models[1:]


def load_post_model(model_path, config_file, rank_id, device_id):
    context = mslite.Context()
    print('device_id: ', device_id)
    print('rank_id: ', rank_id)
    context.ascend.device_id = device_id
    context.ascend.rank_id = rank_id
    context.ascend.provider = "ge"
    context.target = ["Ascend"]
    print('post sampling config: ', config_file)
    model = mslite.Model()
    print('post model path:', type(model_path[0]))
    model.build_from_file(model_path[0], mslite.ModelType.MINDIR, context, config_file)
    print(f"load post-sampling model  {model_path[0]} successful")
    return model


class DecodeParams:
    def __init__(self,
                 do_sample: bool = True,
                 top_k: int = 1,
                 top_p: float = 1.0,
                 temperature: float = 1.0,
                 repetition_penalty: float = 1.0,
                 decode_index: int = -1,
                 current_index: int = 0,
                 valid_length: int = 0,
                 init_reset: bool = False,
                 ge_token: int = 0
                 ):
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.decode_index = decode_index
        self.current_index = current_index
        self.valid_length = valid_length
        self.init_reset = init_reset
        self.ge_token = ge_token


"""
work_agent.proto实现, 供worker调用
"""


class WorkAgent:
    def __init__(self, args):
        device_id = args.device_id
        rank_id = args.rank_id
        self.rank_id = rank_id
        config_file = args.config_file
        config_inc_file = args.config_inc_file
        config_post_sampling = args.config_post_sampling
        model0_path = args.model0_path
        model1_path = args.model1_path
        model2_path = args.post_sampling_model_path
        model3_path = args.post_sampling_model_path2
        self.index = args.index
        print(f'model0_path is {model0_path}')
        print(f'model1_path is {model1_path}')
        self.prefill, self.decode = load_model(model0_path, model1_path, config_file, config_inc_file, rank_id,
                                               device_id)
        self.argmax_model = load_post_model(model2_path, config_post_sampling, rank_id, device_id)
        self.topk_model = load_post_model(model3_path, config_post_sampling, rank_id, device_id)
        self.shm_names = []
        self.init_reset = None
        self.current_index = None
        self.valid_length = None
        self.tensor_shape = None
        self.pre_input_ids = None
        self.is_prefill = True
        self.target = None
        self.post_mode_list = None
        self.input_length = None
        self.targets = []
        self.decode_params_map = {}
        self.status = AgentStatus.unconnected
        self.current_batch_size = None

    def _post_smapling_argmax_npu(self, outputs_np) -> np.ndarray:
        """
        Args:
           outputs_np: np.ndarray or ms.Tensor, (bs, 1, vocab_size)
        """
        post_inputs = self.argmax_model.get_inputs()
        if isinstance(outputs_np, numpy.ndarray):
            post_inputs[0].shape = outputs_np.shape
            post_inputs[0].set_data_from_numpy(outputs_np)
        else:
            post_inputs[0].shape = outputs_np.shape
            post_inputs[0] = outputs_np
        post_sampling_out = self.argmax_model.predict(post_inputs)
        return post_sampling_out[0].get_data_to_numpy().astype(np.int32)

    def _post_sampling_argmax_host(self, outputs) -> np.ndarray:
        if isinstance(outputs, Tensor):
            outputs = outputs.get_data_to_numpy()
        outputs.reshape((outputs.shape[0], outputs.shape[-1]))
        argmax_out = np.argmax(outputs, axis=-1)
        return np.array([argmax_out]).astype(np.int32)[0]

    def do_sample(self, decode_params, p_args, outs, targets, index, candidate_token_num: int = 100) -> int:
        """
        Args:
           p_args: numpy.ndarray, index
           outs: numpy.ndarray, probs
        """
        topp = decode_params.top_p
        topk = decode_params.top_k
        if topk > 100:
            logging.error('top k is out of range,please set topk in [1,100]')
            topk = 100
        if topp < 1.0:
            top_p_num = sum(outs > topp)
            if top_p_num == 0:
                top_p_num = candidate_token_num
            top_p_num = min(top_p_num, topk)
            outs = outs[:top_p_num]
            p_args = p_args[:top_p_num]
            if np.sum(outs) == 0:
                outs = np.array([1 / top_p_num for _ in range(top_p_num)])
            p = softmax_np(outs)
        else:
            p = outs[:topk]
            p = softmax_np(p)
            p_args = p_args
        target_index = np.random.choice(len(p), p=p)
        targets[index] = p_args[target_index]

    def _post_sampling_topk_npu(self, outputs_np, decode_index, prefill=True) -> np.ndarray:
        """
        Args:
           outputs_np: np.ndarray or ms.Tensor, (bs, 1, vocab_size)
        """
        decode_params = self.decode_params_map[int(decode_index[0])]
        self.targets.clear()
        tempreture_ = np.array([decode_params.temperature], dtype=np.float32)

        post_inputs = self.topk_model.get_inputs()

        if isinstance(outputs_np, numpy.ndarray):
            post_inputs[0].shape = outputs_np.shape
            post_inputs[0].set_data_from_numpy(outputs_np)

        else:
            post_inputs[0].shape = outputs_np.shape
            post_inputs[0] = outputs_np

        post_inputs[1].shape = tempreture_.shape
        post_inputs[1].set_data_from_numpy(tempreture_)

        post_sampling_out = self.topk_model.predict(post_inputs)
        outs = post_sampling_out[0].get_data_to_numpy().astype(np.float16)
        p_args = post_sampling_out[1].get_data_to_numpy()
        thread_num = 1
        if prefill:
            thread_num = Baseconfig.prefill_batch_size
        else:
            thread_num = self.current_batch_size
        targets = np.zeros((thread_num,), np.int32)
        all_task = [pool.submit(self.do_sample, self.decode_params_map[decode_index[i]], p_args[i], outs[i], targets, i)
                    for i in range(thread_num)]
        wait(all_task)
        return targets

    def _post_sampling_topk_host(self, outputs, decode_index, prefill):
        """
         topk top-p in cpu, time-cost function,
        """
        if isinstance(outputs, Tensor):
            outputs = outputs.get_data_to_numpy()
        outputs = np.reshape(outputs, (outputs.shape[0], outputs.shape[-1]))
        if prefill:
            thread_num = Baseconfig.prefill_batch_size
        else:
            thread_num = self.current_batch_size
        targets = np.zeros((thread_num,), np.int32)
        all_task = [pool.submit(post_sampling, np.array(item), self.decode_params_map[decode_index[i]], targets, i)
                    for i, item in enumerate(outputs)]
        wait(all_task)
        return targets


    def multi_thread_post_sampling(self, outputs_np, outputs_shm, decode_index_np, bs=1):

        self.targets.clear()
        all_task = [pool.submit(self.do_post_sampling, outputs_np[i], outputs_shm,
                                decode_index_np[i], i) for i in range(bs)]

        for x in as_completed(all_task):
            res = x.result()
            self.targets.append(res)
        return self.targets

    def get_consistent_batch(self, decode_index):
        not_do_sample_list = []
        do_sample_list = []
        for index in decode_index:
            do_sample_index = self.decode_params_map[index].do_sample
            if do_sample_index is True:
                do_sample_list.append(index)
            else:
                not_do_sample_list.append(index)
        if len(do_sample_list) >= 1 and len(not_do_sample_list) >= 1:
            for item in not_do_sample_list:
                self.decode_params_map[item].top_k = 1
            do_sample = True
        else:
            do_sample = self.decode_params_map[decode_index[0]].do_sample
        return do_sample

    def do_post_sampling(self, outputs_np, outputs_shm, decode_index, prefill=True) -> np.ndarray:
        index = int(decode_index[0])
        do_sample = self.get_consistent_batch(decode_index)
        if AgentConfig.enable_host_post_sampling:
            if not do_sample:
                target = self._post_sampling_argmax_host(outputs_np)
                if prefill:
                    target.reshape((1,))
                else:
                    target.reshape((self.current_batch_size,))
                    target = np.squeeze(target, axis=1)
            else:
                target = self._post_sampling_topk_host(outputs_np, decode_index, prefill)
        else:
            if not do_sample:
                target = self._post_smapling_argmax_npu(outputs_np)
            else:
                target = self._post_sampling_topk_npu(outputs_np, decode_index, prefill)
        if self.index == 0 and prefill:
            tmp = np.ndarray((index + 1,), dtype=target.dtype, buffer=outputs_shm.buf)
            tmp[index: index + 1] = target[:]
            self.targets[index: index + 1] = target[:]
        elif self.index == 0 and prefill == False:
            tmp = np.ndarray((self.current_batch_size,), dtype=target.dtype, buffer=outputs_shm.buf)
            tmp[:] = target[:]
            self.targets[:] = target[:]

    def model_choice_seq(self, act_len, decode_model_map):
        if len(decode_model_map) == 1:
            return decode_model_map[0]
        act_len_list = Baseconfig.zactivate_len
        if len(act_len_list) != len(decode_model_map):
            logging.error('act_len config is inconsistent with decode'
                          'model ini,please check them')
        model_index = act_len_list.index(act_len)
        logging.debug('current act_len model is: {}'.format(act_len))
        return decode_model_map[model_index]

    def predict(self, shape_list=None, current_batch=None, batch_valid_flag=None):
        self.status = AgentStatus.busy
        tmp_shms = []
        start_time = time.time()
        existing_shm0 = shared_memory.SharedMemory(name=self.shm_names[0])
        tmp_shms.append(existing_shm0)

        output_shm = shared_memory.SharedMemory(name=self.shm_names[5])
        tmp_shms.append(output_shm)

        gen_parms_id = 4

        gen_parms_shm = shared_memory.SharedMemory(name=self.shm_names[gen_parms_id])
        tmp_shms.append(gen_parms_shm)
        self.current_batch_size = current_batch if current_batch else Baseconfig.prefill_batch_size

        logging.info(f"batch_size right now is {self.current_batch_size}")

        if self.is_prefill:
            first_group = np.ndarray((shape_list[0]), dtype=np.int32, buffer=existing_shm0.buf)
            current_index_ = first_group[:, shape_list[0][1] - 3: shape_list[0][1] - 2]
            current_index = np.squeeze(current_index_, axis=-1)

            valid_length_ = first_group[:, shape_list[0][1] - 1: shape_list[0][1]]
            if Baseconfig.model_type == 0:
                valid_length = np.squeeze(valid_length_, axis=-1).astype(np.int64)
            else:
                valid_length = np.squeeze(valid_length_, axis=-1).astype(np.int32)
            input_ids = first_group[:, :shape_list[0][1] - 3]
            self.input_lentgth = input_ids.shape[1]
            gen_parms_id = -1
            shape_parms = shape_list[gen_parms_id]
            gen_parms = np.ndarray((shape_parms), dtype=np.float16, buffer=gen_parms_shm.buf)

            do_sample_list = gen_parms[:, 0].astype(np.bool_)
            top_p_list = gen_parms[:, 1]
            top_k_list = gen_parms[:, 2].astype(np.int32)
            temperature_list = gen_parms[:, 3]
            repetition_penalty_list = gen_parms[:, 4]
            decode_index_list = gen_parms[:, 5].astype(np.int32)
            extra_input = []
            for i in range(1, len(shape_list) - 1):
                existing_shm = shared_memory.SharedMemory(name=self.shm_names[i])
                tmp_shms.append(existing_shm)
                # To Do np.int64 ?
                extra_input.append(np.ndarray((shape_list[i]), dtype=np.int64, buffer=existing_shm.buf))

            decode_params = DecodeParams(
                do_sample=bool(do_sample_list[0]),
                top_p=top_p_list[0],
                top_k=int(top_k_list[0]),
                temperature=temperature_list[0],
                repetition_penalty=repetition_penalty_list[0],
                decode_index=int(decode_index_list[0]),
                current_index=int(current_index[0]),
                valid_length=int(valid_length[0]),
                init_reset=True
            )

            self.decode_params_map[decode_params.decode_index] = decode_params
            init_reset = np.array([decode_params.init_reset], dtype=np.bool_)
            decode_index_np = np.array([decode_params.decode_index], dtype=np.int64)
            self.shape_list = shape_list

        else:
            # keep decode map size equal to current batch size
            # extend
            current_index = []
            valid_length = []
            init_reset = []
            decode_index = []
            current_batch_size = self.current_batch_size
            if self.current_batch_size != len(batch_valid_flag):
                logging.error("batch size is not equal to the length of batch_valid_flag: batch size is {}, "
                              "batch_valid_flag is {}".format(self.current_batch_size, batch_valid_flag))
                batch_valid_flag.clear()
                batch_valid_flag = [1 for _ in range(self.current_batch_size)]
            before_batch_size = len(self.decode_params_map.keys())
            if before_batch_size < current_batch_size:
                input_ids = np.ndarray((before_batch_size,), dtype=np.int32, buffer=output_shm.buf)
                pad_input_id = Baseconfig.end_token
                add_length = self.current_batch_size - before_batch_size
                addition_input_ids = np.array(add_length * [pad_input_id], dtype=np.int32)
                input_ids = np.append(input_ids, addition_input_ids)
                target_batch = self.current_batch_size
                pad_key = list(self.decode_params_map.keys())[-1]
                padding_obj = self.decode_params_map[pad_key]
                for j in range(target_batch):
                    if j not in self.decode_params_map:
                        self.decode_params_map[j] = padding_obj
            else:
                # pop
                while len(self.decode_params_map.keys()) > current_batch_size:
                    self.decode_params_map.popitem()
                input_ids = np.ndarray((current_batch_size,), dtype=np.int32, buffer=output_shm.buf)

            self.decode_params_map = dict(sorted(self.decode_params_map.items(), key=lambda x: x[0]))

            for key in self.decode_params_map.keys():
                decode_params = self.decode_params_map[key]
                decode_params.current_index = decode_params.current_index + 1
                decode_params.valid_length = decode_params.valid_length + 1
                decode_params.init_reset = False
                if batch_valid_flag[key] == 1:
                    current_index.append(decode_params.current_index)
                    valid_length.append(decode_params.valid_length)
                else:
                    current_index.append(0)
                    valid_length.append(0)
                init_reset.append(decode_params.init_reset)
                decode_index.append(decode_params.decode_index)

            extra_input = ExtraInput(input_ids, current_index, None, False, valid_length)
            current_index = np.array(current_index, dtype=np.int32)
            if Baseconfig.model_type == 0:
                valid_length = np.array(valid_length, dtype=np.int64)
            else:
                valid_length = np.array(valid_length, dtype=np.int32)
            init_reset = np.array(init_reset, dtype=np.bool_)
            decode_index_np = np.array(decode_index, dtype=np.int64)
            input_ids = input_ids.reshape((-1, 1))

        if Baseconfig['batching_strategy'] == 'continuous':
            logging.info('continous batching input')
            tmp_in = [input_ids, current_index, valid_length, decode_index_np]

        else:
            tmp_in = [input_ids, current_index, init_reset, valid_length]

        if len(extra_input) > 0:
            tmp_in.extend(extra_input)
        for tmp in tmp_in:
            logging.debug("item shape is {}, dtype is {}".format(tmp.shape, tmp.dtype))
            logging.debug("item is {}".format(tmp))
        # 调用ms lite进行推理
        if len(extra_input[0]) > 0:
            model = self.prefill if self.is_prefill else self.model_choice_seq(len(extra_input[0]), self.decode)
        else:
            model = self.prefill if self.is_prefill else self.decode[0]
        lite_inputs = [mslite.Tensor(item) for item in tmp_in]

        logging.info('agent pre-process time is {}'.format((time.time() - start_time) * 1000))

        input_batch_size, seq_length = input_ids.shape
        predict_time = time.time()
        if self.is_prefill:
            outputs_list = model.predict(lite_inputs)
        else:
            outputs_list = model.predict(lite_inputs)
        logging.debug("outputs tensor after model predict is {}".format(outputs_list[0]))
        logging.info('predict time is {}'.format((time.time() - predict_time) * 1000))

        post_time = time.time()
        if self.rank_id == 0:
            multi_thread_time = time.time()
            if self.is_prefill:
                self.do_post_sampling(outputs_list[0], output_shm, decode_index_np, prefill=True)
            else:
                self.do_post_sampling(outputs_list[0], output_shm, decode_index_np, prefill=False)
                logging.info('multi_thread_post_sampling time is {}'.format((time.time() - multi_thread_time) * 1000))
            logging.info('target is  {}'.format((self.targets)))
        logging.info('post_time is {}'.format((time.time() - post_time) * 1000))
        logging.info('npu_total_time is {}'.format((time.time() - start_time) * 1000))
        self.status &= ~AgentStatus.busy
        return self.targets, tmp_shms


def warmup_models(work_agent):
    logging.info('warmup prefill model ...')
    print('warmup prefill model ...')
    prefill_inputs_list = get_warmup_inputs(batch_size=1, full_model=True)
    prefill_lite_inputs = [mslite.Tensor(item) for item in prefill_inputs_list]

    for item in prefill_lite_inputs:
        print("prefill item ", item.shape, item.dtype)
    work_agent.prefill.predict(prefill_lite_inputs)
    logging.info('warmup prefill model finish')
    print('warmup prefill model finish')

    logging.info('warmup decode model ...')
    print('warmup decode model ...')
    decode_inputs_list = get_warmup_inputs(seq_length=1, full_model=False)
    decode_lite_inputs = [mslite.Tensor(item) for item in decode_inputs_list]
    for item in decode_lite_inputs:
        print("decode item ", item.shape, item.dtype)
    work_agent.decode.predict(decode_lite_inputs)
    print('warmup decode model finish')


def start_agent_socket_server(config, startup_queue):
    logging.basicConfig(level=logging.DEBUG,
                        filename=f"./output/agent_{config.rank_id}.log",
                        filemode='w',
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    """启动agent进程, 由_agent_process进行调用, 创建agent进程"""
    work_agent = WorkAgent(config)

    parent_process = psutil.Process(os.getppid())
    print(config.agent_address)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(config.agent_address)
    server.listen(5)

    startup_queue.put(config.rank_id)

    # 绑定method
    print("start agent socket server in rank{}".format(config.rank_id), flush=True)

    while True:
        if not parent_process.is_running():
            logging.warning(f"detect parent pid={parent_process.pid} has exited, child begin to exit")
            server.close()
            return

        conn, client_addr = server.accept()
        # todo workagent = WorkAgent(config)
        while True:
            if not parent_process.is_running():
                logging.warning(
                    f"detect parent pid={parent_process.pid} has exited, child begin to exit")
                server.close()
                return
            try:
                data = conn.recv(4096)
                if not data:
                    break
                data = data.decode()
                logging.info(f"data received is {data}")
                # worker 和 agent建联
                if data.startswith('#'):
                    if work_agent.status & AgentStatus.unconnected == AgentStatus.unconnected:
                        data = data[1:]
                        work_agent.shm_names = data.split(",")
                        work_agent.status = AgentStatus.connected
                        print("send succes")
                        conn.sendall("succes".encode())
                    else:
                        print("send failed")
                        conn.sendall("failed".encode())
                elif data.startswith('*'):
                    # 全量推理
                    work_agent.is_prefill = True
                    data = data[1:]
                    shape_strs = data.split(",")
                    input_shapes = []
                    for shape_str in shape_strs:
                        shape = list(map(int, shape_str.split(" ")))
                        input_shapes.append(shape)
                    _, _ = work_agent.predict(shape_list=input_shapes)
                    conn.sendall("1".encode())
                elif data.startswith('a'):
                    # 增量推理
                    decode_data = data.split('_')
                    current_batch_dyn = int(decode_data[-2])
                    batch_valid_flag = []
                    for ele in decode_data[-1].split(" "):
                        batch_valid_flag.append(int(ele))
                    logging.debug("batch valid flag received is {}".format(batch_valid_flag))
                    work_agent.is_prefill = False
                    _, _ = work_agent.predict(current_batch=current_batch_dyn, batch_valid_flag=batch_valid_flag)
                    conn.sendall("1".encode())
                elif data.startswith('e'):
                    # worker退出获取agent状态，free状态下才允许退出
                    if work_agent.status & AgentStatus.busy == AgentStatus.busy:
                        print("busy")
                        conn.sendall("busy".encode())
                    else:
                        work_agent.status = AgentStatus.unconnected
                        print("free")
                        conn.sendall("free".encode())
            except ConnectionResetError:
                break
        conn.close()


def handler(sig_num, addition):
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


def startup_agents(config_file,
                   config_inc_file,
                   config_post_sampling,
                   rank_size,
                   model0_paths,
                   model1_paths,
                   post_sampling_model_path,
                   post_sampling_model_path2,
                   startup_queue):
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    agent_ports = AgentConfig.AgentPorts
    subprocess_list = []
    log_dir = os.path.join(os.getcwd(), "output")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    for i in range(rank_size):
        agent_address = (AgentIP, agent_ports[i])
        config = Config(i + AgentConfig.device_start, i, config_file, config_inc_file, config_post_sampling,
                        model0_paths[i], model1_paths[i], post_sampling_model_path[0], post_sampling_model_path2[0],
                        agent_address, i)
        p = Process(target=start_agent_socket_server, args=(config, startup_queue))
        p.start()
        subprocess_list.append(p)
    listen_agents_after_startup(subprocess_list)
