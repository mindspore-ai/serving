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

import grpc
import mindspore_lite as mslite
import psutil
import os
import numpy as np
import proto.work_agent_pb2 as agent_pb2
import proto.work_agent_pb2_grpc as agent_pb2_grpc
from sub_process.sub_process import listen_agents_after_startup
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
pool = ThreadPoolExecutor(max_workers=20, thread_name_prefix='test_thread')
from worker.worker_to_agent import *
import shared_memory
import logging
import time
from config.serving_config import Baseconfig, AgentConfig, AgentIP
from models.post_sampling.topk import post_sampling, softmax_np

VOCAB_LEN = Baseconfig.vocab_size
PORTS = AgentConfig.AgentPorts

shms = []


class Config:
    def __init__(self, device_id, rank_id, config_file, config_inc_file, config_post_sampling,
                 model0_path, model1_path, post_sampling_model_path, post_sampling_model_path2, agent_address, index):
        print("device_id", device_id)
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

       

def load_model(model0_path, model1_path, config_file, config_inc_file, rank_id, device_id):
    # 加载模型
    context = mslite.Context()
    print('device_id: ', device_id)
    print('rank_id: ', rank_id)
    context.ascend.device_id = device_id
    context.ascend.rank_id = rank_id
    context.ascend.provider = "ge"
    context.target = ["Ascend"]
    # TODO 单模型
    if len(model1_path) == 0:
        model0 = mslite.Model()
        model0.build_from_file(model0_path, mslite.ModelType.MINDIR, context, config_file)
        model1 = None
        return model0, model1

    # rank_table_file放在config_file中
    model0 = mslite.Model()
    model1 = mslite.Model()
    model_group = mslite.ModelGroup(mslite.ModelGroupFlag.SHARE_WEIGHT)
    model_group.add_model([model0, model1])
    model0.build_from_file(model0_path, mslite.ModelType.MINDIR, context, config_file)
    model1.build_from_file(model1_path, mslite.ModelType.MINDIR, context, config_inc_file)
    print("load model successful")
    return model0, model1


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
        
        self.prefill, self.decode = load_model(model0_path, model1_path, config_file, config_inc_file, rank_id, device_id)
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
        self.top_p_list = None
        self.top_k_list = None
        self.temperature_list = None
        self.repetition_penalty = None
        self.input_length = None

    def _post_smapling_argmax_npu(self, outputs_np):
        
        post_inputs = self.argmax_model.get_inputs()
        post_inputs[0].set_data_from_numpy(outputs_np)
        post_sampling_out = self.argmax_model.predict(post_inputs)
        self.target = post_sampling_out[0].get_data_to_numpy().astype(np.int32)

    def _post_sampling_topk_npu(self, outputs_np):
        post_inputs = self.topk_model.get_inputs()
        post_inputs[0].set_data_from_numpy(outputs_np)
        post_inputs[1].set_data_from_numpy(self.temperature_list)
        post_sampling_out = self.topk_model.predict(post_inputs)
        outs = post_sampling_out[0].get_data_to_numpy().astype(np.float16)
        p_args = post_sampling_out[1].get_data_to_numpy()
        topp = self.top_p_list[0]
        top_k_num = self.top_k_list[0]
        if topp < 1.0:
            top_p_num = sum(outs[0] > topp)
            if top_p_num == 0:
                top_p_num = 100
            outs = outs[0][:top_p_num]
            p_args = p_args[0][:top_p_num]
            if np.sum(outs) == 0:
                outs = np.array([1 / top_p_num for _ in range(top_p_num)])
            p = softmax_np(outs)
        else:
            p = outs[0]
            p_args = p_args[0]
        target_index = np.random.choice(len(p), p=p)
        self.target = np.array([p_args[target_index]]).astype(np.int32)

    def _post_sampling_topk_host(self, outputs_np):
        outputs_np = outputs_np.reshape(VOCAB_LEN)
        all_task = [pool.submit(post_sampling, np.array(item), self.top_p_list[0],
                                self.top_k_list[0]) for item in [outputs_np]]
        target = []
        for x in as_completed(all_task):
            res = x.result()
            target.append(res)
        self.target = np.array(target).astype(np.int32)

    def do_post_sampling(self, outputs_np, existing_shm0):
        # DO Argmax NPU
        if int(self.post_mode_list[0]) == 0 or self.top_k_list[0] == 1:
            self._post_smapling_argmax_npu(outputs_np)
        else:

            self._post_sampling_topk_npu(outputs_np)
        """
        TODO: topK后处理入图
        elif int(self.post_mode_list[0]) == 0 and self.top_k_list[0] == 100:
            print('_post_smapling_topk_npu')
            self._post_sampling_topk_npu(outputs_np)
        """
        if self.index == 0:
            tmp = np.ndarray((self.target.shape), dtype=self.target.dtype, buffer=existing_shm0.buf)
            tmp[:] = self.target[:]

    def predict(self, shape_list=None):
        tmp_shms = []
        start_time = time.time()
        existing_shm0 = shared_memory.SharedMemory(name=self.shm_names[0])
        tmp_shms.append(existing_shm0)
        self.target = np.ndarray((1,), dtype=np.int32, buffer=existing_shm0.buf)
        input_ids = self.target

        gen_parms_id = 4
        gen_parms_shm = shared_memory.SharedMemory(name=self.shm_names[gen_parms_id])
        tmp_shms.append(gen_parms_shm)
        if self.is_prefill:
        # 从共享内存中读取第一个array
            first_group = np.ndarray((shape_list[0]), dtype=np.int32, buffer=existing_shm0.buf)
            current_index_ = first_group[:, shape_list[0][1] -3 :shape_list[0][1] - 2]
            self.current_index = np.squeeze(current_index_, axis=-1)
            valid_length_ = first_group[:, shape_list[0][1] - 1: shape_list[0][1]]
            self.valid_length = np.squeeze(valid_length_, axis=-1)
            self.init_reset = np.array([False], dtype=np.bool)
            input_ids = first_group[:, :shape_list[0][1] - 3]
            self.input_lentgth = input_ids.shape[1]
            gen_parms_id = 4
            shape_parms = shape_list[gen_parms_id]


            gen_parms = np.ndarray((shape_parms), dtype=np.float16, buffer=gen_parms_shm.buf)
            # post_mode_np, top_p_np, top_k_np, temperature_np
            self.post_mode_list = gen_parms[:, 0].astype(np.int32)
            # post_mode_np = np.squeeze(post_mode_np, axis=-1).astype(np.int32)
            self.top_p_list = gen_parms[:, 1]
            # top_p_list = np.squeeze(top_p_list, axis=-1)
            self.top_k_list = gen_parms[:, 2].astype(np.int32)
            # top_k_list = np.squeeze(top_k_list, axis=-1).astype(np.int32)
            self.temperature_list = gen_parms[:, 3]
            self.repetition_penalty = gen_parms[:, 4]
        else:
            self.current_index = self.current_index + 1
            self.valid_length = self.valid_length + 1
            self.init_reset = np.array([True], dtype=np.bool)

        tmp_in = [input_ids, self.current_index, self.init_reset, self.valid_length]
        if self.is_prefill:
            for i in range(1, len(shape_list) - 1):
                    existing_shm = shared_memory.SharedMemory(name=self.shm_names[i])
                    tmp_shms.append(existing_shm)
                    
                    if i == 1:
                        tmp_in.append(np.ndarray((shape_list[i]), dtype=np.float16, buffer=existing_shm.buf))
                    else:
                        tmp_in.append(np.ndarray((shape_list[i]), dtype=np.float32, buffer=existing_shm.buf))
       
        # 调用ms lite进行推理
        model = self.prefill if self.is_prefill else self.decode
        lite_inputs = [mslite.Tensor(item) for item in tmp_in]
        logging.info('agent pre-process time is {}'.format((time.time() - start_time) * 1000))

        predict_time = time.time()
        outputs = model.predict(lite_inputs)
        logging.info('predict_time is {}'.format((time.time() - predict_time) * 1000))

        outputs = outputs[0]  # [1, 310, 32000] type mslite.tensor
        outputs_np = outputs.get_data_to_numpy()
        output_lenght = outputs_np.shape[1]
        post_time = time.time()
        if self.is_prefill:
            outputs_np = outputs_np.reshape((output_lenght, VOCAB_LEN))
            outputs_np = outputs_np[self.current_index[0]].astype(np.float16)
            outputs_np = outputs_np.reshape((1, -1))
        else:
            outputs_np = outputs_np.reshape((output_lenght, VOCAB_LEN)).astype(np.float16)
        # ToDo: multi-batch adapter
        if self.rank_id == 0:
            self.do_post_sampling(outputs_np, existing_shm0)
        logging.info('post_time is {}'.format((time.time() - post_time) * 1000))
        logging.info('npu_total_time is {}'.format((time.time() - start_time) * 1000))
        return self.target, tmp_shms


def start_agent_socket_server(config):
    """启动agent进程, 由_agent_process进行调用, 创建agent进程"""
    parent_process = psutil.Process(os.getppid())
    print(config.agent_address)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(config.agent_address)
    server.listen(5)
    work_agent = WorkAgent(config)
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
                # #name1,name2  第一次
                # todo 判断第一次
                if data.startswith('#'):
                    data = data[1:]
                    work_agent.shm_names = data.split(",")
                    continue
                if data.startswith('*'):
                    work_agent.is_prefill = True
                    data = data[1:]
                    shape_strs = data.split(",")
                    input_shapes = []
                    for shape_str in shape_strs:
                        shape = list(map(int, shape_str.split(" ")))
                        input_shapes.append(shape)
                    output_shape, shm = work_agent.predict(input_shapes)

                    conn.sendall("1".encode())
                elif data.startswith('a'):
                    work_agent.is_prefill = False
                    output_shape, shm = work_agent.predict()
                    conn.sendall("1".encode())
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
                   agent_start_port=PORTS[0]):
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    subprocess_list = []
    for i in range(rank_size):
        logging.basicConfig(level=logging.DEBUG,
                    filename=f"./output/agent_{i}.log",
                    filemode='w',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        agent_address = (AgentIP, agent_start_port + i)
        config = Config(i + AgentConfig.device_start, i, config_file, config_inc_file, config_post_sampling,
                        model0_paths[i], model1_paths[i], post_sampling_model_path[0], post_sampling_model_path2[0], agent_address, i)
        p = Process(target=start_agent_socket_server, args=(config,))
        p.start()
        subprocess_list.append(p)
    listen_agents_after_startup(subprocess_list)
