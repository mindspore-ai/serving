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
from worker.worker_to_agent import *
import shared_memory
import logging
import time
from config.serving_config import Baseconfig

shms = []


class Config:
    def __init__(self, device_id, rank_id, config_file, config_inc_file, config_post_sampling,
                 model0_path, model1_path, post_sampling_model_path, agent_address, index):
        print("device_id",device_id)
        self.device_id = device_id
        self.rank_id = rank_id
        self.config_file = config_file
        self.config_inc_file = config_inc_file
        self.config_post_sampling = config_post_sampling
        self.model0_path = model0_path
        self.model1_path = model1_path
        self.post_sampling_model_path = post_sampling_model_path
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
    print('model path:' , model_path)
    model.build_from_file(model_path[0], mslite.ModelType.MINDIR, context, config_file)
    print("load post-sampling model successful")
    return model

"""
work_agent.proto实现, 供worker调用
"""
class WorkAgent:
    def __init__(self, args):
        device_id = args.device_id
        rank_id = args.rank_id
        config_file = args.config_file
        config_inc_file = args.config_inc_file
        config_post_sampling = args.config_post_sampling
        model0_path = args.model0_path
        model1_path = args.model1_path
        model2_path = args.post_sampling_model_path
        self.index = args.index
        self.model0, self.model1 = load_model(model0_path, model1_path, config_file, config_inc_file, rank_id, device_id)
        self.model2 = load_post_model(model2_path, config_post_sampling, rank_id, device_id)
        self.shm_names = []
        self.init_reset = None
        self.current_index = None
        self.valid_length = None
        self.tensor_shape = None
        self.pre_input_ids = None
        self.is_prefill = True
        self.target = None

    def _respahe_tensor(self, tmp_in, model):
        input_list = []
        input_list_shape = [input.shape for input in tmp_in]
        model.resize(model.get_inputs(), input_list_shape)
        for data in tmp_in:
            if data is None:
                continue
            input_list.append(data)
        lite_inputs = model.get_inputs()
        for input_np, tensor in zip(input_list, lite_inputs):
            tensor.shape = input_np.shape
            self.tensor_shape = input_np.shape
            tensor.set_data_from_numpy(input_np)
        return lite_inputs

    def predict_backup_post_in_npu(self, shape_list):
        tmp_shms = []
        start_time = time.time()

        # 从共享内存中读取第一个array
        existing_shm0 = shared_memory.SharedMemory(name=self.shm_names[0])
        first_group = np.ndarray((shape_list[0]), dtype=np.int32, buffer=existing_shm0.buf)
        tmp_shms.append(existing_shm0)
        # 对第一个array进行切分 -> input_ids, current_index, valid_length, init_reset
        input_ids = first_group[:, :shape_list[0][1] - 3]
        logging.info('first_group time is {}'.format((time.time() - start_time) * 1000))

        current_index = first_group[:, shape_list[0][1] -3 :shape_list[0][1] - 2]
        current_index = np.squeeze(current_index, axis=-1)
        logging.info('current_index time is {}'.format((time.time() - start_time) * 1000))
        # todo : current_index 递增，只在第一次取
        init_reset = first_group[:, shape_list[0][1] - 2 : shape_list[0][1] - 1]
        # todo :init_reset 首次过来才需要，增量不需要（风险）
        valid_length = first_group[:, shape_list[0][1] - 1: shape_list[0][1]]
        valid_length = np.squeeze(valid_length, axis=-1)
        logging.info('valid time is {}'.format((time.time() - start_time) * 1000))
        if init_reset[0][0] == 0:
            prefill_flag = True
        else:
            prefill_flag = False
        init_reset = np.array([False], dtype=np.bool) if prefill_flag else np.array([True], dtype=np.bool)

        logging.info('init_reset time is {}'.format((time.time() - start_time) * 1000))
        print('input_ids: ', input_ids)
        tmp_in = [input_ids, current_index, init_reset, valid_length]
        # 如果是prefill ，读取mask, freq_cos, freq_sin
        if prefill_flag:
            for i in range(1, len(shape_list)):
                existing_shm = shared_memory.SharedMemory(name=self.shm_names[i])
                tmp_shms.append(existing_shm)
                if i == 1:
                    tmp_in.append(np.ndarray((shape_list[i]), dtype=np.float16, buffer=existing_shm.buf))
                else:
                    tmp_in.append(np.ndarray((shape_list[i]), dtype=np.float32, buffer=existing_shm.buf))
        

        # 调用ms lite进行推理
        model = self.model0 if prefill_flag else self.model1

        reshape_time = time.time()
        input_list = []
        input_list_shape = [input.shape for input in tmp_in]
        
        model.resize(model.get_inputs(), input_list_shape)
        # todo: 只在首次的时候，进行model.resize()
        logging.info('model resize time is {}'.format((time.time() - start_time) * 1000))
        
        for data in tmp_in:
            if data is None:
                continue
            input_list.append(data)
        lite_inputs = model.get_inputs()
        for input_np, tensor in zip(input_list, lite_inputs):
            print('input_np: ', input_np.shape)
            print('input_np_type: ', input_np.dtype)
            print('tensor: ', tensor.shape, tensor.dtype)
            tensor.shape = input_np.shape
            print('>>>>>tensor: ', tensor.shape, tensor.dtype)
            
            tensor.set_data_from_numpy(input_np)
        # ToDo: reshape 首次执行，结果存在成员变量 1ms

        logging.info('reshape time is {}'.format((time.time() - start_time) * 1000))
        predict_time = time.time()
        outputs = model.predict(lite_inputs)
        logging.info('predict_time is {}'.format((time.time() - start_time) * 1000))
        
        outputs = outputs[0] # [1, 310, 32000] type mslite.tensor
        outputs_np = outputs.get_data_to_numpy()

        if prefill_flag:
            outputs_np = outputs_np.reshape((310, 32000))
            outputs_np = outputs_np[self.current_index[0]].astype(np.float16)
        else:
            outputs_np = outputs_np.reshape((1, 32000)).astype(np.float16)
        post_inputs = self.model2.get_inputs()
        
        post_inputs[0].set_data_from_numpy(outputs_np)
        logging.info('pre_post_time is {}'.format((time.time() - start_time) * 1000))
        post_sampling_out = self.model2.predict(post_inputs)
        logging.info('post_time is {}'.format((time.time() - start_time) * 1000))
        # ToDO: 后处理CPU上执行
        outs = post_sampling_out[1].get_data_to_numpy()
        logging.info('out_time is {}'.format((time.time() - start_time) * 1000))
       
        parg_s = post_sampling_out[2].get_data_to_numpy()
        logging.info('parg_s_time is {}'.format((time.time() - start_time) * 1000))
        final_outputs = np.vstack((outs[0], parg_s[0].astype(np.float16)))
        logging.info('final_outputs_time is {}'.format((time.time() - start_time) * 1000))
        # 只有第一个agent进行output覆写
        if self.index == 0:
            tmp = np.ndarray((final_outputs.shape), dtype=final_outputs.dtype, buffer=existing_shm0.buf)
            tmp[:] = final_outputs[:]
        # 返回outputs的shape
        logging.info('npu_total_time is {}'.format((time.time() - start_time) * 1000))
        return final_outputs.shape, tmp_shms

    def predict(self, shape_list=None):
        tmp_shms = []
        start_time = time.time()
        input_ids = self.target
        existing_shm0 = shared_memory.SharedMemory(name=self.shm_names[0])
        tmp_shms.append(existing_shm0)
        if self.is_prefill:
        # 从共享内存中读取第一个array
            first_group = np.ndarray((shape_list[0]), dtype=np.int32, buffer=existing_shm0.buf)
            init_reset = first_group[:, shape_list[0][1] - 2 : shape_list[0][1] - 1]
            current_index_ = first_group[:, shape_list[0][1] -3 :shape_list[0][1] - 2]
            self.current_index = np.squeeze(current_index_, axis=-1)
            valid_length_ = first_group[:, shape_list[0][1] - 1: shape_list[0][1]]
            self.valid_length = np.squeeze(valid_length_, axis=-1)
            self.init_reset = np.array([False], dtype=np.bool)
            input_ids = first_group[:, :shape_list[0][1] - 3]
        else:
            self.current_index = self.current_index + 1
            self.valid_length = self.valid_length + 1
            self.init_reset = np.array([True], dtype=np.bool)
        tmp_in = [input_ids, self.current_index, self.init_reset, self.valid_length]
        if self.is_prefill:
            for i in range(1, len(shape_list)):
                    existing_shm = shared_memory.SharedMemory(name=self.shm_names[i])
                    tmp_shms.append(existing_shm)
                    if i == 1: 
                        tmp_in.append(np.ndarray((shape_list[i]), dtype=np.float16, buffer=existing_shm.buf))
                    else:
                        tmp_in.append(np.ndarray((shape_list[i]), dtype=np.float32, buffer=existing_shm.buf))

        # 调用ms lite进行推理
        model = self.model0 if self.is_prefill else self.model1
        reshape_time = time.time()
        lite_inputs = [mslite.Tensor(item) for item in tmp_in]

        # todo: 只在首次的时候，进行model.resize()
        logging.info('agent pre-process time is {}'.format((time.time() - start_time) * 1000))

        predict_time = time.time()
        outputs = model.predict(lite_inputs)
        logging.info('predict_time is {}'.format((time.time() - start_time) * 1000))
        
        outputs = outputs[0] # [1, 310, 32000] type mslite.tensor
        outputs_np = outputs.get_data_to_numpy()

        if self.is_prefill:
            outputs_np = outputs_np.reshape((310, 32000))
            outputs_np = outputs_np[self.current_index[0]].astype(np.float16)
            outputs_np = outputs_np.reshape((1,-1))#outputs_np[self.current_index[0]].astype(np.float16)
        else:
            outputs_np = outputs_np.reshape((1, 32000)).astype(np.float16)
        post_inputs = self.model2.get_inputs()
        
        post_inputs[0].set_data_from_numpy(outputs_np)
        post_sampling_out = self.model2.predict(post_inputs)
        logging.info('post_time is {}'.format((time.time() - start_time) * 1000))
        outs = post_sampling_out[1].get_data_to_numpy()
       
        parg_s = post_sampling_out[2].get_data_to_numpy()
        p = outs[0] / sum(outs[0])
        target_index = np.random.choice(len(p), p=p)
        target = np.array([parg_s[0][target_index]]).astype(np.int32)
        self.target = target

        if self.index == 0:
            tmp = np.ndarray((target.shape), dtype=target.dtype, buffer=existing_shm0.buf)
            tmp[:] = target[:]
        logging.info('npu_total_time is {}'.format((time.time() - start_time) * 1000))
        return target, tmp_shms
      

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
    print("start agent socket server in rank{}".format(config.rank_id))

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
                    # #name1,name2  第一次
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
                   agent_start_port=7000):
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    subprocess_list = []

    for i in range(rank_size):
        logging.basicConfig(level=logging.DEBUG,
                    filename=f"./output/agent_{i}.log",
                    filemode='w',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        agent_address = ('localhost', agent_start_port + i)
        config = Config(i + 4, i, config_file, config_inc_file, config_post_sampling,
                        model0_paths[i], model1_paths[i], post_sampling_model_path, agent_address, i)
        p = Process(target=start_agent_socket_server, args=(config,))
        p.start()
        subprocess_list.append(p)
    listen_agents_after_startup(subprocess_list)
