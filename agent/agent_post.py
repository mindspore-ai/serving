"""agent"""

import argparse
import logging
import signal
import threading
import time
from concurrent import futures
from multiprocessing import Process

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

shms = []


class Config:
    def __init__(self, device_id, rank_id, config_file, config_inc_file, config_post_sampling,
                 model0_path, model1_path, post_sampling_model_path, agent_address, index,
                 shm):
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
        self.shm = shm


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
    print("load predict model successful")
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
class WorkAgent(agent_pb2_grpc.WorkAgentServicer):
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
        
        # TODO warm up
        self.shm = args.shm

    def Predict(self, request, context) -> np.array:
        parse_time = time.time()

        inputs = []
        # 第一个input_ids为shm
        shm_tensor = request.inputs[0]

        existing_shm = shared_memory.SharedMemory(name=shm_tensor.shm_data.memory_key)
        input_ids = np.ndarray((shm_tensor.shape.dims), dtype=dtype_map_rev[shm_tensor.dtype], buffer=existing_shm.buf)

        inputs.append(input_ids)
        for item in request.inputs[1:]:
            tensor = np.frombuffer(item.data, dtype_map_rev[item.dtype]).reshape(item.shape.dims)
            inputs.append(tensor)

        current_index = inputs[1]
        print('current_index: ', current_index)
        prefill_flag = request.if_prefill
        logging.info('parse time is {}'.format((time.time() - parse_time) * 1000))
        tmp_in = []
        for i in inputs:
            tmp_in.append(i)
        # list[np.array]->list[Tensor]
        # 调用ms lite进行推理
        time_start = time.time()
        model = self.model0 if prefill_flag else self.model1

        reshape_time = time.time()
        input_list = []
        input_list_shape = [input.shape for input in inputs]
        model.resize(model.get_inputs(), input_list_shape)
        for data in inputs:
            if data is None:
                continue
            input_list.append(data)
        lite_inputs = model.get_inputs()
        for input_np, tensor in zip(input_list, lite_inputs):
            print('input_np: ', input_np.shape)
            print('input_np_type: ', input_np.dtype)
            print('tensor: ', tensor.shape)
            tensor.shape = input_np.shape
            print('>>>>>tensor: ', tensor.shape)
            tensor.set_data_from_numpy(input_np)

        logging.info('reshape time is {}'.format((time.time() - reshape_time) * 1000))
        # lite_inputs = [mslite.Tensor(item) for item in inputs]
        predict_time = time.time()
        # outputs -> list tensor
        outputs = model.predict(lite_inputs)
        logging.info('predict_time is {}'.format((time.time() - predict_time) * 1000))    
        logging.info("outputs {}".format(outputs))
        outputs = outputs[0] # [1, 310, 32000] type mslite.tensor
        outputs_np = outputs.get_data_to_numpy()
        print('outputs_np: ', outputs_np.shape)

        if prefill_flag:
            outputs_np = outputs_np.reshape((310, 32000))
            outputs_np = outputs_np[current_index].astype(np.float16)
        else:
            outputs_np = outputs_np.reshape((1, 32000)).astype(np.float16)
        print('outputs_np: ', outputs_np)
        print('>>>>>>>>>>outputs_np: ', outputs_np.shape)
        post_inputs = self.model2.get_inputs()
        print('post inputs: ', len(post_inputs))
        
        post_inputs[0].set_data_from_numpy(outputs_np)
        post_sampling_out = self.model2.predict(post_inputs)
        # 转换为proto格式数据返回
        create_proto_time = time.time()
        outs = post_sampling_out[1].get_data_to_numpy()
        print('>>>>>>>>>>>>>>outs: ', outs)
       
        parg_s = post_sampling_out[2].get_data_to_numpy()
        print('>>>>>>>>>>>>>>post_sampling_outs: ', parg_s)
        final_outputs = np.vstack((outs[0], parg_s[0].astype(np.float16)))
        print('>>>>>>>>>>>>>>final_outputs: ', final_outputs)
        # 只有第一个agent进行output覆写
        if self.index == 0:
            tmp = np.ndarray((final_outputs.shape), dtype=final_outputs.dtype, buffer=existing_shm.buf)
            tmp[:] = final_outputs[:]
        final_outputs = create_proto_reply_new(final_outputs.shape, final_outputs.dtype, existing_shm.name)
        logging.info('create proto time is {}'.format((time.time() - create_proto_time) * 1000))
        return final_outputs


def start_agent_grpc_server(config):
    """启动agent进程, 由_agent_process进行调用, 创建agent进程"""
    parent_process = psutil.Process(os.getppid())
    # 启动grpc服务
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # 绑定method
    print("--------------shm name is {}".format(config.shm.name))
    agent_pb2_grpc.add_WorkAgentServicer_to_server(WorkAgent(config), server)

    print(config.agent_address)
    server.add_insecure_port(config.agent_address)
    server.start()
    print("start agent grpc server in rank{}".format(config.rank_id))
    while True:
        if not parent_process.is_running():
            logging.warning(f"detect parent pid={parent_process.pid} has exited, child begin to exit")
            # TODO stop all agents
            return
        time.sleep(0.1)


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
    tmp = shared_memory.SharedMemory(create=True, size=1024*1024*1024)
    shm_list = []
    for i in range(rank_size):
        shm = shared_memory.SharedMemory(create=True, size=1024*1024*1024)
        shm_list.append(shm)
    for i in range(rank_size):
        agent_address = "localhost:" + str(agent_start_port + i)
        print('shm_list: ', shm_list[i])
        config = Config(i, i, config_file, config_inc_file, config_post_sampling, 
                        model0_paths[i], model1_paths[i], post_sampling_model_path, agent_address, i, shm_list[i])
        p = Process(target=start_agent_grpc_server, args=(config,))
        p.start()
        subprocess_list.append(p)
    listen_agents_after_startup(subprocess_list)
