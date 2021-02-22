# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Serving, distributed worker agent startup"""

import os
import time
import sys
import traceback
from multiprocessing import Process, Pipe

from mindspore_serving._mindspore_serving import ExitSignalHandle_
from mindspore_serving._mindspore_serving import WorkerAgent_, AgentStartUpConfig_

from mindspore_serving import log as logger
from mindspore_serving.worker import check_type
from mindspore_serving.worker.distributed import worker_agent


def _get_local_ip(rank_list, port):
    """Get the local ip from the rank table config"""
    import socket
    ip_list = set()
    for item in rank_list:
        ip_list.add(item.ip)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        for ip in ip_list:
            try:
                s.bind((ip, port))
                logger.info(f"Get local machine ip success, ip {ip}")
                return ip
            # pylint: disable=bare-except
            except:
                pass
    raise RuntimeError(f"Get local machine ip failed, rank table ips: {ip_list}, bind port {port}")


def _update_model_files_path(model_files, group_config_files):
    """Check and return model files or group config files"""
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    logger.info(f"input model files: {model_files}")
    logger.info(f"input group config files: {group_config_files}")
    model_files_temp = []
    for item in model_files:
        file_name = os.path.join(script_dir, item)
        if not os.access(file_name, os.R_OK):
            raise RuntimeError(f"Cannot access model file '{file_name}'")
        model_files_temp.append(file_name)

    group_files_temp = []
    for item in group_config_files:
        file_name = os.path.join(script_dir, item)
        if not os.access(file_name, os.R_OK):
            raise RuntimeError(f"Cannot access group config file '{file_name}'")
        group_files_temp.append(file_name)

    logger.info(f"absolute model files: {model_files_temp}")
    logger.info(f"absolute group config files: {group_files_temp}")
    return model_files_temp, group_files_temp


def _make_json_table_file(distributed_config):
    """Make rank table json file"""
    rank_size = len(distributed_config.rank_list)
    runtime_dir = os.path.abspath(".")
    time_stamp = str(time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())))
    rank_table_file_name = os.path.join(runtime_dir, f"hccl_rank_table_{time_stamp}_{rank_size}p.json")
    with open(rank_table_file_name, "w") as fp:
        fp.write(distributed_config.rank_table_content)
    return rank_table_file_name


signal_success = "Success"
signal_exit = "Exit"
signal_heartbeat = "HeartBeat"


def _recv_parent(index, recv_pipe, handle_stop_signal=True):
    """Receive message from Start up process.
    Return False on Ctrl+C(and worker Stop message) Exit Signal, heartbeat failed, and signal_exit.
    Return True on receiving signal_success."""
    try:
        while True:
            heartbeat_count = 0
            while not recv_pipe.poll(0.1):
                if handle_stop_signal and ExitSignalHandle_.has_stopped():
                    logger.warning(f"Child {index}: Exit on Ctrl+C or stop message from worker")
                    return False
                heartbeat_count += 1
                if heartbeat_count >= 30:  # 3s
                    logger.warning(f"Child {index}: Exit on failure of receiving parent message")
                    return False
            parent_signal = recv_pipe.recv()
            if parent_signal != signal_heartbeat:
                break
        if parent_signal == signal_success:
            logger.info(f"Child {index}: Receive success")
            return True
        if parent_signal == signal_exit:
            logger.warning(f"Child {index}: Exit on receiving exit message")
        else:
            logger.warning(f"Child {index}: Exit on receiving unknown message {parent_signal}")
    # pylint: disable=broad-except
    except Exception as e:
        logger.warning(f"Child {index}: Exit on exception: {e}")
    return False


def _agent_process(send_pipe, recv_pipe, index, start_config):
    """Agent process"""
    try:
        # listening success or failed message from parent process
        ExitSignalHandle_.start()  # Set flag to running and receive Ctrl+C message
        worker_agent.start_worker_agent(start_config=start_config)
        send_pipe.send((index, signal_success))
        success_msg = _recv_parent(index, recv_pipe)
        if not success_msg:
            worker_agent.stop()
        send_pipe.close()
        recv_pipe.close()
    # pylint: disable=broad-except
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Child {index}: Catch exception and notify exit of others")
        send_pipe.send((index, e))
        _recv_parent(index, recv_pipe, False)
        logger.error(f"Child {index}: end send message to parent")


def _start_listening_child_processes(p_recv_pipe, send_pipe_list, subprocess_list):
    """Listening child process"""

    def send_pipe_msg(send_pipe, msg):
        try:
            send_pipe.send(msg)
        # pylint: disable=broad-except
        except Exception as e:
            logger.warning(f"Send pipe message exception happen: {e}")

    def send_exit_msg():
        index = 0
        for send_pipe, process in zip(send_pipe_list, subprocess_list):
            if process.is_alive():
                logger.warning(f"Send exit message to Child {index}")
                send_pipe_msg(send_pipe, signal_exit)
                logger.warning(f"End send exit message to Child {index}")
            else:
                logger.warning(f"Child {index} is not alive")
            index += 1

    count = len(send_pipe_list)
    for _ in range(count):
        while True:
            if p_recv_pipe.poll(0.1):
                break
            for send_pipe, process in zip(send_pipe_list, subprocess_list):
                if process.is_alive():
                    continue
                logger.warning("Fail to start agents because of death of one agent")
                send_exit_msg()
                return False
            for send_pipe in send_pipe_list:
                send_pipe_msg(send_pipe, signal_heartbeat)

        index, msg = p_recv_pipe.recv()
        logger.info(f"Receive msg from Child {index}: {msg}")
        if isinstance(msg, Exception):
            logger.warning("Fail to start agents because of exception raise by one agent")
            send_exit_msg()
            return False

    for send_pipe in send_pipe_list:
        send_pipe_msg(send_pipe, signal_success)
    logger.info("Success to start agents")
    return True


def _startup_all_agents(common_meta, worker_ip, worker_port,
                        agent_ip, agent_start_port, device_id_list, rank_id_list,
                        model_files, group_config_files, rank_table_file):
    """Start up all agents in one machine"""
    servable_name = common_meta.servable_name
    index = 0
    send_pipe_list = []
    subprocess_list = []
    c_send_pipe, p_recv_pipe = Pipe()
    for device_id, rank_id, model_file, group_file in zip(device_id_list, rank_id_list, model_files,
                                                          group_config_files):
        p_send_pipe, c_recv_pipe = Pipe()
        send_pipe_list.append(p_send_pipe)

        agent_port = agent_start_port + index

        start_config = AgentStartUpConfig_()
        start_config.rank_id = rank_id
        start_config.device_id = device_id
        start_config.model_file_name = model_file
        start_config.group_file_name = group_file
        start_config.rank_table_json_file_name = rank_table_file
        start_config.agent_ip = agent_ip
        start_config.agent_port = agent_port
        start_config.worker_ip = worker_ip
        start_config.worker_port = worker_port
        start_config.common_meta = common_meta

        process = Process(target=_agent_process,
                          args=(c_send_pipe, c_recv_pipe, index, start_config),
                          name=f"{servable_name}_worker_agent_rank{rank_id}_device{device_id}")
        process.start()
        subprocess_list.append(process)
        index += 1
    ret = _start_listening_child_processes(p_recv_pipe, send_pipe_list, subprocess_list)
    if not ret:
        WorkerAgent_.notify_failed(worker_ip, worker_port)


def startup_worker_agents(worker_ip, worker_port, model_files, group_config_files, agent_start_port=7000):
    """Start up all needed worker agents on one machine"""
    check_type.check_str("worker_ip", worker_ip)
    check_type.check_ip_port("worker_port", worker_port)
    check_type.check_int("agent_start_port", agent_start_port, 1, 65535 - 7)
    model_files = check_type.check_and_as_str_tuple_list("model_files", model_files)
    group_config_files = check_type.check_and_as_str_tuple_list("group_config_files", group_config_files)

    ExitSignalHandle_.start()
    distributed_config = WorkerAgent_.get_agents_config_from_worker(worker_ip, worker_port)

    # get machine ip
    rank_list = distributed_config.rank_list
    local_ip = _get_local_ip(rank_list, agent_start_port)
    # get all device_id and rank_id
    local_device_id_list = []
    local_rank_id_list = []
    for rank_id, item in enumerate(rank_list):
        if item.ip == local_ip:
            local_device_id_list.append(item.device_id)
            local_rank_id_list.append(rank_id)

    # handle model files and group config files
    if len(local_device_id_list) != len(model_files):
        raise RuntimeError(f"Card count {local_device_id_list} described rank table does not equal to model files size "
                           f"{len(model_files)}, model files: {model_files}")

    if len(local_device_id_list) != len(group_config_files):
        raise RuntimeError(f"Card count {local_device_id_list} described rank table does not equal to group config "
                           f"files size {len(group_config_files)}, group config files: {group_config_files}")

    model_files, group_config_files = _update_model_files_path(model_files, group_config_files)

    # make json table file and export env
    rank_table_file = _make_json_table_file(distributed_config)
    _startup_all_agents(distributed_config.common_meta, worker_ip, worker_port, local_ip, agent_start_port,
                        local_device_id_list, local_rank_id_list,
                        model_files, group_config_files, rank_table_file)
