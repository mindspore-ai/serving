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
import signal
from multiprocessing import Process, Pipe
import threading
import psutil

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
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        for ip in ip_list:
            try:
                s.bind((ip, port))
                logger.info(f"Get local machine ip success, ip {ip}")
                return ip
            # pylint: disable=bare-except
            except:
                pass
    raise RuntimeError(f"Get local machine ip failed, rank table ips: {ip_list}, bind port {port}")


def _check_local_ip(agent_ip, port):
    """Check the local ip"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        for i in range(8):
            try:
                s.bind((agent_ip, port + i))
                logger.info(f"Check local machine ip success, ip {agent_ip}")
                return True
            # pylint: disable=bare-except
            except:
                pass
    return False


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

    if group_config_files is not None:
        group_files_temp = []
        for item in group_config_files:
            file_name = os.path.join(script_dir, item)
            if not os.access(file_name, os.R_OK):
                raise RuntimeError(f"Cannot access group config file '{file_name}'")
            group_files_temp.append(file_name)
    else:
        group_files_temp = None
    logger.info(f"absolute model files: {model_files_temp}")
    logger.info(f"absolute group config files: {group_files_temp}")
    return model_files_temp, group_files_temp


def _make_json_table_file(distributed_config):
    """Make rank table json file"""
    rank_size = len(distributed_config.rank_list)
    runtime_dir = os.path.abspath(".")
    time_stamp = str(time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())))
    rank_table_dir = os.path.join(runtime_dir, "temp_rank_table")
    try:
        os.mkdir(rank_table_dir)
    except FileExistsError:
        pass
    rank_table_file_name = os.path.join(rank_table_dir, f"hccl_rank_table_{time_stamp}_{rank_size}p.json")
    with open(rank_table_file_name, "w") as fp:
        fp.write(distributed_config.rank_table_content)
    return rank_table_file_name


signal_success = "Success"
signal_exit = "Exit"


def _recv_parent(parent_process, index, recv_pipe, handle_stop_signal=True):
    """Receive message from Start up process.
    Return False on Ctrl+C(and worker Stop message) Exit Signal, heartbeat failed, and signal_exit.
    Return True on receiving signal_success."""
    try:
        while True:
            while not recv_pipe.poll(0.1):
                if handle_stop_signal and ExitSignalHandle_.has_stopped():
                    logger.warning(f"Child {index}: Exit on Ctrl+C or stop message from worker")
                    return False
                if not parent_process.is_running():  # 3s
                    logger.warning(f"Child {index}: Exit on failure of exit of parent process")
                    return False
            parent_signal = recv_pipe.recv()
            break
        if parent_signal == signal_success:
            logger.info(f"Child {index}: Receive success")
            return True
        if parent_signal == signal_exit:
            logger.warning(f"Child {index}: Exit on receiving exit message")
    # pylint: disable=broad-except
    except Exception as e:
        logger.warning(f"Child {index}: Exit on exception: {e}")
    return False


def _agent_process(send_pipe, recv_pipe, index, start_config):
    """Agent process"""
    parent_process = psutil.Process(os.getppid())
    try:
        # listening success or failed message from parent process
        worker_agent.start_worker_agent(start_config=start_config)
        send_pipe.send((index, signal_success))
        success_msg = _recv_parent(parent_process, index, recv_pipe)
        if not success_msg:
            worker_agent.stop()
        send_pipe.close()
        recv_pipe.close()
        while not ExitSignalHandle_.has_stopped():
            if not parent_process.is_running():
                logger.warning(f"Child {index}, detect parent pid={parent_process.pid} has exited, child begin to exit")
                worker_agent.stop()
                return
            time.sleep(0.1)
    # pylint: disable=broad-except
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Child {index}: Catch exception and notify exit of others")
        send_pipe.send((index, e))
        _recv_parent(parent_process, index, recv_pipe, False)
        logger.error(f"Child {index}: end send message to parent")


def _send_pipe_msg(send_pipe, msg):
    """Send pipe message"""
    try:
        send_pipe.send(msg)
    # pylint: disable=broad-except
    except Exception as e:
        logger.warning(f"Send pipe message exception happen: {e}")


def _send_exit_signal_to_children(subprocess_list):
    """Send exit signal to all child processes, and terminate all child processes when they are still alive
    in some seconds later"""

    def wait_exit(wait_seconds, msg):
        for i in range(wait_seconds):
            all_exit = True
            for process in subprocess_list:
                if process.is_alive():
                    logger.warning(f"There are still child processes that have not exited and {msg} in "
                                   f"{wait_seconds - i} seconds.")
                    time.sleep(1)
                    all_exit = False
                    break
            if all_exit:
                logger.info(f"All Child process exited")
                return True
        return False

    if wait_exit(3, "SIGINT will be sent"):
        return
    # Send signal SIGINT
    for index, process in enumerate(subprocess_list):
        if process.is_alive():
            logger.warning(f"Send signal SIGINT to {index}")
            try:
                child_process = psutil.Process(process.pid)
                children_of_child = child_process.children(recursive=True)
                for item in children_of_child:
                    os.kill(item.pid, signal.SIGINT)
            # pylint: disable=broad-except
            except Exception as e:
                logger.warning(f"Get exception when send signal SIGINT to children of child {index}, exception: {e}")
            os.kill(process.pid, signal.SIGINT)

    if wait_exit(10, "will be forcibly killed"):
        return

    for index, process in enumerate(subprocess_list):
        if process.is_alive():
            logger.warning(f"Kill Child process {index}")
            try:
                child_process = psutil.Process(process.pid)
                children_of_child = child_process.children(recursive=True)
                for item in children_of_child:
                    os.kill(item.pid, signal.SIGKILL)
            # pylint: disable=broad-except
            except Exception as e:
                logger.warning(f"Get exception when send signal SIGKILL to children of child {index}, exception: {e}")
            os.kill(process.pid, signal.SIGKILL)


def _send_exit_msg_to_children(send_pipe_list, subprocess_list):
    """Send exit msg to all child processes, and terminate all child processes when they are still alive
    in some seconds later"""
    index = 0
    for send_pipe, process in zip(send_pipe_list, subprocess_list):
        if process.is_alive():
            logger.warning(f"Send exit message to Child {index}")
            _send_pipe_msg(send_pipe, signal_exit)
            logger.warning(f"End send exit message to Child {index}")
        else:
            logger.warning(f"Child {index} is not alive")
        index += 1
    _send_exit_signal_to_children(subprocess_list)


def _listening_agents_when_startup(p_recv_pipe, send_pipe_list, subprocess_list):
    """Listening child process"""
    count = len(send_pipe_list)
    for _ in range(count):
        while True:
            if p_recv_pipe.poll(0.1):
                break
            if ExitSignalHandle_.has_stopped():
                logger.warning("Fail to start agents because of Ctrl+C")
                _send_exit_msg_to_children(send_pipe_list, subprocess_list)
                return False
            for send_pipe, process in zip(send_pipe_list, subprocess_list):
                if process.is_alive():
                    continue
                logger.warning("Fail to start agents because of death of one agent")
                _send_exit_msg_to_children(send_pipe_list, subprocess_list)
                return False

        index, msg = p_recv_pipe.recv()
        logger.info(f"Receive msg from Child {index}: {msg}")
        if isinstance(msg, Exception):
            logger.warning("Fail to start agents because of exception raise by one agent")
            _send_exit_msg_to_children(send_pipe_list, subprocess_list)
            return False

    for send_pipe in send_pipe_list:
        _send_pipe_msg(send_pipe, signal_success)
    return True


def _listening_agents_after_startup(subprocess_list, worker_ip, worker_port, agent_ip):
    """Listening agent status after success start up of agents"""

    def wait_child_exit():
        while not ExitSignalHandle_.has_stopped():
            for index, process in enumerate(subprocess_list):
                if not process.is_alive():
                    logger.warning(f"Child {index}, pid={process.pid} has exited")
                    return
            time.sleep(0.1)

    def listening_thread_fun():
        wait_child_exit()
        WorkerAgent_.startup_notify_exit(worker_ip, worker_port, agent_ip)
        _send_exit_signal_to_children(subprocess_list)

    thread = threading.Thread(target=listening_thread_fun)
    thread.start()


def _startup_agents(common_meta, worker_ip, worker_port,
                    agent_ip, agent_start_port, device_id_list, rank_id_list,
                    model_files, group_config_files, rank_table_file):
    """Start up all agents in one machine"""
    servable_name = common_meta.servable_name
    send_pipe_list = []
    subprocess_list = []
    c_send_pipe, p_recv_pipe = Pipe()
    group_file = ""
    agents_count = len(device_id_list)
    for index in range(agents_count):
        device_id, rank_id, model_file = device_id_list[index], rank_id_list[index], model_files[index]
        if group_config_files is not None:
            group_file = group_config_files[index]

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
    ret = _listening_agents_when_startup(p_recv_pipe, send_pipe_list, subprocess_list)

    msg = f"worker_ip: {worker_ip}, worker_port: {worker_port}, agent_ip: {agent_ip}, " \
          f"agent_start_port: {agent_start_port}, device ids: {device_id_list}, rank ids: {rank_id_list}, " \
          f"rank table file: {rank_table_file}, model files: {model_files}, group config files: {group_config_files}"
    if not ret:
        WorkerAgent_.notify_failed(worker_ip, worker_port)
        logger.info(f"Failed to start agents, {msg}")
        print(f"Failed to start agents, {msg}")
        return

    logger.info(f"Success to start agents, {msg}")
    print(f"Success to start agents, {msg}")
    _listening_agents_after_startup(subprocess_list, worker_ip, worker_port, agent_ip)


def startup_worker_agents(worker_ip, worker_port, model_files, group_config_files=None,
                          agent_start_port=7000, agent_ip=None, rank_start=None):
    r"""
    Start up all needed worker agenton current machine.

    Serving has two running modes. One is running in a single process, providing the Serving service of a single model.
    The other includes a master and multiple workers. This interface is for the second scenario.

    The master is responsible for providing the Serving access interface for clients,
    while the worker is responsible for providing the inference service of the specific model. The communications
    between the master and workers through gPRC are defined as (master_ip, master_port) and (worker_ip, worker_port).

    Args:
        worker_ip (str): The worker ip the agents linked to.
        worker_port (int): The worker port the agents linked to.
        model_files (list or tuple of str): All model files need in current machine, absolute path or path relative to
            this startup python script.
        group_config_files (None, list or tuple of str): All group config files need in current machine, absolute path
            or path relative to this startup python script, default None, which means there are no configuration files.
        agent_start_port (int): The starting agent port of the agents link to worker.
        agent_ip (str or None): The local agent ip, if it's None, the agent ip will be obtained from rank table file.
            Default None. Parameter agent_ip and parameter rank_start must have values at the same time,
            or both None at the same time.
        rank_start (int or None): The starting rank id of this machine, if it's None, the rank ip will be obtained from
            rank table file. Default None. Parameter agent_ip and parameter rank_start must have values at the same
            time, or both None at the same time.

    Examples:
        >>> import os
        >>> from mindspore_serving.worker import distributed
        >>> model_files = []
        >>> for i in range(8):
        >>>    model_files.append(f"models/device{i}/matmul.mindir")
        >>> distributed.startup_worker_agents(worker_ip="127.0.0.1", worker_port=6200, model_files=model_files)
    """
    check_type.check_str("worker_ip", worker_ip)
    check_type.check_ip_port("worker_port", worker_port)
    check_type.check_int("agent_start_port", agent_start_port, 1, 65535 - 7)
    model_files = check_type.check_and_as_str_tuple_list("model_files", model_files)
    if group_config_files is not None:
        group_config_files = check_type.check_and_as_str_tuple_list("group_config_files", group_config_files)

    ExitSignalHandle_.start()
    distributed_config = WorkerAgent_.get_agents_config_from_worker(worker_ip, worker_port)

    # get machine ip
    rank_list = distributed_config.rank_list
    local_device_id_list = []
    local_rank_id_list = []
    if agent_ip is None:
        if rank_start is not None:
            raise RuntimeError("Parameter 'agent_ip' and parameter 'rank_start' must have values at the same time, "
                               "or both None at the same time.")
        local_ip = _get_local_ip(rank_list, agent_start_port)
        # get all device_id and rank_id
        for rank_id, item in enumerate(rank_list):
            if item.ip == local_ip:
                local_device_id_list.append(item.device_id)
                local_rank_id_list.append(rank_id)
    else:
        if rank_start is None:
            raise RuntimeError("Parameter 'agent_ip' and parameter 'rank_start' must have values at the same time, "
                               "or both None at the same time.")
        check_type.check_str("agent_ip", agent_ip)
        check_type.check_int("rank_start", rank_start, 0)
        if rank_start >= len(rank_list):
            raise RuntimeError(f"Parameter 'rank_start' cannot equal or larger than rank size {len(rank_list)}.")
        if not _check_local_ip(agent_ip, agent_start_port):
            raise RuntimeError(f"Check ip 'agent_ip' valid failed, agent_ip: {agent_ip}")
        local_ip = agent_ip
        rank_table_ip = rank_list[rank_start].ip
        for rank_id, item in enumerate(rank_list):
            if item.ip == rank_table_ip:
                local_device_id_list.append(item.device_id)
                local_rank_id_list.append(rank_id)

    # handle model files and group config files
    if len(local_device_id_list) != len(model_files):
        raise RuntimeError(f"Card count {local_device_id_list} described rank table does not equal to model files size "
                           f"{len(model_files)}, model files: {model_files}")

    if group_config_files is not None and len(model_files) != len(group_config_files):
        raise RuntimeError(f"Model files count {len(model_files)} does not equal to group config files "
                           f"count {len(group_config_files)} when group_config_files is not None, "
                           f"model files: {model_files}, group config files: {group_config_files}")

    model_files, group_config_files = _update_model_files_path(model_files, group_config_files)

    # make json table file and export env
    rank_table_file = _make_json_table_file(distributed_config)
    _startup_agents(distributed_config.common_meta, worker_ip, worker_port, local_ip, agent_start_port,
                    local_device_id_list, local_rank_id_list,
                    model_files, group_config_files, rank_table_file)
