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
from mindspore_serving._mindspore_serving import DistributedServableConfig_, OneRankConfig_

from mindspore_serving import log as logger
from mindspore_serving.server.common import check_type
from mindspore_serving.server.worker.distributed import worker_agent


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


def _check_model_files(num, files, model_files, group_config_files):
    """Check the number of model files or group config files"""
    if isinstance(files, tuple):
        for item in files:
            if isinstance(item, list):
                if num == -1:
                    num = len(item)
                else:
                    if num != len(item):
                        raise RuntimeError(f"please check the number of  model files and group config files, "
                                           f"model files: {model_files}, group config files: {group_config_files}")
            else:
                if num not in (-1, 1):
                    raise RuntimeError(f"please check the number of  model files and group config files, "
                                       f"model files: {model_files}, group config files: {group_config_files}")
                num = 1
    return num


def _check_model_num(model_files, group_config_files):
    """Check the number of model files or group config files"""
    num = _check_model_files(-1, model_files, model_files, group_config_files)
    if group_config_files is not None:
        num = _check_model_files(-1, group_config_files, model_files, group_config_files)
        if num != 1:
            raise RuntimeError(f"please check the number of  group config files, currently only support one at most")


def _update_model_files_path(model_files, group_config_files):
    """Check and return model files or group config files"""
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    logger.info(f"input model files: {model_files}")
    logger.info(f"input group config files: {group_config_files}")
    model_files_temp = []
    for item in model_files:
        if isinstance(item, list):
            inner_files = []
            for inner in item:
                file_name = os.path.realpath(os.path.join(script_dir, inner))
                if not os.access(file_name, os.R_OK):
                    raise RuntimeError(f"Cannot access model file '{file_name}'")
                inner_files.append(file_name)
            model_files_temp.append(inner_files)
        else:
            file_name = os.path.realpath(os.path.join(script_dir, item))
            if not os.access(file_name, os.R_OK):
                raise RuntimeError(f"Cannot access model file '{file_name}'")
            model_files_temp.append(file_name)

    if group_config_files is not None:
        group_files_temp = []
        for item in group_config_files:
            if isinstance(item, list):
                inner_files = []
                for inner in item:
                    file_name = os.path.realpath(os.path.join(script_dir, inner))
                    if not os.access(file_name, os.R_OK):
                        raise RuntimeError(f"Cannot access group config file '{file_name}'")
                    inner_files.append(file_name)
                group_files_temp.append(inner_files)
            else:
                file_name = os.path.realpath(os.path.join(script_dir, item))
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
    Return True on receiving signal_success.
    """
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


def _agent_process(send_pipe, recv_pipe, index, start_config, dec_key, dec_mode):
    """Agent process"""
    parent_process = psutil.Process(os.getppid())
    try:
        # listening success or failed message from parent process
        worker_agent.start_worker_agent(start_config=start_config, dec_key=dec_key, dec_mode=dec_mode)
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
        exception = RuntimeError(f"Child {index} exception happen: {e}")
        send_pipe.send((index, exception))
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
    in some seconds later.
    """
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
                raise RuntimeError("Fail to start agents because of Ctrl+C")
            for send_pipe, process in zip(send_pipe_list, subprocess_list):
                if process.is_alive():
                    continue
                logger.warning("Fail to start agents because of death of one agent")
                _send_exit_msg_to_children(send_pipe_list, subprocess_list)
                raise RuntimeError("Fail to start agents because of death of one agent")

        index, msg = p_recv_pipe.recv()
        logger.info(f"Receive msg from Child {index}: {msg}")
        if isinstance(msg, Exception):
            logger.warning("Fail to start agents because of exception raise by one agent")
            _send_exit_msg_to_children(send_pipe_list, subprocess_list)
            raise msg

    for send_pipe in send_pipe_list:
        _send_pipe_msg(send_pipe, signal_success)


def _listening_agents_after_startup(subprocess_list, distributed_address, agent_ip):
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
        WorkerAgent_.startup_notify_exit(distributed_address, agent_ip)
        _send_exit_signal_to_children(subprocess_list)

    thread = threading.Thread(target=listening_thread_fun)
    thread.start()


def _startup_agents(common_meta, distributed_address,
                    agent_ip, agent_start_port, device_id_list, rank_id_list,
                    model_files, group_config_files, rank_table_file,
                    dec_key, dec_mode):
    """Start up all agents in one machine"""
    servable_name = common_meta.model_key
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
        start_config.model_file_names = model_file
        if group_config_files is not None:
            start_config.group_file_names = group_file
        start_config.rank_table_json_file_name = rank_table_file
        start_config.agent_address = agent_ip + ":" + str(agent_port)
        start_config.distributed_address = distributed_address
        start_config.common_meta = common_meta

        process = Process(target=_agent_process,
                          args=(c_send_pipe, c_recv_pipe, index, start_config, dec_key, dec_mode),
                          name=f"{servable_name}_worker_agent_rank{rank_id}_device{device_id}")
        process.start()
        subprocess_list.append(process)

    msg = f"distributed worker_address: {distributed_address}, agent_ip: {agent_ip}, " \
          f"agent_start_port: {agent_start_port}, device ids: {device_id_list}, rank ids: {rank_id_list}, " \
          f"rank table file: {rank_table_file}, model files: {model_files}, group config files: {group_config_files}"

    try:
        _listening_agents_when_startup(p_recv_pipe, send_pipe_list, subprocess_list)
    # pylint: disable=broad-except
    except Exception as e:
        WorkerAgent_.notify_failed(distributed_address)
        logger.error(f"Failed to start agents, {msg}")
        print(f"Failed to start agents, {msg}")
        raise e

    logger.info(f"Success to start agents, {msg}")
    print(f"Success to start agents, {msg}")
    _listening_agents_after_startup(subprocess_list, distributed_address, agent_ip)


class DistributedServableConfig:
    """Python DistributedServableConfig"""

    def __init__(self):
        self.rank_table_content = ""
        self.rank_list = None
        self.common_meta = None
        self.distributed_meta = None

    def set(self, config):
        """Set from C++ DistributedServableConfig_ obj"""
        self.rank_table_content = config.rank_table_content
        self.rank_list = []
        for item in config.rank_list:
            new_item = {"device_id": item.device_id, "ip": item.ip}
            self.rank_list.append(new_item)
        self.common_meta = {"model_key": config.common_meta.model_key,
                            "with_batch_dim": config.common_meta.with_batch_dim,
                            "without_batch_dim_inputs": config.common_meta.without_batch_dim_inputs,
                            "inputs_count": config.common_meta.inputs_count,
                            "outputs_count": config.common_meta.outputs_count}

        self.distributed_meta = {"rank_size": config.distributed_meta.rank_size,
                                 "stage_size": config.distributed_meta.stage_size}

    def get(self):
        """Get as C++ DistributedServableConfig_ obj"""
        config = DistributedServableConfig_()
        config.rank_table_content = self.rank_table_content
        rank_list = []
        for item in self.rank_list:
            new_item = OneRankConfig_()
            new_item.device_id = item["device_id"]
            new_item.ip = item["ip"]
            rank_list.append(new_item)
        config.rank_list = rank_list
        config.common_meta.model_key = self.common_meta["model_key"]
        config.common_meta.with_batch_dim = self.common_meta["with_batch_dim"]
        config.common_meta.without_batch_dim_inputs = self.common_meta["without_batch_dim_inputs"]
        config.common_meta.inputs_count = self.common_meta["inputs_count"]
        config.common_meta.outputs_count = self.common_meta["outputs_count"]

        config.distributed_meta.rank_size = self.distributed_meta["rank_size"]
        config.distributed_meta.stage_size = self.distributed_meta["stage_size"]
        return config


def _get_worker_distributed_config(distributed_address):
    """Get worker distributed config from worker through sub process"""
    c_send_pipe, p_recv_pipe = Pipe()

    def process_fun(c_send_pipe):
        try:
            distributed_config = WorkerAgent_.get_agents_config_from_worker(distributed_address)
            config = DistributedServableConfig()
            config.set(distributed_config)
            c_send_pipe.send(config)
        # pylint: disable=broad-except
        except Exception as e:
            c_send_pipe.send(e)

    process = Process(target=process_fun, args=(c_send_pipe,),
                      name=f"worker_agent_get_agents_config_from_worker")
    process.start()
    process.join()
    assert not process.is_alive()
    if p_recv_pipe.poll(0.1):
        config = p_recv_pipe.recv()
        if isinstance(config, Exception):
            raise config
        distributed_config = config.get()
        return distributed_config
    raise RuntimeError(f"Failed to get agents config from worker")


def startup_agents(distributed_address, model_files, group_config_files=None,
                   agent_start_port=7000, agent_ip=None, rank_start=None,
                   dec_key=None, dec_mode='AES-GCM'):
    r"""
    Start all required worker agents on the current machine. These worker agent processes are responsible for inference
    tasks on the local machine. For details, please refer to
    `MindSpore Serving-based Distributed Inference Service Deployment
    <https://www.mindspore.cn/serving/docs/en/r2.0/serving_distributed_example.html>`_.

    Args:
        distributed_address (str): The distributed worker address the agents linked to.
        model_files (Union[list[str], tuple[str]]): All model files need in current machine, absolute path or path
            relative to this startup python script.
        group_config_files (Union[list[str], tuple[str]], optional): All group config files need in current machine,
            absolute path or path relative to this startup python script, default None, which means there are no
            configuration files. Default: None.
        agent_start_port (int, optional): The starting agent port of the agents link to worker. Default: 7000.
        agent_ip (str, optional): The local agent ip, if it's None, the agent ip will be obtained from rank table file.
            Default None. Parameter agent_ip and parameter rank_start must have values at the same time,
            or both None at the same time. Default: None.
        rank_start (int, optional): The starting rank id of this machine, if it's None, the rank id will be obtained
            from rank table file. Default None. Parameter agent_ip and parameter rank_start must have values at the same
            time, or both None at the same time. Default: None.
        dec_key (bytes, optional): Byte type key used for decryption. The valid length is 16, 24, or 32. Default: None.
        dec_mode (str, optional): Specifies the decryption mode, take effect when dec_key is set.
            Option: 'AES-GCM' or 'AES-CBC'. Default: 'AES-GCM'.

    Raises:
        RuntimeError: Failed to start agents.

    Examples:
        >>> import os
        >>> from mindspore_serving.server import distributed
        >>> model_files = []
        >>> for i in range(8):
        >>>    model_files.append(f"models/device{i}/matmul.mindir")
        >>> distributed.startup_agents(distributed_address="127.0.0.1:6200", model_files=model_files)
    """
    check_type.check_str("distributed_address", distributed_address)
    check_type.check_int("agent_start_port", agent_start_port, 1, 65535 - 7)
    model_files = check_type.check_and_as_tuple_with_str_list("model_files", model_files)
    if group_config_files is not None:
        group_config_files = check_type.check_and_as_tuple_with_str_list("group_config_files", group_config_files)

    # check dec_key and dec_mode
    if dec_key is not None:
        if not isinstance(dec_key, bytes):
            raise RuntimeError(f"Parameter 'dec_key' should be bytes, but actually {type(dec_key)}")
        if not dec_key:
            raise RuntimeError(f"Parameter 'dec_key' should not be empty bytes")
        if len(dec_key) not in (16, 24, 32):
            raise RuntimeError(f"Parameter 'dec_key' length {len(dec_key)} expected to be 16, 24 or 32")
    check_type.check_str("dec_mode", dec_mode)
    if dec_mode not in ('AES-GCM', 'AES-CBC'):
        raise RuntimeError(f"Parameter 'dec_mode' expected to be 'AES-GCM' or 'AES-CBC'")

    ExitSignalHandle_.start()
    distributed_config = _get_worker_distributed_config(distributed_address)

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

    _check_model_num(model_files, group_config_files)
    model_files, group_config_files = _update_model_files_path(model_files, group_config_files)

    # make json table file and export env
    rank_table_file = _make_json_table_file(distributed_config)
    _startup_agents(distributed_config.common_meta, distributed_address, local_ip, agent_start_port,
                    local_device_id_list, local_rank_id_list,
                    model_files, group_config_files, rank_table_file, dec_key, dec_mode)
