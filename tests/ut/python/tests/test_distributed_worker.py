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
"""test distributed worker"""

import os
import signal
import time
from multiprocessing import Process, Pipe
import numpy as np
import psutil

from common import serving_test, create_client, ServingTestBase
from mindspore_serving.worker import distributed
from mindspore_serving import master

distributed_import = r"""
import numpy as np
from mindspore_serving.worker import distributed
from mindspore_serving.worker import register
"""

distributed_declare_servable = r"""
distributed.declare_distributed_servable(rank_size=8, stage_size=1, with_batch_dim=False)
"""

rank_table_content = r"""
{
  "version": "1.0", "server_count": "1",
  "server_list": [
    {
      "server_id": "127.0.0.1",
      "device": [
        { "device_id": "0", "device_ip": "192.1.27.6", "rank_id": "0" },
        { "device_id": "1", "device_ip": "192.2.27.6", "rank_id": "1" },
        { "device_id": "2", "device_ip": "192.3.27.6", "rank_id": "2" },
        { "device_id": "3", "device_ip": "192.4.27.6", "rank_id": "3" },
        { "device_id": "4", "device_ip": "192.1.27.7", "rank_id": "4" },
        { "device_id": "5", "device_ip": "192.2.27.7", "rank_id": "5" },
        { "device_id": "6", "device_ip": "192.3.27.7", "rank_id": "6" },
        { "device_id": "7", "device_ip": "192.4.27.7", "rank_id": "7" }
      ],
      "host_nic_ip": "reserve"
    }
  ],
  "status": "completed"
}
"""


def init_distributed_servable():
    base = ServingTestBase()
    servable_content = distributed_import
    servable_content += distributed_declare_servable
    servable_content += r"""
@register.register_method(output_names=["y"])
def predict(x1, x2):
    y = register.call_servable(x1, x2)
    return y
"""
    base.init_distributed_servable(servable_content, 8, rank_table_content)
    return base


def start_distributed_grpc_server():
    base = init_distributed_servable()
    return base


def start_distributed_worker(base):
    def worker_process():
        distributed.start_distributed_servable_in_master(base.servable_dir, base.servable_name,
                                                         rank_table_json_file=base.rank_table_content_path,
                                                         worker_ip="127.0.0.1", worker_port=6200)
        master.start_grpc_server("0.0.0.0", 5500)

    worker = Process(target=worker_process)
    worker.start()
    time.sleep(0.5)  # wait parse rank table ready
    assert worker.is_alive()
    return worker


def start_agents(model_file_list, group_config_list):
    send_pipe, recv_pipe = Pipe()

    def agent_process(send_pipe):
        distributed.startup_worker_agents(worker_ip="127.0.0.1", worker_port=6200, model_files=model_file_list,
                                          group_config_files=group_config_list)
        send_pipe.send("Success")
        send_pipe.close()

    agent = Process(target=agent_process, args=(send_pipe,))
    agent.start()
    index = 0
    while index < 50 and agent.is_alive():  # wait max 5 s
        index += 1
        if recv_pipe.poll(0.1):
            break
    assert index < 50
    assert agent.is_alive()
    return agent


def send_exit(process):
    if not process.is_alive():
        return
    parent_process = psutil.Process(process.pid)
    child_processes = parent_process.children(recursive=True)

    def children_alive():
        return any([item.is_running() for item in child_processes])
    os.kill(process.pid, signal.SIGINT)
    for _ in range(50):  # 50*0.1s
        if not process.is_alive() and not children_alive():
            break
        time.sleep(0.1)
    for item in child_processes:
        if item.is_running():
            os.kill(item.pid, signal.SIGKILL)
    if process.is_alive():
        os.kill(process.pid, signal.SIGKILL)


@serving_test
def test_distributed_worker_worker_exit_success():
    base = start_distributed_grpc_server()
    worker_process = start_distributed_worker(base)
    base.add_on_exit(lambda: send_exit(worker_process))
    agent_process = start_agents(base.model_file_list, base.group_config_list)
    base.add_on_exit(lambda: send_exit(agent_process))

    client = create_client("localhost", 5500, base.servable_name, "predict")
    instances = [{}, {}, {}]
    y_data_list = []
    for index, instance in enumerate(instances):
        instance["x1"] = np.array([[1.1, 1.2], [2.2, 2.3]], np.float32) * (index + 1)
        instance["x2"] = np.array([[3.3, 3.4], [4.4, 4.5]], np.float32) * (index + 1)
        y_data_list.append((instance["x1"] + instance["x2"]).tolist())

    result = client.infer(instances)
    print(result)
    assert len(result) == 3
    assert result[0]["y"].dtype == np.float32
    assert result[1]["y"].dtype == np.float32
    assert result[2]["y"].dtype == np.float32
    assert result[0]["y"].tolist() == y_data_list[0]
    assert result[1]["y"].tolist() == y_data_list[1]
    assert result[2]["y"].tolist() == y_data_list[2]

    # send SIGINT to worker, expect worker and all agents exit
    agents = psutil.Process(agent_process.pid).children()

    def agents_alive():
        return any([item.is_running() for item in agents])
    os.kill(worker_process.pid, signal.SIGINT)
    for _ in range(50):  # 50*0.1s
        if not worker_process.is_alive() and not agent_process.is_alive() and not agents_alive():
            break
        time.sleep(0.1)
    assert not worker_process.is_alive()
    assert not agent_process.is_alive()
    assert not agents_alive()


@serving_test
def test_distributed_worker_agent_exit_success():
    base = start_distributed_grpc_server()
    worker_process = start_distributed_worker(base)
    base.add_on_exit(lambda: send_exit(worker_process))
    agent_process = start_agents(base.model_file_list, base.group_config_list)
    base.add_on_exit(lambda: send_exit(agent_process))

    client = create_client("localhost", 5500, base.servable_name, "predict")
    instances = [{}, {}, {}]
    y_data_list = []
    for index, instance in enumerate(instances):
        instance["x1"] = np.array([[1.1, 1.2], [2.2, 2.3]], np.float32) * (index + 1)
        instance["x2"] = np.array([[3.3, 3.4], [4.4, 4.5]], np.float32) * (index + 1)
        y_data_list.append((instance["x1"] + instance["x2"]).tolist())

    result = client.infer(instances)
    print(result)
    assert len(result) == 3
    assert result[0]["y"].tolist() == y_data_list[0]
    assert result[1]["y"].tolist() == y_data_list[1]
    assert result[2]["y"].tolist() == y_data_list[2]

    # send SIGINT to worker, expect worker and all agents exit
    agents = psutil.Process(agent_process.pid).children()

    def agents_alive():
        return any([item.is_running() for item in agents])
    os.kill(agent_process.pid, signal.SIGINT)
    for _ in range(50):  # 50*0.1s
        if not worker_process.is_alive() and not agent_process.is_alive() and not agents_alive():
            break
        time.sleep(0.1)
    assert not worker_process.is_alive()
    assert not agent_process.is_alive()
    assert not agents_alive()


@serving_test
def test_distributed_worker_agent_startup_killed_exit_success():
    base = start_distributed_grpc_server()
    worker_process = start_distributed_worker(base)
    base.add_on_exit(lambda: send_exit(worker_process))
    agent_process = start_agents(base.model_file_list, base.group_config_list)
    base.add_on_exit(lambda: send_exit(agent_process))

    client = create_client("localhost", 5500, base.servable_name, "predict")
    instances = [{}, {}, {}]
    y_data_list = []
    for index, instance in enumerate(instances):
        instance["x1"] = np.array([[1.1, 1.2], [2.2, 2.3]], np.float32) * (index + 1)
        instance["x2"] = np.array([[3.3, 3.4], [4.4, 4.5]], np.float32) * (index + 1)
        y_data_list.append((instance["x1"] + instance["x2"]).tolist())

    result = client.infer(instances)
    print(result)
    assert len(result) == 3
    assert result[0]["y"].tolist() == y_data_list[0]
    assert result[1]["y"].tolist() == y_data_list[1]
    assert result[2]["y"].tolist() == y_data_list[2]

    # send SIGINT to worker, expect worker and all agents exit
    agents = psutil.Process(agent_process.pid).children()

    def agents_alive():
        return any([item.is_running() for item in agents])
    os.kill(agent_process.pid, signal.SIGKILL)  # kill msg
    for _ in range(50):  # 50*0.1s
        # test agent_process.is_alive() first, it will make agents(children) notify exit of their parent
        if not agent_process.is_alive() and not worker_process.is_alive() and not agents_alive():
            break
        time.sleep(0.1)
    assert not worker_process.is_alive()
    assert not agent_process.is_alive()
    assert not agents_alive()


@serving_test
def test_distributed_worker_agent_killed_exit_success():
    base = start_distributed_grpc_server()
    worker_process = start_distributed_worker(base)
    base.add_on_exit(lambda: send_exit(worker_process))
    agent_process = start_agents(base.model_file_list, base.group_config_list)
    base.add_on_exit(lambda: send_exit(agent_process))

    client = create_client("localhost", 5500, base.servable_name, "predict")
    instances = [{}, {}, {}]
    y_data_list = []
    for index, instance in enumerate(instances):
        instance["x1"] = np.array([[1.1, 1.2], [2.2, 2.3]], np.float32) * (index + 1)
        instance["x2"] = np.array([[3.3, 3.4], [4.4, 4.5]], np.float32) * (index + 1)
        y_data_list.append((instance["x1"] + instance["x2"]).tolist())

    result = client.infer(instances)
    print(result)
    assert len(result) == 3
    assert result[0]["y"].tolist() == y_data_list[0]
    assert result[1]["y"].tolist() == y_data_list[1]
    assert result[2]["y"].tolist() == y_data_list[2]

    # send SIGINT to worker, expect worker and all agents exit
    agents = psutil.Process(agent_process.pid).children()
    assert agents

    def agents_alive():
        return any([item.is_running() for item in agents])
    os.kill(agents[0].pid, signal.SIGKILL)  # kill msg
    for _ in range(50):  # 50*0.1s
        if not worker_process.is_alive() and not agent_process.is_alive() and not agents_alive():
            break
        time.sleep(0.1)

    assert not worker_process.is_alive()
    assert not agent_process.is_alive()
    assert not agents_alive()
