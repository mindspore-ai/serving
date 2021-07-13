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
"""MindSpore Serving Distributed Worker."""

from mindspore_serving.server import distributed
from mindspore_serving.server.common.decorator import deprecated


@deprecated("1.3.0", "mindspore_serving.server.distributed.start_servable")
def start_distributed_servable_in_master(servable_directory, servable_name, rank_table_json_file, version_number=1,
                                         worker_ip="0.0.0.0", worker_port=6200, wait_agents_time_in_seconds=0):
    r"""
    Start up the servable named 'servable_name' defined in 'servable_directory', and link the worker to the master
    through gRPC (master_ip, master_port).

    Serving has two running modes. One is running in a single process, providing the Serving service of a single model.
    The other includes a master and multiple workers. This interface is for the second scenario.

    The master is responsible for providing the Serving access interface for clients,
    while the worker is responsible for providing the inference service of the specific model. The communications
    between the master and workers through gPRC are defined as (master_ip, master_port) and (worker_ip, worker_port).

    Args:
        servable_directory (str): The directory where the servable is located in. There expects to has a directory
            named `servable_name`. For more detail:
            `How to config Servable <https://www.mindspore.cn/serving/docs/zh-CN/master/serving_model.html>`_ .

        servable_name (str): The servable name.
        version_number (int): Servable version number to be loaded. The version number should be a positive integer,
            starting from 1, and 0 means to load the latest version. Default: 0.
        rank_table_json_file (str): The ranke table json file name.
        master_ip (str): The master ip the worker linked to.
        master_port (int): The master port the worker linked to.
        worker_ip (str): The worker ip the master and agents linked to.
        worker_port (int): The worker port the master and agents linked to.
        wait_agents_time_in_seconds(int): The maximum time in seconds the worker waiting ready of all agents,
            0 means unlimited time, default 0

    Examples:
        >>> import os
        >>> from mindspore_serving.worker import distributed
        >>>
        >>> servable_dir = os.path.abspath(".")
        >>> distributed.start_distributed_servable_in_master(servable_dir, "matmul", \
        ...                                                  startup_worker_agents="hccl_8p.json", \
        ...                                                  worker_ip="127.0.0.1", worker_port=6200)
    """
    distributed.start_servable(servable_directory=servable_directory, servable_name=servable_name,
                               rank_table_json_file=rank_table_json_file, version_number=version_number,
                               distributed_address=f"{worker_ip}:{worker_port}",
                               wait_agents_time_in_seconds=wait_agents_time_in_seconds)


@deprecated("1.3.0", "mindspore_serving.server.distributed.startup_agents")
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
    distributed.startup_agents(distributed_address=f"{worker_ip}:{worker_port}",
                               model_files=model_files, group_config_files=group_config_files,
                               agent_start_port=agent_start_port, agent_ip=agent_ip, rank_start=rank_start)


@deprecated("1.3.0", "mindspore_serving.server.distributed.declare_servable")
def declare_distributed_servable(rank_size, stage_size, with_batch_dim=True, without_batch_dim_inputs=None):
    """declare distributed servable in servable_config.py.

    Args:
        rank_size (int): Te rank size of the distributed model.
        stage_size (int): The stage size of the distributed model.
        with_batch_dim (bool): Whether the first shape dim of the inputs and outputs of model is batch, default True.
        without_batch_dim_inputs (None, int, tuple or list of int): Index of inputs that without batch dim
            when with_batch_dim is True.

    Examples:
        >>> from mindspore_serving.worker import distributed
        >>> distributed.declare_distributed_servable(rank_size=8, stage_size=1)
    """
    distributed.declare_servable(rank_size=rank_size, stage_size=stage_size, with_batch_dim=with_batch_dim,
                                 without_batch_dim_inputs=without_batch_dim_inputs)
