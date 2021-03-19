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
"""Serving, distributed worker startup"""
import sys
import os
from mindspore_serving._mindspore_serving import Worker_

from mindspore_serving import log as logger
from mindspore_serving.common import check_type
from mindspore_serving.worker._worker import _start_py_task, _start_wait_and_clear
from mindspore_serving.worker._worker import stop_on_except, _load_servable_config


def _get_rank_table_abs_path(rank_table_json_file):
    """Get absolute path of rank table file"""
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    logger.info(f"input rank table file: {rank_table_json_file}")
    rank_table_json_file = os.path.realpath(os.path.join(script_dir, rank_table_json_file))
    logger.info(f"absolute path of rank table file: {rank_table_json_file}")
    return rank_table_json_file


@stop_on_except
def start_distributed_servable(servable_directory, servable_name, rank_table_json_file, version_number=1,
                               worker_ip="0.0.0.0", worker_port=6200, master_ip="0.0.0.0", master_port=6100,
                               wait_agents_time_in_seconds=0):
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
            `How to config Servable <https://www.mindspore.cn/tutorial/inference/zh-CN/r1.2/serving_model.html>`_ .

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
        >>> distributed.start_distributed_servable(servable_dir, "matmul", rank_table_json_file="hccl_8p.json", \
        ...                                        worker_ip="127.0.0.1", worker_port=6200,   \
        ...                                        master_ip="127.0.0.1", master_port=6500)
    """
    check_type.check_str('servable_directory', servable_directory)
    check_type.check_str('servable_name', servable_name)
    check_type.check_int('version_number', version_number, 0)
    if version_number == 0:
        version_number = 1
    check_type.check_str('rank_table_json_file', rank_table_json_file)

    check_type.check_str('master_ip', master_ip)
    check_type.check_ip_port('master_port', master_port)

    check_type.check_str('worker_ip', worker_ip)
    check_type.check_ip_port('worker_port', worker_port)

    rank_table_json_file = _get_rank_table_abs_path(rank_table_json_file)

    _load_servable_config(servable_directory, servable_name)
    Worker_.start_distributed_servable(servable_directory, servable_name, rank_table_json_file, version_number,
                                       worker_ip, worker_port, master_ip, master_port, wait_agents_time_in_seconds)
    _start_py_task(Worker_.get_batch_size())
    _start_wait_and_clear()


@stop_on_except
def start_distributed_servable_in_master(servable_directory, servable_name, rank_table_json_file, version_number=1,
                                         worker_ip="0.0.0.0", worker_port=6200, wait_agents_time_in_seconds=0):
    r"""
    Start up the servable named 'servable_name' defined in 'svable_directory', and the worker will run in
    the process of the master.

    Serving has two running modes. One is running in a single process, providing the Serving service of a single model.
    The other includes a master and multiple workers. This interface is for the first scenario.

    Args:
        servable_directory (str): The directory where the servable is located in. There expects to has a directory named
            `servable_name`. For more detail:
            `How to config Servable <https://www.mindspore.cn/tutorial/inference/zh-CN/r1.2/serving_model.html>`_ .

        servable_name (str): The servable name.
        version_number (int): Servable version number to be loaded. The version number should be a positive integer,
            starting from 1, and 0 means to load the latest version. Default: 0.
        rank_table_json_file (str): The ranke table json file name.
        worker_ip (str): The worker ip the agents linked to.
        worker_port (int): The worker port the agents linked to.
        wait_agents_time_in_seconds(int): The maximum time in seconds the worker waiting ready of all agents,
            0 means unlimited time, default 0.

    Examples:
        >>> import os
        >>> from mindspore_serving.worker import distributed
        >>> from mindspore_serving import master
        >>>
        >>> servable_dir = os.path.abspath(".")
        >>> distributed.start_distributed_servable_in_master(servable_dir, "matmul", \
        ...                                                  rank_table_json_file="hccl_8p.json", \
        ...                                                  worker_ip="127.0.0.1", worker_port=6200)
        >>>
        >>> master.start_grpc_server("0.0.0.0", 5500)
        >>> master.start_restful_server("0.0.0.0", 1500)
    """
    check_type.check_str('servable_directory', servable_directory)
    check_type.check_str('servable_name', servable_name)
    check_type.check_int('version_number', version_number, 0)
    if version_number == 0:
        version_number = 1

    check_type.check_str('rank_table_json_file', rank_table_json_file)

    check_type.check_str('worker_ip', worker_ip)
    check_type.check_ip_port('worker_port', worker_port)

    rank_table_json_file = _get_rank_table_abs_path(rank_table_json_file)

    _load_servable_config(servable_directory, servable_name)
    Worker_.start_distributed_servable_in_master(servable_directory, servable_name, rank_table_json_file,
                                                 version_number, worker_ip, worker_port, wait_agents_time_in_seconds)
    _start_py_task(Worker_.get_batch_size())
    _start_wait_and_clear()
