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
"""Startup serving server with distributed servable"""

from ._servable_distributed import DistributedStartConfig


def start_servable(servable_directory, servable_name, rank_table_json_file, version_number=1,
                   distributed_address="0.0.0.0:6200", wait_agents_time_in_seconds=0):
    r"""
    Start up the servable named 'servable_name' defined in 'servable_directory'.

    Args:
        servable_directory (str): The directory where the servable is located in. There expects to has a directory
            named `servable_name`. For more detail:
            `How to config Servable <https://www.mindspore.cn/serving/docs/zh-CN/r2.0/serving_model.html>`_ .

        servable_name (str): The servable name.
        version_number (int, optional): Servable version number to be loaded. The version number should be a positive
            integer, starting from 1. Default: 1.
        rank_table_json_file (str): The rank table json file name.
        distributed_address (str, optional): The distributed worker address the worker agents linked to.
            Default: "0.0.0.0:6200".
        wait_agents_time_in_seconds(int, optional): The maximum time in seconds the worker waiting ready of all agents,
            0 means unlimited time. Default: 0.

    Raises:
        RuntimeError: Failed to start the distributed servable.

    Examples:
        >>> import os
        >>> from mindspore_serving.server import distributed
        >>>
        >>> servable_dir = os.path.abspath(".")
        >>> distributed.start_servable(servable_dir, "matmul", startup_worker_agents="hccl_8p.json", \
        ...                            distributed_address="127.0.0.1:6200")
    """
    from mindspore_serving.server import start_servables
    config = DistributedStartConfig(servable_directory=servable_directory, servable_name=servable_name,
                                    rank_table_json_file=rank_table_json_file, version_number=version_number,
                                    distributed_address=distributed_address,
                                    wait_agents_time_in_seconds=wait_agents_time_in_seconds)
    start_servables(config)
