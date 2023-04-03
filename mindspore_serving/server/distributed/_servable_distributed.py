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
"""Distributed servable config"""

import os
import sys
import subprocess
from mindspore_serving.server.common import check_type, get_abs_path
import mindspore_serving.log as logger
from mindspore_serving.server._servable_common import ServableContextDataBase


class DistributedStartConfig:
    r"""
    Distributed servable start-up config.

    Args:
        servable_directory (str): The directory where the servable is located in. There expects to has a directory
            named `servable_name`. For more detail:
            `How to config Servable <https://www.mindspore.cn/serving/docs/zh-CN/r2.0/serving_model.html>`_ .
        servable_name (str): The servable name.
        rank_table_json_file (str): The rank table json file name.
        version_number (int): Servable version number to be loaded. The version number should be a positive integer,
            starting from 1, and 0 means to load the latest version. Default: 0.
        distributed_address (str): The worker address the agents linked to.
        wait_agents_time_in_seconds(int): The maximum time in seconds the worker waiting ready of all agents,
            0 means unlimited time, default 0

    Raises:
        RuntimeError: Input parameters are invalid.
    """

    def __init__(self, servable_directory, servable_name, rank_table_json_file, version_number,
                 distributed_address, wait_agents_time_in_seconds):
        super(DistributedStartConfig, self).__init__()
        check_type.check_str('servable_directory', servable_directory)
        logger.info(f"input servable directory: {servable_directory}")
        servable_directory = get_abs_path(servable_directory)
        logger.info(f"absolute servable directory: {servable_directory}")

        check_type.check_str('servable_name', servable_name)
        check_type.check_int('version_number', version_number, 0)
        if version_number == 0:
            version_number = 1

        check_type.check_str('rank_table_json_file', rank_table_json_file)
        logger.info(f"input rank table file: {rank_table_json_file}")
        rank_table_json_file = get_abs_path(rank_table_json_file)
        logger.info(f"absolute path of rank table file: {rank_table_json_file}")

        check_type.check_str('distributed_address', distributed_address)
        check_type.check_int('wait_agents_time_in_seconds', wait_agents_time_in_seconds, 0)

        self.servable_directory_ = servable_directory
        self.servable_name_ = servable_name
        self.version_number_ = version_number
        self.rank_table_json_file_ = rank_table_json_file
        self.distributed_address_ = distributed_address
        self.wait_agents_time_in_seconds_ = wait_agents_time_in_seconds

    @property
    def servable_directory(self):
        return self.servable_directory_

    @property
    def servable_name(self):
        return self.servable_name_

    @property
    def version_number(self):
        return self.version_number_

    @property
    def rank_table_json_file(self):
        return self.rank_table_json_file_

    @property
    def distributed_address(self):
        return self.distributed_address_

    @property
    def wait_agents_time_in_seconds(self):
        return self.wait_agents_time_in_seconds_


class DistributedContextData(ServableContextDataBase):
    """Used to start distributed servable worker process"""

    def __init__(self, distributed_config, master_address):
        super(DistributedContextData, self).__init__()
        if not isinstance(distributed_config, DistributedStartConfig):
            raise RuntimeError(f"Parameter '{distributed_config}' should be instance of DistributedStartConfig, "
                               f"but actually {type(distributed_config)}")
        self.distributed_config_ = distributed_config
        self.master_address_ = master_address
        self.log_new_file = True

    @property
    def servable_name(self):
        return self.distributed_config_.servable_name

    @property
    def version_number(self):
        return self.distributed_config_.version_number

    def to_string(self):
        """Used in logging"""
        return f"distributed servable name: {self.servable_name}"

    def new_worker_process(self):
        """Start distributed worker process"""
        python_exe = sys.executable
        script_dir = os.path.dirname(os.path.abspath(__file__))
        py_script = os.path.join(script_dir, "start_distributed_worker.py")
        config = self.distributed_config_
        arg = f"{python_exe} {py_script} {config.servable_directory} {config.servable_name} " \
              f"{config.version_number} {config.rank_table_json_file} {config.distributed_address} " \
              f"{config.wait_agents_time_in_seconds} {self.master_address_} True"
        args = arg.split(" ")

        serving_logs_dir = "serving_logs"
        try:
            os.mkdir(serving_logs_dir)
        except FileExistsError:
            pass

        write_mode = "w" if self.log_new_file else "a"
        self.log_new_file = False
        log_file_name = f"{serving_logs_dir}/log_{self.servable_name}_distributed.log"
        with open(log_file_name, write_mode) as fp:
            sub = subprocess.Popen(args=args, shell=False, stdout=fp, stderr=fp)
        return sub

    def can_restart(self):
        """Whether the worker can restart"""
        return False
