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
import inspect

from mindspore_serving.worker import check_type


def startup_worker_agents(worker_ip, worker_port,
                          get_model_files_fun, get_group_configs_fun,
                          rank_start, agent_start_port=7000):
    """Start up all needed worker agents on one machine
    """
    check_type.check_str("worker_ip", worker_ip)
    check_type.check_ip_port("worker_port", worker_port)
    check_type.check_int("agent_start_port", agent_start_port, 1, 65535 - 7)
    if inspect.isfunction(get_model_files_fun):
        pass
    else:
        if not isinstance(get_model_files_fun, [list, tuple]):
            raise RuntimeError(f"Check failed, get_model_files_fun first must be function or tuple/list of str, "
                               f"now is {type(get_model_files_fun)}")
    if inspect.isfunction(get_group_configs_fun):
        pass
    else:
        if not isinstance(get_group_configs_fun, [list, tuple]):
            raise RuntimeError(f"Check failed, get_group_configs_fun first must be function or tuple/list of str, "
                               f"now is {type(get_group_configs_fun)}")
    check_type.check_int("rank_start", rank_start, 0)
    if rank_start % 8 != 0:
        raise RuntimeError(f"Parameter 'rank_start' must be mulfiply of 8, now is {rank_start}")
