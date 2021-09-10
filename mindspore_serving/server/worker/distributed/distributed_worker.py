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
from mindspore_serving._mindspore_serving import Worker_

from mindspore_serving.server.common import check_type
from mindspore_serving.server.worker._worker import _start_py_task
from mindspore_serving.server.worker._worker import stop_on_except, _load_servable_config


@stop_on_except
def start_servable(servable_directory, servable_name, rank_table_json_file, version_number,
                   distributed_address, wait_agents_time_in_seconds,
                   master_address, worker_address):
    r"""
    Start up the servable named 'servable_name' defined in 'servable_directory'.
    """
    check_type.check_str('servable_directory', servable_directory)
    check_type.check_str('servable_name', servable_name)
    check_type.check_int('version_number', version_number, 1)
    check_type.check_str('rank_table_json_file', rank_table_json_file)
    check_type.check_str('distributed_address', distributed_address)

    _load_servable_config(servable_directory, servable_name)
    Worker_.start_distributed_servable(servable_directory, servable_name, rank_table_json_file, version_number,
                                       distributed_address, master_address, worker_address,
                                       wait_agents_time_in_seconds)
    _start_py_task()
