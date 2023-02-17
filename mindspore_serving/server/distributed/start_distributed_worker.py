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
"""Start distributed worker process"""

import os
import sys

from mindspore_serving.server.worker import distributed
from mindspore_serving.server.common import check_type
from mindspore_serving._mindspore_serving import ExitSignalHandle_
from mindspore_serving._mindspore_serving import Worker_


def start_worker(servable_directory, servable_name, version_number, rank_table_json_file,
                 distributed_address, wait_agents_time_in_seconds,
                 master_address, listening_master=False):
    """Start distributed worker process"""
    check_type.check_str('servable_directory', servable_directory)
    check_type.check_str('servable_name', servable_name)
    check_type.check_int('version_number', version_number, 0)
    check_type.check_str('rank_table_json_file', rank_table_json_file)
    check_type.check_str('distributed_address', distributed_address)
    check_type.check_int('wait_agents_time_in_seconds', wait_agents_time_in_seconds, 0)

    check_type.check_str('master_address', master_address)
    check_type.check_bool('listening_master', listening_master)

    ExitSignalHandle_.start()  # Set flag to running and receive Ctrl+C message

    worker_pid = os.getpid()
    unix_socket_dir = "unix_socket_files"
    try:
        os.mkdir(unix_socket_dir)
    except FileExistsError:
        pass
    worker_address = f"unix:{unix_socket_dir}/serving_worker_{servable_name}_distributed_{worker_pid}"
    if len(worker_address) > 90:  # limit maximum unix domain socket address length
        worker_address = worker_address[:40] + "___" + worker_address[-40:]
    try:
        distributed.start_servable(servable_directory=servable_directory, servable_name=servable_name,
                                   version_number=version_number, rank_table_json_file=rank_table_json_file,
                                   distributed_address=distributed_address,
                                   wait_agents_time_in_seconds=wait_agents_time_in_seconds,
                                   master_address=master_address, worker_address=worker_address)
    except RuntimeError as ex:
        Worker_.notify_failed(master_address, f"{{distributed servable:{servable_name}, {ex}}}")
        raise


def parse_args_and_start():
    """Parse args and start distributed worker"""
    if len(sys.argv) != 9:
        raise RuntimeError("Expect length of input argv to be 8: str{servable_directory} str{servable_name} "
                           "int{version_number} str{rank_table_json_file} str{distributed_address} "
                           "int{wait_agents_time_in_seconds} str{master_address} bool{listening_master}")
    servable_directory = sys.argv[1]
    servable_name = sys.argv[2]
    version_number = int(sys.argv[3])
    rank_table_json_file = sys.argv[4]
    distributed_address = sys.argv[5]
    wait_agents_time_in_seconds = int(sys.argv[6])
    master_address = sys.argv[7]
    # pylint: disable=simplifiable-if-expression
    listening_master = True if sys.argv[8].lower() == "true" else False
    start_worker(servable_directory, servable_name, version_number, rank_table_json_file, distributed_address,
                 wait_agents_time_in_seconds, master_address, listening_master)


if __name__ == '__main__':
    parse_args_and_start()
