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
"""Start worker process with single core servable"""

import os
import sys
import time
import threading
import signal
import psutil

import mindspore_serving.log as logger
from mindspore_serving.server import worker
from mindspore_serving.server.common import check_type
from mindspore_serving._mindspore_serving import ExitSignalHandle_
from mindspore_serving._mindspore_serving import Worker_


def start_listening_parent_thread(servable_name, device_id):
    """listening to parent process status"""

    def worker_listening_parent_thread():
        parent_process = psutil.Process(os.getppid())
        while not ExitSignalHandle_.has_stopped():
            if not parent_process.is_running():
                logger.warning(f"Worker {servable_name} device_id {device_id}, detect parent "
                               f"pid={parent_process.pid} has exited, worker begin to exit")
                worker.stop()
                cur_process = psutil.Process(os.getpid())
                for _ in range(100):  # 100x0.1=10s
                    try:
                        children = cur_process.children(recursive=True)
                        if not children:
                            logger.info(f"All current children processes have exited")
                            break
                        for child in children:
                            os.kill(child.pid, signal.SIGTERM)
                        time.sleep(0.1)
                    # pylint: disable=broad-except
                    except Exception as e:
                        logger.warning(f"Kill children catch exception {e}")

                return
            time.sleep(0.1)

    thread = threading.Thread(target=worker_listening_parent_thread)
    thread.start()


def start_worker(servable_directory, servable_name, version_number,
                 device_type, device_id, master_address, listening_master=False):
    """Start worker process with single core servable"""
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)  # for ccec compiler
    check_type.check_str('servable_directory', servable_directory)
    check_type.check_str('servable_name', servable_name)
    check_type.check_int('version_number', version_number, 0)

    check_type.check_int('device_id', device_id, 0)

    check_type.check_str('master_address', master_address)
    check_type.check_bool('listening_master', listening_master)

    ExitSignalHandle_.start()  # Set flag to running and receive Ctrl+C message
    if listening_master:
        start_listening_parent_thread(servable_name, device_id)

    worker_pid = os.getpid()
    unix_socket_dir = "unix_socket_files"
    try:
        os.mkdir(unix_socket_dir)
    except FileExistsError:
        pass
    worker_address = f"unix:{unix_socket_dir}/serving_worker_{servable_name}_device{device_id}_{worker_pid}"
    if len(worker_address) > 107:  # maximum unix domain socket address length
        worker_address = worker_address[:50] + "___" + worker_address[-50:]
    try:
        worker.start_servable(servable_directory=servable_directory, servable_name=servable_name,
                              version_number=version_number, device_type=device_type, device_id=device_id,
                              master_address=master_address, worker_address=worker_address)
    except Exception as ex:
        Worker_.notify_failed(master_address,
                              f"{{servable name:{servable_name}, device id:{device_id}, <{ex}>}}")
        raise


def parse_args_and_start():
    """Parse args and start distributed worker"""
    if len(sys.argv) != 8:
        raise RuntimeError("Expect length of input argv to be 8: str{servable_directory} str{servable_name} "
                           "int{version_number} str{device_type} int{device_id} str{master_address} "
                           "bool{listening_master}")
    servable_directory = sys.argv[1]
    servable_name = sys.argv[2]
    version_number = int(sys.argv[3])
    device_type = sys.argv[4]
    device_id = int(sys.argv[5])
    master_address = sys.argv[6]
    # pylint: disable=simplifiable-if-expression
    listening_master = True if sys.argv[7].lower() == "true" else False
    start_worker(servable_directory, servable_name, version_number, device_type, device_id, master_address,
                 listening_master)


if __name__ == '__main__':
    parse_args_and_start()
