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
"""Interface for start up servable"""

import os
import time
import threading
import signal
import psutil

import mindspore_serving.log as logger
from mindspore_serving._mindspore_serving import WorkerContext_


class ServableContextDataBase:
    """Used to startup servable process"""

    def __init__(self):
        pass

    @property
    def servable_name(self):
        raise NotImplementedError

    @property
    def version_number(self):
        raise NotImplementedError

    def to_string(self):
        """For logging"""
        raise NotImplementedError

    def new_worker_process(self):
        """Start worker process to provide servable"""
        raise NotImplementedError

    def can_restart(self):
        """Whether the worker can restart"""
        return True

    def own_device(self):
        """Whether the worker occupy device"""
        return True


class WorkerContext:
    """Used to monitor and manage workers"""

    def __init__(self, context_data, master_address, sub_process):
        if not isinstance(context_data, ServableContextDataBase):
            raise RuntimeError(f"Parameter '{context_data}' should be instance of ServableReprInfo, "
                               f"but actually {type(context_data)}")
        self.context_data_ = context_data
        self.master_address_ = master_address
        self.sub_process_ = sub_process
        self.last_not_alive_time_ = None
        self.is_in_process_switching_ = False
        self.context = WorkerContext_.init_worker(context_data.servable_name, context_data.version_number,
                                                  context_data.to_string(), sub_process.pid)

    @property
    def servable_name(self):
        return self.context_data_.servable_name

    @property
    def worker_pid(self):
        return self.sub_process_.pid

    @property
    def master_address(self):
        return self.master_address_

    def to_string(self):
        """For logging"""
        return f"{self.context_data_.to_string()}, pid: {self.worker_pid}"

    @property
    def is_in_process_switching(self):
        return self.is_in_process_switching_

    def own_device(self):
        return self.context_data_.own_device()

    def ready(self):
        """Is worker ready to provide service"""
        return self.context.ready()

    def print_status(self):
        """DEBUG, used to print worker status"""
        self.context.print_status()

    def is_in_starting(self):
        """Whether the worker is in the process of startup"""
        return self.context.is_in_starting()

    def has_error_notified(self):
        """Whether error is reported by worker process during startup"""
        return self.context.has_error_notified()  # Error message of worker notifying master

    def get_notified_error(self):
        return self.context.get_notified_error()

    def has_exit_notified(self):
        """Whether exit is reported by worker process"""
        return self.context.has_exit_notified()  # Exit message of worker notifying master

    def can_be_restart(self):
        """Whether can restart the worker process"""
        if not self.context_data_.can_restart():
            return False
        normal_handled_count = self.context.normal_handled_count
        return normal_handled_count > 0

    def exit_for_enough_time(self):
        """ whether has exited for 1s, wait 1s for worker exit or error message"""
        return self.last_not_alive_time_ and (time.time() - self.last_not_alive_time_ > 1)

    def is_alive(self):
        """Whether the worker process is alive"""
        alive = (self.sub_process_.poll() is None)
        if not alive:
            if not self.last_not_alive_time_:
                self.context.notify_not_alive()
                self.last_not_alive_time_ = time.time()
        else:
            self.last_not_alive_time_ = None
        return alive

    def is_unavailable(self):
        """Whether the working process can link and provide services"""
        if self.is_in_process_switching:  # restart: shutdown and start worker
            return False
        if self.is_in_starting():  # start worker
            return False
        return self.context.is_unavailable

    def update_worker_process(self, new_sub_process):
        """Update worker process pid"""
        self.context.update_worker_pid(new_sub_process.pid)
        self.sub_process_ = new_sub_process
        self.last_not_alive_time_ = None

    def _terminate(self):
        self.sub_process_.terminate()

    def _shutdown_worker(self):
        """Shutdown worker process"""
        if not self.is_alive():
            return
        self._terminate()
        for _ in range(100):  # 10s
            if not self.is_alive():
                return
            time.sleep(0.1)
        self.send_exit_signal(signal.SIGKILL)
        self.context.notify_not_alive()

    def _restart_worker(self):
        """Restart worker process"""
        logger.info(f"restart worker, {self.to_string()}")
        self._shutdown_worker()
        try:
            new_sub_process = self.context_data_.new_worker_process()
        except RuntimeError as e:
            logger.error(f"Start worker failed: {e}")
            self.context.notify_start_failed(f"Start worker failed: {e}")
            return
        self.update_worker_process(new_sub_process)

    def shutdown_worker(self):
        """Shutdown worker process in thread"""
        self.handle_worker_process(self._shutdown_worker)

    def restart_worker(self):
        """Restart worker process in thread"""
        self.handle_worker_process(self._restart_worker)

    def handle_worker_process(self, thread_fun):
        """Used to do something in thread"""
        self.is_in_process_switching_ = True

        def handle_thread():
            thread_fun()
            self.is_in_process_switching_ = False

        thread = threading.Thread(target=handle_thread)
        thread.start()

    def send_exit_signal(self, sig):
        """Send signal to worker process, used to exit the worker process"""
        if not self.is_alive():
            return
        logger.warning(f"Send signal {sig} to worker, {self.to_string()}")
        try:
            child_process = psutil.Process(self.worker_pid)
            if not child_process.is_running():
                return
            children_of_child = child_process.children(recursive=True)
            for item in children_of_child:
                os.kill(item.pid, sig)
            self.sub_process_.send_signal(sig)
        except psutil.NoSuchProcess:
            return
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(f"Get exception when send signal {sig} to worker, {self.to_string()}, "
                           f"exception: {e}")
