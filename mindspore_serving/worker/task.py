# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Python run preprocess and postprocess in python"""

import threading
import time
import logging
from mindspore_serving._mindspore_serving import Worker_
from mindspore_serving.worker.register.preprocess import preprocess_storage
from mindspore_serving.worker.register.postprocess import postprocess_storage


class ServingSystemException(Exception):
    def __init__(self, msg):
        super(ServingSystemException, self).__init__()
        self.msg = msg

    def __str__(self):
        return "Serving system error: " + self.msg


task_type_stop = "stop"
task_type_empty = "empty"
task_type_preprocess = "preprocess"
task_type_postprocess = "postprocess"


class PyTask:
    def __init__(self, switch_batch, task_name):
        super(PyTask, self).__init__()
        self.task_name = task_name
        self.switch_batch = switch_batch
        self.temp_result = None
        self.task = None
        self.index = 0
        self.instances_size = 0
        self.stop_flag = False
        self.result_batch = []

    def push_failed_impl(self, count):
        raise NotImplementedError

    def push_result_batch_impl(self, result_batch):
        raise NotImplementedError

    def get_task_info(self, task_name):
        raise NotImplementedError

    def push_failed(self, count):
        self.push_result_batch()  # push success first
        self.push_failed_impl(count)
        self.index += count

    def push_result_batch(self):
        if len(self.result_batch) == 0:
            return

        get_result_time_end = time.time()
        last_index = self.index

        self.index += len(self.result_batch)
        self.push_result_batch_impl(tuple(self.result_batch))
        self.result_batch = []

        get_result_time = time.time()
        print(f"-----------------{self.task_name} push result {last_index} ~ {self.index - 1} cost time",
              (get_result_time - get_result_time_end) * 1000, "ms")

    def in_processing(self):
        return self.temp_result is not None

    def run(self, task=None):
        if not self.temp_result:
            assert task is not None
            self.instances_size = len(task.instance_list)
            self.index = 0
            self.task = task
            self.temp_result = self._handle_task()
            if not self.temp_result:
                return
        while self.index < self.instances_size:
            try:
                get_result_time_end = time.time()
                last_index = self.index

                for i in range(self.index, min(self.index + self.switch_batch, self.instances_size)):
                    output = next(self.temp_result)
                    output = self._handle_result(output)
                    self.result_batch.append(output)

                get_result_time = time.time()
                print(f"-----------------{self.task_name} get result {last_index} ~  cost time",
                      (get_result_time - get_result_time_end) * 1000, "ms")

                self.push_result_batch()
                break
            except StopIteration:
                self.push_result_batch()
                self.push_failed(self.instances_size - self.index)
                raise RuntimeError(
                    f"expecting {self.task_name} yield count equal to instance size {self.instances_size}")
            except ServingSystemException as e:
                raise e
            except Exception as e:  # catch exception and try next
                print("{self.task_name} get result catch exception: ")
                logging.exception(e)
                self.push_failed(1)  # push success results and a failed result
                self.temp_result = self._handle_task_continue()

        if self.index >= self.instances_size:
            self.temp_result = None

    def _handle_task(self):
        self.task_info = self.get_task_info(self.task.name)
        instance_list = self.task.instance_list

        self.context_list = self.task.context_list
        # check input
        for input in instance_list:
            if not isinstance(input, tuple) or len(input) != self.task_info["inputs_count"]:
                raise RuntimeError("length of given inputs " + str(len(input))
                                   + " not match {self.task_name} required " + str(self.task_info["inputs_count"]))
        return self._handle_task_continue()

    def _handle_task_continue(self):
        if self.index >= self.instances_size:
            return None
        instance_list = self.task.instance_list
        try:
            outputs = self.task_info["fun"](instance_list[self.index:])
            return outputs
        except Exception as e:
            print(f"{self.task_name} invoke catch exception: ")
            logging.exception(e)
            self.push_failed(len(instance_list) - self.index)
            return None

    def _handle_result(self, output):
        if not isinstance(output, (tuple, list)):
            output = (output,)
        if len(output) != self.task_info["outputs_count"]:
            raise ServingSystemException(f"length of return output {len(output)} "
                                         f"not match {self.task_name} signatures " +
                                         str(self.task_info["outputs_count"]))
        output = (item.asnumpy() if callable(getattr(item, "asnumpy", None)) else item for item in output)
        return output


class PyPreprocess(PyTask):
    def __init__(self, switch_batch):
        super(PyPreprocess, self).__init__(switch_batch, "preprocess")

    def push_failed_impl(self, count):
        Worker_.push_preprocess_failed(count)

    def push_result_batch_impl(self, result_batch):
        Worker_.push_preprocess_result(result_batch)

    def get_task_info(self, task_name):
        return preprocess_storage.get(task_name)


class PyPostprocess(PyTask):
    def __init__(self, switch_batch):
        super(PyPostprocess, self).__init__(switch_batch, "postprocess")

    def push_failed_impl(self, count):
        Worker_.push_postprocess_failed(count)

    def push_result_batch_impl(self, result_batch):
        Worker_.push_postprocess_result(result_batch)

    def get_task_info(self, task_name):
        return postprocess_storage.get(task_name)


class PyTaskThread(threading.Thread):
    def __init__(self, switch_batch):
        super(PyTaskThread, self).__init__()
        self.switch_batch = switch_batch
        if self.switch_batch <= 0:
            self.switch_batch = 8
        self.preprocess = PyPreprocess(self.switch_batch)
        self.postprocess = PyPostprocess(self.switch_batch)

    def run(self):
        print("start py task for preprocess and postprocess, switch_batch", self.switch_batch)
        preprocess_turn = True
        while True:
            try:
                if not self.preprocess.in_processing() and not self.postprocess.in_processing():
                    task = Worker_.get_py_task()
                    if task.task_type == task_type_stop:
                        break
                    if task.task_type == task_type_preprocess:
                        self.preprocess.run(task)
                        preprocess_turn = False
                    elif task.task_type == task_type_postprocess:
                        self.postprocess.run(task)
                        preprocess_turn = True

                # in preprocess turn, when preprocess is still running, switch to running preprocess
                # otherwise try get next preprocess task when postprocess is running
                # when next preprocess is not available, switch to running postprocess
                if preprocess_turn:
                    if self.preprocess.in_processing():
                        self.preprocess.run()
                    elif self.postprocess.in_processing():
                        task = Worker_.try_get_preprocess_py_task()
                        if task.task_type == task_type_stop:
                            break
                        if task.task_type != task_type_empty:
                            self.preprocess.run(task)
                    preprocess_turn = False
                else:
                    if self.postprocess.in_processing():
                        self.postprocess.run()
                    elif self.preprocess.in_processing():
                        task = Worker_.try_get_postprocess_py_task()
                        if task.task_type == task_type_stop:
                            break
                        if task.task_type != task_type_empty:
                            self.postprocess.run(task)
                    preprocess_turn = True

            except Exception as e:
                print("py task catch exception and exit: ")
                logging.exception(e)
                break
        print("end py task for preprocess and postprocess")
        Worker_.stop()


py_task_thread = None


def start_py_task(switch_batch):
    global py_task_thread
    if py_task_thread is None:
        py_task_thread = PyTaskThread(switch_batch)
        py_task_thread.start()
