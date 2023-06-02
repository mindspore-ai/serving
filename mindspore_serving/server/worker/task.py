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

import time
import logging
import numpy as np
from mindspore_serving._mindspore_serving import Worker_
from mindspore_serving._mindspore_serving import ExitSignalHandle_
from mindspore_serving.server.register.stage_function import stage_function_storage
from mindspore_serving import log as logger


class ServingSystemException(Exception):
    """Exception notify system error of worker, and need to exit py task"""

    def __init__(self, msg):
        super(ServingSystemException, self).__init__()
        self.msg = msg

    def __str__(self):
        return self.msg


def has_worker_stopped():
    """Whether worker has stopped"""
    return ExitSignalHandle_.has_stopped()


class PyTaskHandler:
    """Handling preprocess and postprocess"""

    def run(self):
        """Run tasks of preprocess and postprocess, switch to other type of process when some instances are handled"""
        logger.info(f"start python task handling thread")
        while True:
            try:
                if has_worker_stopped():
                    logger.info("Worker has exited, exit python task handling thread")
                    break
                task = Worker_.get_py_task()
                if task.has_stopped:
                    logger.info("Worker has exited, exit python task handling thread")
                    break
                self.run_inner(task)
            except Exception as e:  # pylint: disable=broad-except
                logger.error(f"py task catch exception and exit: {e}")
                logging.exception(e)
                break
        logger.info("end python task handling thread")
        Worker_.stop_and_clear()

    @staticmethod
    def run_inner(task):
        """Iterator get result, and push it to c++"""
        task_name = task.task_name
        task_info = stage_function_storage.get(task_name)
        instance_list = task.instance_list
        # check input
        inputs_count = task_info["inputs_count"]
        for item in instance_list:
            if not isinstance(item, tuple) or len(item) != inputs_count:
                raise RuntimeError(f"The inputs number {len(item)} provided is not equal to the inputs number "
                                   f"{inputs_count} required by function {task_name}, stage index {task.stage_index}")

        instances_size = len(task.instance_list)
        index = 0
        while index < instances_size:
            get_result_time_end = time.time()
            try:
                result = task_info["fun"](instance_list[index:])  # user-defined, may raise Exception
                if isinstance(result, (tuple, list)):  # convert return result to yield
                    result = iter(result)
            # pylint: disable=broad-except
            except Exception as e:
                logger.warning(f"{task_name} invoke catch exception: ")
                logging.exception(e)
                PyTaskHandler.push_failed(instances_size - index, str(e))
                return  # return will not terminate thread

            try:
                start_index = index
                for _ in range(index, instances_size):
                    output = next(result)  # user-defined, may raise Exception
                    if not isinstance(output, (tuple, list)):
                        output = (output,)
                    # check output count
                    if len(output) != task_info["outputs_count"]:
                        error_msg = f"The outputs number {len(output)} of one instance returned by function " \
                                    f"'{task_name}' is not equal to the outputs number {task_info['outputs_count']} " \
                                    f" registered in method {task.method_name}"
                        PyTaskHandler.push_system_failed(error_msg)
                        raise ServingSystemException(error_msg)
                    instance_result = []
                    for item in output:
                        # convert MindSpore Tensor to numpy
                        if callable(getattr(item, "asnumpy", None)):
                            item = item.asnumpy()
                        if isinstance(item, np.ndarray) and (not item.flags['FORC']):
                            item = np.ascontiguousarray(item)
                        instance_result.append(item)
                    # raise ServingSystemException when user-defined output is invalid
                    PyTaskHandler.push_result(instance_result)  # push outputs of one instance
                    index += 1

                get_result_time = time.time()
                logger.info(f"method {task.method_name} stage {task.stage_index} function {task_name} get result "
                            f"{start_index} ~ {instances_size - 1} cost time "
                            f"{(get_result_time - get_result_time_end) * 1000} ms")

            except StopIteration:  # raise by next
                error_msg = f"The number {index} of instances returned by function '{task_name}' is " \
                            f"not equal to the number {instances_size} of instances provided to this function."
                PyTaskHandler.push_system_failed(error_msg)
                raise RuntimeError(error_msg)
            except ServingSystemException as e:
                logger.error(f"{task_name} handling catch exception: {e}")
                PyTaskHandler.push_system_failed(e.msg)
                raise
            except Exception as e:  # pylint: disable=broad-except
                # catch exception and try next
                logger.warning(f"{task_name} get result catch exception: {e}")
                logging.exception(e)
                PyTaskHandler.push_failed(1, str(e))  # push success results and a failed result
                index += 1

    @staticmethod
    def push_failed(count, failed_msg):
        """Push failed result"""
        Worker_.push_pytask_failed(count, failed_msg)

    @staticmethod
    def push_system_failed(failed_msg):
        """Push failed result"""
        Worker_.push_pytask_system_failed(failed_msg)

    @staticmethod
    def push_result(instance_result):
        """Push success result"""
        try:
            Worker_.push_pytask_result(tuple(instance_result))
        except Exception as e:
            raise ServingSystemException(f"Push py task result cause exception: {e}")


def _start_py_task():
    """Start python thread for python task"""
    if Worker_.enable_pytask_que():
        PyTaskHandler().run()
    else:
        Worker_.wait_and_clear()
