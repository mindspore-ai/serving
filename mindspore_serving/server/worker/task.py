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
from mindspore_serving._mindspore_serving import ExitSignalHandle_
from mindspore_serving.server.register.preprocess import preprocess_storage
from mindspore_serving.server.register.postprocess import postprocess_storage
from mindspore_serving.server.register.pipeline import pipeline_storage
from mindspore_serving import log as logger


class ServingSystemException(Exception):
    """Exception notify system error of worker, and need to exit py task"""

    def __init__(self, msg):
        super(ServingSystemException, self).__init__()
        self.msg = msg

    def __str__(self):
        return "Serving system error: " + self.msg


class ServingExitException(Exception):
    """Exception notify exit of worker, and need to exit py task"""

    def __str__(self):
        return "Serving has exited"


task_type_stop = "stop"
task_type_empty = "empty"
task_type_preprocess = "preprocess"
task_type_postprocess = "postprocess"
task_type_pipeline = "pipeline"

def has_worker_stopped():
    """Whether worker has stopped"""
    return ExitSignalHandle_.has_stopped()


class PyTask:
    """Base class for pipeline, preprocess and postprocess"""

    def __init__(self, switch_batch, task_name):
        super(PyTask, self).__init__()
        self.task_name = task_name
        self.switch_batch = switch_batch
        self.index = 0
        self.instances_size = 0
        self.result_batch = []
        self.task_info = None
        self.temp_result = None

    def push_failed_impl(self, count):
        """Base method to push failed result"""
        raise NotImplementedError

    def push_system_failed_impl(self):
        """Base method to push failed result"""
        raise NotImplementedError

    def push_result_batch_impl(self, result_batch):
        """Base method to push success result"""
        raise NotImplementedError

    def get_task_info(self, task_name):
        """Base method to get task info"""
        raise NotImplementedError

    def push_failed(self, count):
        """Push failed result"""
        self.push_result_batch()  # push success first
        self.push_failed_impl(count)
        self.index += count

    def push_system_failed(self):
        """Push failed result"""
        self.push_system_failed_impl()

    def push_result_batch(self):
        """Push success result"""
        if not self.result_batch:
            return
        result_batch = self.result_batch
        self.result_batch = []
        try:
            self.push_result_batch_impl(tuple(result_batch))
        except Exception as e:
            raise ServingSystemException(f"Push {self.task_name} result cause exception: {e}")
        self.index += len(result_batch)

    def has_next(self):
        """Is there result not handled"""
        return self.index < self.instances_size

    def run(self, task=None):
        """Run preprocess or postprocess, if last task has not been handled, continue to handle,
        or handle new task, every task has some instances"""
        if not self.temp_result:
            assert task is not None
            self.temp_result = self._run_inner(task)
        try:
            next(self.temp_result)
            if not self.has_next():
                self.temp_result = None
        except StopIteration:
            raise RuntimeError(f"Get next '{self.task_name}' result failed")

    def runpipeline(self, task):
        """Run pipeline"""
        instance_list = task.instance_list
        get_begin_time = time.time()
        self.task_info = self.get_task_info(task.name)
        for item in instance_list:
            result = self._handle_one_task(item)
            if result is not None:
                output = self._handle_result(result)
                self.result_batch.append(output)
        get_end_time = time.time()
        logger.info(f"{self.task_name} get result, "
                    f"cost time "
                    f"{(get_end_time - get_begin_time) * 1000} ms")
        self.push_result_batch()

    def _run_inner(self, task):
        """Iterator get next result, and push it to c++"""
        instances_size = len(task.instance_list)
        self.index = 0
        self.instances_size = len(task.instance_list)

        self.task_info = self.get_task_info(task.name)
        instance_list = task.instance_list
        # check input
        for item in instance_list:
            if not isinstance(item, tuple) or len(item) != self.task_info["inputs_count"]:
                raise RuntimeError(f"length of given inputs {len(item)}"
                                   f" not match {self.task_name} required " + str(self.task_info["inputs_count"]))

        result = self._handle_task(instance_list)
        while self.index < instances_size:
            try:
                get_result_time_end = time.time()
                last_index = self.index

                for _ in range(self.index, min(self.index + self.switch_batch, instances_size)):
                    if has_worker_stopped():
                        logger.info("Worker has exited, exit py task")
                        raise ServingExitException()

                    output = next(result)
                    output = self._handle_result(output)
                    self.result_batch.append(output)

                get_result_time = time.time()
                logger.info(f"{self.task_name} get result "
                            f"{last_index} ~ {last_index + len(self.result_batch) - 1} cost time "
                            f"{(get_result_time - get_result_time_end) * 1000} ms")

                self.push_result_batch()
                yield self.index  # end current coroutine, switch to next coroutine

            except StopIteration:
                result_count = self.index + len(self.result_batch)
                self.push_system_failed()
                raise RuntimeError(
                    f"expecting '{self.task_name}' yield count {result_count} equal to "
                    f"instance size {instances_size}")
            except ServingSystemException as e:
                self.push_system_failed()
                logger.warning(f"{self.task_name} catch system exception: {e}")
                raise e
            except ServingExitException as e:
                raise e
            except Exception as e:  # pylint: disable=broad-except
                # catch exception and try next
                logger.warning(f"{self.task_name} get result catch exception: {e}")
                logging.exception(e)
                self.push_failed(1)  # push success results and a failed result
                yield self.index  # end current coroutine, switch to next coroutine
                result = self._handle_task(instance_list[self.index:])

    def _handle_task(self, instance_list):
        """Continue to handle task on new task or task exception happened"""
        if not instance_list:
            return None
        try:
            outputs = self.task_info["fun"](instance_list)
            return outputs
        # pylint: disable=broad-except
        except Exception as e:
            logger.warning(f"{self.task_name} invoke catch exception: ")
            logging.exception(e)
            self.push_failed(len(instance_list))
            return None

    def _handle_one_task(self, instance):
        """Continue to handle task on new task or task exception happened"""
        try:
            inputs = []
            for i in range(self.task_info["inputs_count"]):
                inputs.append(instance[i])
            output = self.task_info["fun"](*inputs)
            return output
        # pylint: disable=broad-except
        except Exception as e:
            logger.warning(f"{self.task_name} invoke catch exception: ")
            logging.exception(e)
            self.push_failed(1)
            return None

    def _handle_result(self, output):
        """Further pipeline, processing results of preprocess or postprocess"""
        if not isinstance(output, (tuple, list)):
            output = (output,)
        if len(output) != self.task_info["outputs_count"]:
            raise ServingSystemException(f"length of return output {len(output)} "
                                         f"not match {self.task_name} signatures " +
                                         str(self.task_info["outputs_count"]))
        output = (item.asnumpy() if callable(getattr(item, "asnumpy", None)) else item for item in output)
        return output


class PyPreprocess(PyTask):
    """Preprocess implement"""

    def __init__(self, switch_batch):
        super(PyPreprocess, self).__init__(switch_batch, "preprocess")

    def push_failed_impl(self, count):
        """Push failed preprocess result to c++ env"""
        Worker_.push_preprocess_failed(count)

    def push_system_failed_impl(self):
        """Push failed preprocess result to c++ env"""
        Worker_.push_preprocess_system_failed()

    def push_result_batch_impl(self, result_batch):
        """Push success preprocess result to c++ env"""
        Worker_.push_preprocess_result(result_batch)

    def get_task_info(self, task_name):
        """Get preprocess task info, including inputs, outputs count, function of preprocess"""
        return preprocess_storage.get(task_name)


class PyPostprocess(PyTask):
    """Postprocess implement"""

    def __init__(self, switch_batch):
        super(PyPostprocess, self).__init__(switch_batch, "postprocess")

    def push_failed_impl(self, count):
        """Push failed postprocess result to c++ env"""
        Worker_.push_postprocess_failed(count)

    def push_system_failed_impl(self):
        """Push failed preprocess result to c++ env"""
        Worker_.push_postprocess_system_failed()

    def push_result_batch_impl(self, result_batch):
        """Push success postprocess result to c++ env"""
        Worker_.push_postprocess_result(result_batch)

    def get_task_info(self, task_name):
        """Get postprocess task info, including inputs, outputs count, function of postprocess"""
        return postprocess_storage.get(task_name)

class PyPipeline(PyTask):
    """PyPipeline implement"""

    def __init__(self, switch_batch):
        super(PyPipeline, self).__init__(switch_batch, "pipeline")

    def push_failed_impl(self, count):
        """Push failed pipeline result to c++ env"""
        Worker_.push_pipeline_failed(count)

    def push_system_failed_impl(self):
        """Push failed pipeline result to c++ env"""
        Worker_.push_pipeline_system_failed()

    def push_result_batch_impl(self, result_batch):
        """Push success pipeline result to c++ env"""
        Worker_.push_pipeline_result(result_batch)

    def get_task_info(self, task_name):
        """Get pipeline task info, including inputs, outputs count, function of pipeline"""
        return pipeline_storage.get(task_name)

class PyTaskThread(threading.Thread):
    """Thread for handling preprocess and postprocess"""

    def __init__(self, switch_batch):
        super(PyTaskThread, self).__init__()
        self.switch_batch = switch_batch
        if self.switch_batch <= 0:
            self.switch_batch = 1
        self.preprocess = PyPreprocess(self.switch_batch)
        self.postprocess = PyPostprocess(self.switch_batch)

    def run(self):
        """Run tasks of preprocess and postprocess, switch to other type of process when some instances are handled"""
        logger.info(f"start py task for preprocess and postprocess, switch_batch {self.switch_batch}")
        preprocess_turn = True
        while True:
            try:
                if has_worker_stopped():
                    logger.info("Worker has exited, exit py task")
                    break

                if not self.preprocess.has_next() and not self.postprocess.has_next():
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
                    if self.preprocess.has_next():
                        self.preprocess.run()
                    elif self.postprocess.has_next():
                        task = Worker_.try_get_preprocess_py_task()
                        if task.task_type == task_type_stop:
                            break
                        if task.task_type != task_type_empty:
                            self.preprocess.run(task)
                    preprocess_turn = False
                else:
                    if self.postprocess.has_next():
                        self.postprocess.run()
                    elif self.preprocess.has_next():
                        task = Worker_.try_get_postprocess_py_task()
                        if task.task_type == task_type_stop:
                            break
                        if task.task_type != task_type_empty:
                            self.postprocess.run(task)
                    preprocess_turn = True
            except ServingExitException:
                logger.info("Catch ServingExitException and exit py task")
                break
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(f"py task catch exception and exit: {e}")
                logging.exception(e)
                break
        logger.info("end py task for preprocess and postprocess")
        Worker_.stop_and_clear()

class PyPinelineThread(threading.Thread):
    """Thread for handling pipeline"""

    def __init__(self, switch_batch):
        super(PyPinelineThread, self).__init__()
        self.switch_batch = switch_batch
        if self.switch_batch <= 0:
            self.switch_batch = 1
        self.pipeline = PyPipeline(self.switch_batch)

    def run(self):
        """Run tasks of pipeline, switch to other type of process when some instances are handled"""
        logger.info(f"start py task for pipeline, switch_batch {self.switch_batch}")
        while True:
            try:
                if has_worker_stopped():
                    logger.info("Worker has exited, exit py task")
                    break
                task = Worker_.get_pipeline_task()
                if task.task_type == task_type_stop:
                    break
                if task.task_type != task_type_empty:
                    self.pipeline.runpipeline(task)

            except ServingExitException:
                logger.info("Catch ServingExitException and exit py task")
                break
            except Exception as e:  # pylint: disable=broad-except
                logger.error(f"py task catch exception and exit: {e}")
                logging.exception(e)
                break
        logger.info("end py task for pipeline")

py_task_thread = None
py_pipeline_thread = None

def _start_py_task(switch_batch):
    """Start python thread for pipelining, proprocessing and postprocessing"""
    global py_task_thread
    if py_task_thread is None:
        py_task_thread = PyTaskThread(switch_batch)
        py_task_thread.start()
    if pipeline_storage.has_registered():
        global py_pipeline_thread
        if py_pipeline_thread is None:
            py_pipeline_thread = PyPinelineThread(switch_batch)
            py_pipeline_thread.start()

def _join_py_task():
    """Join python thread for pipelining, proprocessing and postprocessing"""
    global py_task_thread
    if py_task_thread is not None:
        py_task_thread.join()
        py_task_thread = None
    if pipeline_storage.has_registered():
        global py_pipeline_thread
        if py_pipeline_thread is not None:
            py_pipeline_thread.join()
            py_pipeline_thread = None
