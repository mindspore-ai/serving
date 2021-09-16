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
import os
import sys
import subprocess

from common import serving_test


def start_new_log_process(log_py_context, env_set):
    """start new process with log"""

    with open("test_log.py", "w") as fp:
        fp.write(log_py_context)
    log_file = os.path.join(os.getcwd(), "test_log.py")
    log_text = os.path.join(os.getcwd(), "test_log.txt")
    print(f"\npython {log_file} >& {log_text}")
    arg = f"{sys.executable} {log_file}"
    args = arg.split()

    new_env = os.environ.copy()
    new_env.update(env_set)

    with open(log_text, "w") as fp:
        sub = subprocess.Popen(args=args, shell=False, stdout=fp, stderr=fp, env=new_env)
        sub.wait()

    with open(log_text, "r") as fp:
        lines = fp.read()
        find_info = (lines.find("[INFO]") != -1)
        find_warning = (lines.find("[WARNING]") != -1)
        find_error = (lines.find("[ERROR]") != -1)
        print("log_text:------------------")
        print(lines)
        print("log_text end------------------")
    os.system(f"rm -f {log_file} {log_text}")
    return find_info, find_warning, find_error


def start_new_log_process_py(env_set):
    """start new process with python log"""
    log_py_context = r"""
from mindspore_serving import log as logger
from mindspore_serving import server
def log_process():
    logger.info("info msg test")
    logger.warning("warning msg test")
    logger.error("error msg test")
    logger.debug("debug msg test")

log_process()
    """
    return start_new_log_process(log_py_context, env_set)


def start_new_log_process_cpp(env_set):
    """start new process with cpp log"""
    log_py_context = r"""
from mindspore_serving import log as logger
from mindspore_serving import server
def log_process():
    # info
    server.start_grpc_server("0.0.0.0:5500")
    try:
        # error
        server.start_grpc_server("0.0.0.0:5500")
    except RuntimeError:
        pass

log_process()
    """
    return start_new_log_process(log_py_context, env_set)


@serving_test
def test_log_level_python_debug():
    find_info, find_warning, find_error = start_new_log_process_py({"GLOG_v": "0"})
    assert find_info
    assert find_warning
    assert find_error


@serving_test
def test_log_level_python_info():
    find_info, find_warning, find_error = start_new_log_process_py({"GLOG_v": "1"})
    assert find_info
    assert find_warning
    assert find_error


@serving_test
def test_log_level_python_warning():
    find_info, find_warning, find_error = start_new_log_process_py({"GLOG_v": "2"})
    assert not find_info
    assert find_warning
    assert find_error


@serving_test
def test_log_level_python_error():
    find_info, find_warning, find_error = start_new_log_process_py({"GLOG_v": "3"})
    assert not find_info
    assert not find_warning
    assert find_error


@serving_test
def test_log_level_cpp_debug():
    find_info, _, find_error = start_new_log_process_cpp({"GLOG_v": "0"})
    assert find_info
    assert find_error


@serving_test
def test_log_level_cpp_info():
    find_info, _, find_error = start_new_log_process_cpp({"GLOG_v": "1"})
    assert find_info
    assert find_error


@serving_test
def test_log_level_cpp_warning():
    find_info, _, find_error = start_new_log_process_cpp({"GLOG_v": "2"})
    assert not find_info
    assert find_error


@serving_test
def test_log_level_cpp_error():
    find_info, _, find_error = start_new_log_process_cpp({"GLOG_v": "3"})
    assert not find_info
    assert find_error


@serving_test
def test_log_level_cpp_debug2():
    find_info, _, find_error = start_new_log_process_cpp({"GLOG_v": "3", "MS_SUBMODULE_LOG_v": "{SERVING:0}"})
    assert find_info
    assert find_error


@serving_test
def test_log_level_cpp_info2():
    find_info, _, find_error = start_new_log_process_cpp({"GLOG_v": "3", "MS_SUBMODULE_LOG_v": "{SERVING:1}"})
    assert find_info
    assert find_error


@serving_test
def test_log_level_cpp_warning2():
    find_info, _, find_error = start_new_log_process_cpp({"GLOG_v": "3", "MS_SUBMODULE_LOG_v": "{SERVING:2}"})
    assert not find_info
    assert find_error


@serving_test
def test_log_level_cpp_error2():
    find_info, _, find_error = start_new_log_process_cpp({"GLOG_v": "3", "MS_SUBMODULE_LOG_v": "{SERVING:3}"})
    assert not find_info
    assert find_error
