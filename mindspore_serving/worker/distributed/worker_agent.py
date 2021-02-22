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
"""Serving, distributed worker agent"""

import os
import threading
from mindspore_serving.worker import init_mindspore
from mindspore_serving._mindspore_serving import WorkerAgent_, AgentStartUpConfig_
from mindspore_serving import log as logger


def start_worker_agent(start_config):
    """Start up one worker agent on one device id, invoke by agent_startup.startup_worker_agents
    """
    if not isinstance(start_config, AgentStartUpConfig_):
        raise RuntimeError("Parameter 'start_config' should be instance of AgentStartUpConfig_")

    init_mindspore.init_mindspore_cxx_env()
    os.environ["RANK_ID"] = str(start_config.rank_id)
    os.environ["DEVICE_ID"] = str(start_config.device_id)
    os.environ["MS_ENABLE_HCCL"] = "1"
    os.environ["PARA_GROUP_FILE"] = start_config.group_file_name
    os.environ["RANK_TABLE_FILE"] = start_config.rank_table_json_file_name

    for item in ("RANK_ID", "DEVICE_ID", "MS_ENABLE_HCCL", "PARA_GROUP_FILE", "RANK_TABLE_FILE",
                 "LD_LIBRARY_PATH", "PYTHONPATH"):
        logger.info(f"Env {item}: {os.getenv(item, '')}")
    WorkerAgent_.start_agent(start_config)

    start_wait_and_clear()


_wait_and_clear_thread = None


def start_wait_and_clear():
    """Waiting for Ctrl+C, and clear up environment"""

    def thread_func():
        logger.info("Serving worker Agent: wait for Ctrl+C to exit ------------------------------------")
        print("Serving worker Agent: wait for Ctrl+C to exit ------------------------------------")
        WorkerAgent_.wait_and_clear()
        logger.info("Serving worker Agent: exited ------------------------------------")
        print("Serving worker Agent: exited ------------------------------------")

    global _wait_and_clear_thread
    if not _wait_and_clear_thread:
        _wait_and_clear_thread = threading.Thread(target=thread_func)
        _wait_and_clear_thread.start()


def stop():
    r"""
    Stop the running of agent.
    """
    WorkerAgent_.stop_and_clear()
