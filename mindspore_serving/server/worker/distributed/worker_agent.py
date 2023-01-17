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

from mindspore_serving._mindspore_serving import WorkerAgent_, AgentStartUpConfig_, ExitSignalHandle_

from mindspore_serving import log as logger
from mindspore_serving.server.worker import init_mindspore


def start_worker_agent(start_config, dec_key, dec_mode):
    """Start up one worker agent on one device id, invoke by agent_startup.startup_worker_agents
    """
    if not isinstance(start_config, AgentStartUpConfig_):
        raise RuntimeError("Parameter 'start_config' should be instance of AgentStartUpConfig_")
    logger.info(f"rank_id={start_config.rank_id}, device_id={start_config.device_id}, "
                f"model_file='{start_config.model_file_names}', group_file='{start_config.group_file_names}', "
                f"rank_table_file='{start_config.rank_table_json_file_name}',"
                f"agent_address='{start_config.agent_address}', "
                f"distributed_address='{start_config.distributed_address}'"
                f"with_batch_dim={start_config.common_meta.with_batch_dim}, "
                f"without_batch_dim_inputs={start_config.common_meta.without_batch_dim_inputs}")

    ExitSignalHandle_.start()  # Set flag to running and receive Ctrl+C message

    init_mindspore.init_mindspore_cxx_env(False)
    os.environ["RANK_ID"] = str(start_config.rank_id)
    os.environ["DEVICE_ID"] = str(start_config.device_id)
    os.environ["MS_ENABLE_HCCL"] = "1"
    if start_config.group_file_names:
        os.environ["PARA_GROUP_FILE"] = ';'.join(start_config.group_file_names)

    os.environ["RANK_TABLE_FILE"] = start_config.rank_table_json_file_name

    for item in ("RANK_ID", "DEVICE_ID", "MS_ENABLE_HCCL", "PARA_GROUP_FILE", "RANK_TABLE_FILE",
                 "LD_LIBRARY_PATH", "PYTHONPATH"):
        logger.info(f"Env {item}: {os.getenv(item, None)}")
    if dec_key is None:
        dec_key = ''
    WorkerAgent_.start_agent(start_config, dec_key, dec_mode)

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
