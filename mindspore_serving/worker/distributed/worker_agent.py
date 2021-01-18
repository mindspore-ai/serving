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
"""Serving, distributed worker agent"""


def _start_worker_agent(agent_ip, agent_start_port, worker_ip, worker_port,
                        rank_id, device_id, model_file, group_config_file, rank_table_file,
                        with_bach_dim, without_batch_dim_inputs):
    """Start up one worker agent on one device id, invoke by agent_startup.startup_worker_agents
    """
    pass
