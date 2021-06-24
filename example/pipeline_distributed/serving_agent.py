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
"""Start agents of Distributed Servable matmul"""

from mindspore_serving.server import distributed


def start_agents():
    """Start all the agents in current machine"""
    model_files = []
    group_configs = []
    for i in range(8):
        model_files.append([f"model/device{i}/matmul_0.mindir", f"model/device{i}/matmul_1.mindir"])
        group_configs.append([f"model/device{i}/group_config.pb"])

    distributed.startup_agents(distributed_address="127.0.0.1:6200", model_files=model_files,
                               group_config_files=group_configs)

if __name__ == '__main__':
    start_agents()
