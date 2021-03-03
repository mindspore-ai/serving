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
"""Start Distributed Servable matmul"""

import os
import sys
from mindspore_serving.worker import distributed


def start():
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    distributed.start_distributed_servable(servable_dir, "matmul",
                                           rank_table_json_file="rank_table_8pcs.json",
                                           version_number=1,
                                           worker_ip="127.0.0.1", worker_port=6200,
                                           master_ip="127.0.0.1", master_port=6500)

if __name__ == "__main__":
    start()
