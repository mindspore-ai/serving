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

import os
from mindspore_serving import worker


def start():
    servable_dir = os.path.abspath(".")
    worker.start_servable(servable_dir, "lenet", device_id=0,
                          master_ip="127.0.0.1", master_port=6500,
                          host_ip="127.0.0.1", host_port=6600)


if __name__ == "__main__":
    start()
