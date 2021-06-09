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
"""MindSpore Serving Worker."""

from mindspore_serving import server
from mindspore_serving.server.common.decorator import deprecated

@deprecated("1.3.0", "mindspore_serving.server.start_servables")
def start_servable_in_master(servable_directory, servable_name, version_number=0, device_type=None, device_id=0):

    r"""
    Start up the servable named 'servable_name' defined in 'svable_directory', and the worker will run in
    the process of the master.

    Serving has two running modes. One is running in a single process, providing the Serving service of a single model.
    The other includes a master and multiple workers. This interface is for the first scenario.

    Args:
        servable_directory (str): The directory where the servable is located in. There expects to has a directory named
            `servable_name`. For more detail:
            `How to config Servable <https://www.mindspore.cn/tutorial/inference/zh-CN/master/serving_model.html>`_ .

        servable_name (str): The servable name.
        version_number (int): Servable version number to be loaded. The version number should be a positive integer,
            starting from 1, and 0 means to load the latest version. Default: 0.
        device_type (str): Currently only supports "Ascend", "Davinci" and None, Default: None.
            "Ascend" means the device type can be Ascend910 or Ascend310, etc.
            "Davinci" has the same meaning as "Ascend".
            None means the device type is determined by the MindSpore environment.
        device_id (int): The id of the device the model loads into and runs in.

    Examples:
        >>> import os
        >>> from mindspore_serving import worker
        >>> from mindspore_serving import master
        >>>
        >>> servable_dir = os.path.abspath(".")
        >>> worker.start_servable_in_master(servable_dir, "lenet", device_id=0)
        >>>
        >>> master.start_grpc_server("0.0.0.0", 5500)
        >>> master.start_restful_server("0.0.0.0", 1500)
    """
    config = server.ServableStartConfig(servable_directory=servable_directory, servable_name=servable_name,
                                        version_number=version_number, device_type=device_type, device_ids=device_id)
    server.start_servables(config)

@deprecated("1.3.0", "mindspore_serving.server.stop")
def stop():
    server.stop()
