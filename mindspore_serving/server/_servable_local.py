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
"""Interface for start up single core servable"""
import logging
import os
import sys
import subprocess

from mindspore_serving.server.common import check_type
from mindspore_serving.server._servable_common import ServableContextDataBase
from mindspore_serving._mindspore_serving import Worker_


class ServableStartConfig:
    r"""
    Servable startup configuration.

    For more detail, please refer to
    `MindSpore-based Inference Service Deployment
     <https://www.mindspore.cn/tutorial/inference/zh-CN/master/serving_example.html>`_ and
    `Servable Provided Through Model Configuration
     <https://www.mindspore.cn/tutorial/inference/zh-CN/master/serving_model.html>`_.

    Args:
        servable_directory (str): The directory where the servable is located in. There expects to has a directory
            named `servable_name`.
        servable_name (str): The servable name.
        device_ids (Union[int, list[int], tuple[int]]): The device list the model loads into and runs in.
        version_number (int, optional): Servable version number to be loaded. The version number should be a positive
            integer, starting from 1, and 0 means to load the latest version. Default: 0.
        device_type (str, optional): Currently supports "Ascend", "GPU" and None, Default: None.

            - "Ascend": the platform expected to be Ascend910 or Ascend310, etc.
            - "GPU": the platform expected to be Nvidia GPU.
            - None: the platform is determined by the MindSpore environment.

    Raises:
        RuntimeError: The type or value of the parameters are invalid.
    """

    def __init__(self, servable_directory, servable_name, device_ids, version_number=0, device_type=None):
        super(ServableStartConfig, self).__init__()
        check_type.check_str("servable_directory", servable_directory)
        check_type.check_str("servable_name", servable_name)
        check_type.check_int("version_number", version_number, 0)

        self.servable_directory_ = servable_directory
        self.servable_name_ = servable_name
        self.version_number_ = version_number

        self.device_ids_ = check_type.check_and_as_int_tuple_list("device_ids", device_ids, 0)
        if device_type is not None:
            check_type.check_str("device_type", device_type)
        else:
            device_type = "None"
        self.device_type_ = device_type

    @property
    def servable_directory(self):
        return self.servable_directory_

    @property
    def servable_name(self):
        return self.servable_name_

    @property
    def version_number(self):
        return self.version_number_

    @property
    def device_type(self):
        return self.device_type_

    @property
    def device_ids(self):
        return self.device_ids_


def get_newest_version_number(config):
    """Get newest version number of servable"""
    if not isinstance(config, ServableStartConfig):
        raise RuntimeError(f"Parameter '{config}' should be ServableStartConfig, but actually {type(config)}")
    max_version = 0
    version_root_dir = os.path.join(config.servable_directory, config.servable_name)
    try:
        files = os.listdir(version_root_dir)
    except FileNotFoundError:
        return 0
    for file in files:
        if not os.path.isdir(os.path.join(version_root_dir, file)):
            continue
        if not file.isdigit() or file == "0" and str(int(file)) != file:
            continue
        version = int(file)
        if max_version < version:
            max_version = version
    return max_version


def _check_and_merge_config(configs):
    """Merge ServableStartConfig with the same version number"""
    newest_version_number_list = {}
    servable_dir_list = {}
    servable_device_ids = {}
    servable_device_types = {}

    # Get Device type: AscendCL, AscendMS, Gpu, Cpu, set Ascend in ServableStartConfig instead of AscendCL, AscendMS
    device_type = Worker_.get_device_type()
    if device_type in ("AscendMS", "AscendCL"):
        device_type = "Ascend"

    for config in configs:
        if not isinstance(config, ServableStartConfig):
            continue
        if config.device_type != "None" and config.device_type.lower() != device_type.lower():
            raise RuntimeError(f"The device type '{config.device_type}' of servable name {config.servable_name} "
                               f"is inconsistent with current running environment, supported device type: "
                               f"'None' or '{device_type}'")

        if config.servable_name in servable_dir_list:
            if config.servable_directory != servable_dir_list[config.servable_name]:
                raise RuntimeError(f"The servable directory of servable name {config.servable_name} is different in"
                                   f" multiple configurations, servable directory: "
                                   f"{config.servable_directory} and {servable_dir_list[config.servable_name]}")
            if config.device_type != servable_device_types[config.servable_name]:
                raise RuntimeError(f"The device type of servable name {config.servable_name} is different in "
                                   f"multiple configurations, device type: "
                                   f"{config.device_type} and {servable_device_types[config.device_type]}")
        else:
            config_dir = os.path.join(config.servable_directory, config.servable_name)
            if not os.path.isdir(config_dir):
                raise RuntimeError(f"Check servable config failed, directory '{config_dir}' not exist, servable "
                                   f"directory '{config.servable_directory}', servable name '{config.servable_name}'")

            config_file = os.path.join(config_dir, "servable_config.py")
            if not os.path.isfile(config_file):
                raise RuntimeError(f"Check servable config failed, file '{config_file}' not exist,  servable directory "
                                   f"'{config.servable_directory}', servable name '{config.servable_name}'")

            newest_version = get_newest_version_number(config)
            if newest_version == 0:
                raise RuntimeError(f"There is no valid version of servable, servable directory: "
                                   f"{config.servable_directory}, servable name: {config.servable_name}")
            # pylint: disable=logging-fstring-interpolation
            logging.info(f"The newest version number of servable {config.servable_name} is {newest_version}, "
                         f"servable directory: {config.servable_directory}")
            servable_dir_list[config.servable_name] = config.servable_directory
            newest_version_number_list[config.servable_name] = newest_version
            servable_device_ids[config.servable_name] = {}
            servable_device_types[config.servable_name] = config.device_type

        if config.version_number == 0:  # newest version
            version_number = newest_version_number_list[config.servable_name]
        else:
            version_dir = os.path.join(config.servable_directory, config.servable_name, str(config.version_number))
            if not os.path.exists(version_dir):
                raise RuntimeError(f"There is no servable of the specified version number, "
                                   f"specified version number: {config.version_number}, "
                                   f"servable directory: {config.servable_directory}, "
                                   f"servable name: {config.servable_name}")
            if not os.path.isdir(version_dir):
                raise RuntimeError(f"Expect {version_dir} to be a directory, servable directory: "
                                   f"{config.servable_directory}, servable name: {config.servable_name}")
            version_number = config.version_number
        if version_number not in servable_device_ids[config.servable_name]:
            servable_device_ids[config.servable_name][version_number] = set()
        for device_id in config.device_ids:
            servable_device_ids[config.servable_name][version_number].add(device_id)
    return servable_dir_list, servable_device_ids, servable_device_types


def merge_config(configs):
    """Merge ServableStartConfig with the same version number"""
    servable_dir_list, servable_device_ids, servable_device_types = _check_and_merge_config(configs)
    configs_ret = []

    device_type = Worker_.get_device_type()
    allow_reuse_device = True
    device_ids_used = set()
    if device_type == "AscendMS":
        allow_reuse_device = False

    for servable_name, servable_dir in servable_dir_list.items():
        for version_number, device_ids in servable_device_ids[servable_name].items():
            config = ServableStartConfig(servable_directory=servable_dir, servable_name=servable_name,
                                         device_ids=tuple(device_ids), version_number=version_number,
                                         device_type=servable_device_types[servable_name])
            configs_ret.append(config)
            if not allow_reuse_device:
                for device_id in device_ids:
                    if device_id in device_ids_used:
                        raise RuntimeError(f"Ascend 910 device id {device_id} is used repeatedly in servable "
                                           f"{servable_name}")
                    device_ids_used.add(device_id)
    for config in configs:
        if not isinstance(config, ServableStartConfig):
            configs_ret.append(config)
    return configs_ret


class ServableContextData(ServableContextDataBase):
    """Used to startup servable process"""

    def __init__(self, servable_config, device_id, master_address):
        super(ServableContextData, self).__init__()
        self.servable_config = servable_config
        self.device_id = device_id
        self.master_address = master_address
        self.log_new_file = True

    @property
    def servable_name(self):
        return self.servable_config.servable_name

    @property
    def version_number(self):
        return self.servable_config.version_number

    def to_string(self):
        """For logging"""
        return f"servable name: {self.servable_name}, device id: {self.device_id}"

    def new_worker_process(self):
        """Start worker process to provide servable"""
        python_exe = sys.executable
        servable_config = self.servable_config
        device_type = servable_config.device_type
        if device_type is None:
            device_type = "None"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        py_script = os.path.join(script_dir, "start_worker.py")
        arg = f"{python_exe} {py_script} {servable_config.servable_directory} {servable_config.servable_name} " \
              f"{servable_config.version_number} {device_type} {self.device_id} {self.master_address} True"
        args = arg.split(" ")

        serving_logs_dir = "serving_logs"
        try:
            os.mkdir(serving_logs_dir)
        except FileExistsError:
            pass

        write_mode = "w" if self.log_new_file else "a"
        self.log_new_file = False
        log_file_name = f"{serving_logs_dir}/log_{servable_config.servable_name}_device{self.device_id}" \
                        f"_version{self.version_number}.log"
        with open(log_file_name, write_mode) as fp:
            sub = subprocess.Popen(args=args, shell=False, stdout=fp, stderr=fp)
        return sub.pid
