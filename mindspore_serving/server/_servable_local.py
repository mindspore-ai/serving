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
import os
import random

import sys
import subprocess

from mindspore_serving import log as logger
from mindspore_serving.server.common import check_type, get_abs_path
from mindspore_serving.server.worker import get_newest_version_number
from mindspore_serving.server._servable_common import ServableContextDataBase


def _get_device_type(target_device_type, enable_lite):
    """Get device type supported, this will load libmindspore.so or libmindspore-lite.so"""
    # Get Device type: Ascend, Gpu, Cpu
    args = f"{sys.executable} -c 'from mindspore_serving._mindspore_serving import Worker_;" \
           f"device_type=Worker_.get_device_type(\"{target_device_type}\", {enable_lite});" \
           f"print(\"#get_device_type_result=\", device_type, \"#\", sep=\"\")'"
    process = subprocess.Popen(args=args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()
    result = process.stdout.read().decode("utf-8")
    prefix = "#get_device_type_result="
    index = result.find(prefix)
    if index < 0:
        raise RuntimeError(f"Failed to get device type")
    index += len(prefix)
    end_index = result.find("#", index)
    device_type = result[index:end_index]
    return device_type


def _all_reuse_device():
    """Get device type supported, this will load libmindspore.so or libmindspore-lite.so"""
    # Whether allow reuse device, for Ascend910 return False, other return True
    args = f"{sys.executable} -c 'from mindspore_serving._mindspore_serving import Worker_;" \
           f"reuse_flag=Worker_.support_reuse_device();" \
           f"print(\"#get_reuse_flag_result=\", reuse_flag, \"#\", sep=\"\")'"
    process = subprocess.Popen(args=args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()
    result = process.stdout.read().decode("utf-8")
    prefix = "#get_reuse_flag_result="
    index = result.find(prefix)
    if index < 0:
        raise RuntimeError(f"Failed to get device type")
    index += len(prefix)
    end_index = result.find("#", index)
    # pylint: disable=simplifiable-if-expression
    reuse_flag = True if result[index:end_index] == 'True' else False
    return reuse_flag


class ServableStartConfig:
    r"""
    Servable startup configuration.

    For more detail, please refer to
    `MindSpore-based Inference Service Deployment
    <https://www.mindspore.cn/serving/docs/zh-CN/r2.0/serving_example.html>`_ and
    `Servable Provided Through Model Configuration
    <https://www.mindspore.cn/serving/docs/zh-CN/r2.0/serving_model.html>`_.

    Args:
        servable_directory (str): The directory where the servable is located in. There expects to has a directory
            named `servable_name`.
        servable_name (str): The servable name.
        device_ids (Union[int, list[int], tuple[int]], optional): The device list the model loads into and runs in.
            Used when device type is Nvidia GPU, Ascend 310/310P/910. Default None.
        version_number (int, optional): Servable version number to be loaded. The version number should be a positive
            integer, starting from 1, and 0 means to load the latest version. Default: 0.
        device_type (str, optional): Target device type for model deployment. Currently supports "Ascend", "GPU", "CPU"
            and None. Default: None.

            - "Ascend": the platform expected to be Ascend 310/310P/910, etc.
            - "GPU": the platform expected to be Nvidia GPU.
            - "CPU": the platform expected to be CPU.
            - None: the platform is determined by the MindSpore environment.

        num_parallel_workers (int, optional): The number of processes that process python tasks, at least the number
            of device cards used specified by the parameter device_ids. It will be adjusted to the number of device
            cards when it is less than the number of device cards. The value should be in range [0,64]. Default: 0.
        dec_key (bytes, optional): Byte type key used for decryption. The valid length is 16, 24, or 32. Default: None.
        dec_mode (str, optional): Specifies the decryption mode, take effect when dec_key is set.
            Option: 'AES-GCM' or 'AES-CBC'. Default: 'AES-GCM'.

    Raises:
        RuntimeError: The type or value of the parameters are invalid.
    """

    def __init__(self, servable_directory, servable_name, device_ids=None, version_number=0, device_type=None,
                 num_parallel_workers=0, dec_key=None, dec_mode='AES-GCM'):
        super(ServableStartConfig, self).__init__()
        check_type.check_str("servable_directory", servable_directory)
        logger.info(f"input servable directory: {servable_directory}")
        servable_directory = get_abs_path(servable_directory)
        logger.info(f"absolute servable directory: {servable_directory}")

        check_type.check_str("servable_name", servable_name)
        check_type.check_int("version_number", version_number, 0)
        check_type.check_int("num_parallel_workers", num_parallel_workers, 0, 64)
        if dec_key is not None:
            if not isinstance(dec_key, bytes):
                raise RuntimeError(f"Parameter 'dec_key' should be bytes, but actually {type(dec_key)}")
            if not dec_key:
                raise RuntimeError(f"Parameter 'dec_key' should not be empty bytes")
            if len(dec_key) not in (16, 24, 32):
                raise RuntimeError(f"Parameter 'dec_key' length {len(dec_key)} expected to be 16, 24 or 32")
        check_type.check_str("dec_mode", dec_mode)
        if dec_mode not in ('AES-GCM', 'AES-CBC'):
            raise RuntimeError(f"Parameter 'dec_mode' expected to be 'AES-GCM' or 'AES-CBC'")

        self.servable_directory_ = servable_directory
        self.servable_name_ = servable_name
        self.version_number_ = version_number

        if device_ids is None:
            device_ids = []
        device_ids = check_type.check_and_as_int_tuple_list("device_ids", device_ids, 0)

        if device_type is not None:
            check_type.check_str("device_type", device_type)
        else:
            device_type = "None"

        if device_type.lower() != "none":
            if device_type.lower() not in ("ascend", "gpu", "cpu"):
                raise RuntimeError(f"Unsupported device type '{device_type}', only support 'Ascend', 'GPU', 'CPU' "
                                   f"and None, case ignored")
        # else device_type is None
        # if device_ids is empty, and there are models declared, Cpu target should be support
        # if device_ids is not empty, and there are no models declared, use no device resources
        # if device_ids is not empty, and there are models declared, final device_type depend on inference package
        self.device_ids_ = device_ids
        if not device_ids and not num_parallel_workers:
            self.num_parallel_workers_ = 1
        else:
            self.num_parallel_workers_ = num_parallel_workers
        self.device_type_ = device_type.lower()
        self.dec_key_ = dec_key
        self.dec_mode_ = dec_mode

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

    @property
    def dec_key(self):
        return self.dec_key_

    @property
    def dec_mode(self):
        return self.dec_mode_

    @property
    def num_parallel_workers(self):
        return self.num_parallel_workers_

    def _check_device_type(self, enable_lite):
        """Check whether the device type is valid"""
        device_type = self.device_type_
        if device_type.lower() != "none":
            if device_type.lower() not in ("ascend", "gpu", "cpu"):
                raise RuntimeError(f"Unsupported device type '{device_type}', only support 'Ascend', 'GPU', 'CPU' "
                                   f"and None, case ignored")
            default_device = _get_device_type(None, enable_lite)
            support_cpu = _get_device_type("cpu", enable_lite)
            if support_cpu and support_cpu != default_device:
                support_device = f"None, '{default_device}' or '{support_cpu}'"
            else:
                support_device = f"None or '{default_device}'"
            if not _get_device_type(device_type, enable_lite):
                raise RuntimeError(f"The device type '{device_type}' of servable name {self.servable_name} "
                                   f"is inconsistent with current running environment, supported device type: "
                                   f"{support_device}")


class DeployConfig:
    """Deployment configuration of one version for the servable"""

    def __init__(self, version_number, device_ids, num_parallel_workers=0, dec_key=None, dec_mode='AES-GCM'):
        check_type.check_int("version_number", version_number)
        if device_ids is None:
            device_ids = []
        device_ids = check_type.check_and_as_int_tuple_list("device_ids", device_ids, 0)
        check_type.check_int("num_parallel_workers", num_parallel_workers, 0)

        if dec_key is not None:
            if not isinstance(dec_key, bytes):
                raise RuntimeError(f"Parameter 'dec_key' should be bytes, but actually {type(dec_key)}")
            if not dec_key:
                raise RuntimeError(f"Parameter 'dec_key' should not be empty bytes")
            if len(dec_key) not in (16, 24, 32):
                raise RuntimeError(f"Parameter 'dec_key' length {len(dec_key)} expected to be 16, 24 or 32")
        check_type.check_str("dec_mode", dec_mode)
        if dec_mode not in ('AES-GCM', 'AES-CBC'):
            raise RuntimeError(f"Parameter 'dec_mode' expected to be 'AES-GCM' or 'AES-CBC'")

        self.version_number = version_number
        self.device_ids = set(device_ids)
        if not device_ids and not num_parallel_workers:
            self.num_parallel_workers = 1
        else:
            self.num_parallel_workers = num_parallel_workers
        self.dec_key = dec_key
        self.dec_mode = dec_mode


class ServableStartConfigGroup:
    """Servable start config for one servable with multi version deployment configs"""

    def __init__(self, servable_directory, servable_name, device_type=None):
        check_type.check_str("servable_directory", servable_directory)
        logger.info(f"input servable directory: {servable_directory}")
        servable_directory = get_abs_path(servable_directory)
        logger.info(f"absolute servable directory: {servable_directory}")

        check_type.check_str("servable_name", servable_name)

        if device_type is not None:
            check_type.check_str("device_type", device_type)
        else:
            device_type = "None"

        self.servable_directory = servable_directory
        self.servable_name = servable_name
        self.device_type = device_type
        self.check_servable_location()
        self.deploy_configs = {}
        self.newest_version_number = get_newest_version_number(servable_directory, servable_name)
        logger.info(f"The newest version number of servable {self.servable_name} is {self.newest_version_number}, "
                    f"servable directory: {self.servable_directory}")

    def check_servable_location(self):
        """Check the validity of parameters servable_directory and servable_name"""
        config_dir = os.path.join(self.servable_directory, self.servable_name)
        if not os.path.isdir(config_dir):
            raise RuntimeError(
                f"Check servable config failed, directory '{config_dir}' not exist, servable "
                f"directory '{self.servable_directory}', servable name '{self.servable_name}'")

        config_file = os.path.join(config_dir, "servable_config.py")
        if not os.path.isfile(config_file):
            raise RuntimeError(
                f"Check servable config failed, file '{config_file}' not exist,  servable directory "
                f"'{self.servable_directory}', servable name '{self.servable_name}'")

    def append_deploy(self, deploy_config):
        """Append one deployment configuration of one version for the servable"""
        if not isinstance(deploy_config, DeployConfig):
            raise RuntimeError(f"Parameter 'deploy_config' should be type of DeployConfig")
        if deploy_config.version_number == 0:
            deploy_config.version_number = self.newest_version_number

        if deploy_config.version_number not in self.deploy_configs:
            self.deploy_configs[deploy_config.version_number] = deploy_config
        else:
            last_config = self.deploy_configs[deploy_config.version_number]
            last_config.device_ids = last_config.device_ids.union(deploy_config.device_ids)
            if last_config.dec_key != deploy_config.dec_key or last_config.dec_mode != deploy_config.dec_mode:
                raise RuntimeError(f"The dec key or dec mode of servable name {self.servable_name} is different in "
                                   f"multiple configurations.")
            if deploy_config.num_parallel_workers > last_config.num_parallel_workers:
                last_config.num_parallel_workers = deploy_config.num_parallel_workers

    def export_as_start_configs(self):
        """Export the configuration as list of ServableStartConfig"""
        configs = []
        for config in self.deploy_configs.values():
            start_config = ServableStartConfig(servable_directory=self.servable_directory,
                                               servable_name=self.servable_name,
                                               device_ids=tuple(config.device_ids),
                                               version_number=config.version_number,
                                               device_type=self.device_type,
                                               num_parallel_workers=config.num_parallel_workers,
                                               dec_key=config.dec_key, dec_mode=config.dec_mode)
            configs.append(start_config)
        return configs


def _check_and_merge_config(configs):
    """Merge ServableStartConfig with the same version number"""
    start_config_groups = {}
    for config in configs:
        if not isinstance(config, ServableStartConfig):
            continue
        if config.servable_name in start_config_groups:
            if config.servable_directory != start_config_groups[config.servable_name].servable_directory:
                raise RuntimeError(
                    f"The servable directory of servable name {config.servable_name} is different in"
                    f" multiple configurations, servable directory: "
                    f"{config.servable_directory} and {start_config_groups[config.servable_name].servable_directory}")
        else:
            config_group = ServableStartConfigGroup(config.servable_directory, config.servable_name, config.device_type)
            start_config_groups[config.servable_name] = config_group

        deploy_config = DeployConfig(config.version_number, config.device_ids, config.num_parallel_workers,
                                     config.dec_key, config.dec_mode)
        start_config_groups[config.servable_name].append_deploy(deploy_config)

    return start_config_groups


def merge_config(configs):
    """Merge ServableStartConfig with the same version number"""
    start_config_groups = _check_and_merge_config(configs)
    configs_ret = []

    for config_group in start_config_groups.values():
        start_configs = config_group.export_as_start_configs()
        configs_ret.extend(start_configs)

    allow_reuse_device = None
    device_ids_used = set()
    for config in configs_ret:
        for device_id in config.device_ids:
            if device_id in device_ids_used:
                if allow_reuse_device is None:
                    allow_reuse_device = _all_reuse_device()
                if not allow_reuse_device:
                    raise RuntimeError(f"Ascend 910 device id {device_id} is used repeatedly in servable "
                                       f"{config.servable_name}")
            device_ids_used.add(device_id)
    for config in configs:
        if not isinstance(config, ServableStartConfig):
            configs_ret.append(config)
    return configs_ret


class ServableContextData(ServableContextDataBase):
    """Used to startup servable process"""

    def __init__(self, servable_config, device_id, master_address, enable_lite):
        super(ServableContextData, self).__init__()
        self.servable_config = servable_config
        self.device_id = device_id
        self.master_address = master_address
        self.log_new_file = True
        self.enable_lite = enable_lite

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
        config = self.servable_config
        device_type = config.device_type
        if device_type is None:
            device_type = "None"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        py_script = os.path.join(script_dir, "start_worker.py")

        if self.servable_config.dec_key:
            pipe_file = f"serving_temp_dec_{config.servable_name}_device{self.device_id}_" \
                        f"{random.randrange(1000000, 9999999)}"
            os.mkfifo(pipe_file)
        else:
            pipe_file = 'None'
        enable_lite_str = "True" if self.enable_lite else "False"

        arg = f"{python_exe} {py_script} " \
              f"--servable_directory={config.servable_directory} " \
              f"--servable_name={config.servable_name} " \
              f"--version_number={config.version_number} " \
              f"--device_type={device_type} " \
              f"--device_id={self.device_id} " \
              f"--master_address={self.master_address} " \
              f"--enable_lite={enable_lite_str} " \
              f"--dec_key_pipe_file={pipe_file} " \
              f"--dec_mode={config.dec_mode} " \
              f"--listening_master=True"

        args = arg.split(" ")

        serving_logs_dir = "serving_logs"
        try:
            os.mkdir(serving_logs_dir)
        except FileExistsError:
            pass

        write_mode = "w" if self.log_new_file else "a"
        self.log_new_file = False
        log_file_name = f"{serving_logs_dir}/log_{config.servable_name}_device{self.device_id}" \
                        f"_version{self.version_number}.log"
        with open(log_file_name, write_mode) as fp:
            sub = subprocess.Popen(args=args, shell=False, stdout=fp, stderr=fp)
        if self.servable_config.dec_key:
            with open(pipe_file, "wb") as fp:
                fp.write(self.servable_config.dec_key)
        return sub


class ServableExtraContextData(ServableContextDataBase):
    """Used to startup servable process"""

    def __init__(self, servable_config, master_address, index, device_ids_empty, enable_lite):
        super(ServableExtraContextData, self).__init__()
        self.servable_config = servable_config
        self.master_address = master_address
        self.log_new_file = True
        self.index = index
        self.device_ids_empty = device_ids_empty
        self.enable_lite = enable_lite

    @property
    def servable_name(self):
        return self.servable_config.servable_name

    @property
    def version_number(self):
        return self.servable_config.version_number

    def own_device(self):
        """Whether the worker occupy device"""
        return False

    def to_string(self):
        """For logging"""
        return f"servable name: {self.servable_name}, version: {self.version_number}, extra: {self.index}"

    def new_worker_process(self):
        """Start worker process to provide servable"""
        python_exe = sys.executable
        config = self.servable_config
        script_dir = os.path.dirname(os.path.abspath(__file__))
        py_script = os.path.join(script_dir, "start_extra_worker.py")

        if config.dec_key:
            pipe_file = f"serving_temp_dec_{config.servable_name}_index{self.index}_" \
                        f"{random.randrange(1000000, 9999999)}"
            os.mkfifo(pipe_file)
        else:
            pipe_file = 'None'

        device_type = config.device_type
        if device_type is None:
            device_type = "None"

        enable_lite_str = "True" if self.enable_lite else "False"
        arg = f"{python_exe} {py_script} " \
              f"--servable_directory={config.servable_directory} " \
              f"--servable_name={config.servable_name} " \
              f"--version_number={config.version_number} " \
              f"--device_type={device_type} " \
              f"--device_ids_empty={self.device_ids_empty} " \
              f"--index={self.index} " \
              f"--enable_lite={enable_lite_str} " \
              f"--master_address={self.master_address} " \
              f"--dec_key_pipe_file={pipe_file} " \
              f"--dec_mode={config.dec_mode} " \
              f"--listening_master=True"
        args = arg.split(" ")

        serving_logs_dir = "serving_logs"
        try:
            os.mkdir(serving_logs_dir)
        except FileExistsError:
            pass

        write_mode = "w" if self.log_new_file else "a"
        self.log_new_file = False
        log_file_name = f"{serving_logs_dir}/log_{config.servable_name}_extra{self.index}" \
                        f"_version{self.version_number}.log"
        with open(log_file_name, write_mode) as fp:
            sub = subprocess.Popen(args=args, shell=False, stdout=fp, stderr=fp)
        if self.servable_config.dec_key:
            with open(pipe_file, "wb") as fp:
                fp.write(self.servable_config.dec_key)
        return sub
