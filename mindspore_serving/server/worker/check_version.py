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
"""version and config check"""
import os
import sys
import subprocess
from pathlib import Path
from packaging import version
from mindspore_serving import log as logger


class AscendEnvChecker:
    """ascend environment check"""

    def __init__(self):
        atlas_nnae_version = "/usr/local/Ascend/nnae/latest/compiler/version.info"
        atlas_toolkit_version = "/usr/local/Ascend/ascend-toolkit/latest/compiler/version.info"
        hisi_fwk_version = "/usr/local/Ascend/latest/compiler/version.info"
        if os.path.exists(atlas_nnae_version):
            # atlas default path
            self.fwk_path = "/usr/local/Ascend/nnae/latest"
            self.op_impl_path = "/usr/local/Ascend/nnae/latest/opp/built-in/op_impl/ai_core/tbe"
            self.tbe_path = self.fwk_path + "/lib64"
            self.cce_path = self.fwk_path + "/compiler/ccec_compiler/bin"
            self.fwk_version = atlas_nnae_version
            self.op_path = "/usr/local/Ascend/nnae/latest/opp"
            self.aicpu_path = "/usr/local/Ascend/nnae/latest"
        elif os.path.exists(atlas_toolkit_version):
            # atlas default path
            self.fwk_path = "/usr/local/Ascend/ascend-toolkit/latest"
            self.op_impl_path = "/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe"
            self.tbe_path = self.fwk_path + "/lib64"
            self.cce_path = self.fwk_path + "/compiler/ccec_compiler/bin"
            self.fwk_version = atlas_toolkit_version
            self.op_path = "/usr/local/Ascend/ascend-toolkit/latest/opp"
            self.aicpu_path = "/usr/local/Ascend/ascend-toolkit/latest"
        elif os.path.exists(hisi_fwk_version):
            # hisi default path
            self.fwk_path = "/usr/local/Ascend/latest"
            self.op_impl_path = "/usr/local/Ascend/latest/opp/built-in/op_impl/ai_core/tbe"
            self.tbe_path = self.fwk_path + "/lib64"
            self.cce_path = self.fwk_path + "/compiler/ccec_compiler/bin"
            self.fwk_version = hisi_fwk_version
            self.op_path = "/usr/local/Ascend/latest/opp"
            self.aicpu_path = "/usr/local/Ascend/latest"
        else:
            # custom or unknown environment
            self.fwk_path = ""
            self.op_impl_path = ""
            self.tbe_path = ""
            self.cce_path = ""
            self.fwk_version = ""
            self.op_path = ""
            self.aicpu_path = ""

        # env
        self.path = os.getenv("PATH")
        self.python_path = os.getenv("PYTHONPATH")
        self.ld_lib_path = os.getenv("LD_LIBRARY_PATH")
        self.ascend_opp_path = os.getenv("ASCEND_OPP_PATH")
        self.ascend_aicpu_path = os.getenv("ASCEND_AICPU_PATH")

        # check content
        self.path_check = "/compiler/ccec_compiler/bin"
        self.python_path_check = "opp/built-in/op_impl/ai_core/tbe"
        self.ld_lib_path_check_fwk = "/lib64"
        self.ld_lib_path_check_addons = "/add-ons"
        self.ascend_opp_path_check = "/op"
        self.v = ""

    def check_env(self, e):
        """check system env"""
        self._check_env()
        raise e

    def set_env(self):
        """set env: LD_LIBRARY_PATH, PATH, ASCEND_OPP_PATH"""
        if not self.tbe_path:
            self._check_env()
            return

        if Path(self.tbe_path).is_dir():
            if os.getenv('LD_LIBRARY_PATH'):
                os.environ['LD_LIBRARY_PATH'] = self.tbe_path + ":" + os.environ['LD_LIBRARY_PATH']
            else:
                os.environ['LD_LIBRARY_PATH'] = self.tbe_path
        else:
            logger.warning(f"No such directory: {self.tbe_path}, Please check if Ascend 910 AI software package is "
                           f"installed correctly.")

        if Path(self.op_impl_path).is_dir():
            # python path for sub process
            if os.getenv('PYTHONPATH'):
                os.environ['PYTHONPATH'] = self.op_impl_path + ":" + os.environ['PYTHONPATH']
            else:
                os.environ['PYTHONPATH'] = self.op_impl_path
            # sys path for this process
            sys.path.append(self.op_impl_path)

            os.environ['TBE_IMPL_PATH'] = self.op_impl_path
        else:
            logger.warning(
                f"No such directory: {self.op_impl_path}, Please check if Ascend AI software package (Ascend Data "
                "Center Solution) is installed correctly.")
            return

        if Path(self.cce_path).is_dir():
            os.environ['PATH'] = self.cce_path + ":" + os.environ['PATH']
        else:
            logger.warning(
                f"No such directory: {self.cce_path}, Please check if Ascend AI software package (Ascend Data Center "
                "Solution) is installed correctly.")
            return

        if self.op_path is None:
            pass
        elif Path(self.op_path).is_dir():
            os.environ['ASCEND_OPP_PATH'] = self.op_path
        else:
            logger.warning(
                f"No such directory: {self.op_path}, Please check if Ascend AI software package (Ascend Data Center "
                "Solution) is installed correctly.")
            return

        if self.aicpu_path is None:
            pass
        elif Path(self.aicpu_path).is_dir():
            os.environ['ASCEND_AICPU_PATH'] = self.aicpu_path
        else:
            logger.warning(
                f"No such directory: {self.aicpu_path}, Please check if Ascend AI software package (Ascend Data Center"
                " Solution) is installed correctly.")
            return

    def try_set_env_lib(self):
        """try set env but with no warning: LD_LIBRARY_PATH"""
        if Path(self.tbe_path).is_dir():
            if os.getenv('LD_LIBRARY_PATH'):
                os.environ['LD_LIBRARY_PATH'] = self.tbe_path + ":" + os.environ['LD_LIBRARY_PATH']
            else:
                os.environ['LD_LIBRARY_PATH'] = self.tbe_path

    def _check_env(self):
        """ascend dependence path check"""
        if self.path is None or self.path_check not in self.path:
            logger.warning("Can not find ccec_compiler(need by mindspore-ascend), please check if you have set env "
                           "PATH, you can reference to the installation guidelines https://www.mindspore.cn/install")

        if self.python_path is None or self.python_path_check not in self.python_path:
            logger.warning(
                "Can not find tbe op implement(need by mindspore-ascend), please check if you have set env "
                "PYTHONPATH, you can reference to the installation guidelines "
                "https://www.mindspore.cn/install")

        if self.ld_lib_path is None or not (self.ld_lib_path_check_fwk in self.ld_lib_path and
                                            self.ld_lib_path_check_addons in self.ld_lib_path):
            logger.warning("Can not find driver so(need by mindspore-ascend), please check if you have set env "
                           "LD_LIBRARY_PATH, you can reference to the installation guidelines "
                           "https://www.mindspore.cn/install")

        if self.ascend_opp_path is None or self.ascend_opp_path_check not in self.ascend_opp_path:
            logger.warning(
                "Can not find opp path (need by mindspore-ascend), please check if you have set env ASCEND_OPP_PATH, "
                "you can reference to the installation guidelines https://www.mindspore.cn/install")


class GPUEnvChecker():
    """GPU environment check."""

    def __init__(self):
        self.version = ["10.1"]
        # env
        self.path = os.getenv("PATH")
        self.ld_lib_path = os.getenv("LD_LIBRARY_PATH")

        # check
        self.v = "0"
        self.cuda_lib_path = self._get_lib_path("libcu")
        self.cuda_bin_path = self._get_bin_path("cuda")

    def _get_bin_path(self, bin_name):
        """Get bin path by bin name."""
        if bin_name == "cuda":
            return self._get_cuda_bin_path()
        return []

    def _get_cuda_bin_path(self):
        """Get cuda bin path by lib path."""
        path_list = []
        for path in self.cuda_lib_path:
            path = os.path.abspath(path.strip() + "/bin/")
            if Path(path).is_dir():
                path_list.append(path)
        return list(set(path_list))

    def _get_nvcc_version(self, is_set_env):
        """Get cuda version by nvcc command."""
        nvcc_result = subprocess.run(["nvcc --version | grep release"],
                                     timeout=3, text=True, capture_output=True, check=False, shell=True)
        if nvcc_result.returncode:
            if not is_set_env:
                for path in self.cuda_bin_path:
                    if Path(path + "/nvcc").is_file():
                        os.environ['PATH'] = path + ":" + os.environ['PATH']
                        return self._get_nvcc_version(True)
            return ""
        result = nvcc_result.stdout
        for line in result.split('\n'):
            if line:
                return line.strip().split("release")[1].split(",")[0].strip()
        return ""

    def check_env(self):
        """Check cuda version."""
        version_match = False
        for path in self.cuda_lib_path:
            version_file = path + "/version.txt"
            if not Path(version_file).is_file():
                continue
            if self._check_version(version_file):
                version_match = True
                break
        if not version_match:
            if self.v == "0":
                logger.warning("Cuda version file version.txt is not found, please confirm that the correct "
                               "cuda version has been installed, you can refer to the "
                               "installation guidelines: https://www.mindspore.cn/install")
            else:
                logger.warning(f"MindSpore version and cuda version {self.v} does not match, "
                               "please refer to the installation guide for version matching "
                               "information: https://www.mindspore.cn/install")
        nvcc_version = self._get_nvcc_version(False)
        if nvcc_version and (nvcc_version not in self.version):
            logger.warning(f"MindSpore version and nvcc(cuda bin) version {nvcc_version} "
                           "does not match, please refer to the installation guide for version matching "
                           "information: https://www.mindspore.cn/install")

    def _check_version(self, version_file):
        """Check cuda version by version.txt."""
        v = self._read_version(version_file)
        v = version.parse(v)
        v_str = str(v.major) + "." + str(v.minor)
        if v_str not in self.version:
            return False
        return True

    def _get_lib_path(self, lib_name):
        """Get gpu lib path by ldd command."""
        path_list = []
        current_path = os.path.split(os.path.realpath(__file__))[0]
        mindspore_path = os.path.dirname(os.path.dirname(current_path)) + "/mindspore"
        ldd_result = subprocess.run(["ldd " + mindspore_path + "/_c_expression*.so* | grep " + lib_name],
                                    timeout=3, text=True, capture_output=True, check=False, shell=True)
        if ldd_result.returncode:
            logger.warning(f"{lib_name} so(need by mndspore-gpu) is not found, please confirm that "
                           f"_c_experssion.so depend on {lib_name}, "
                           f"and _c_expression.so in directory:{mindspore_path}")
            return path_list
        result = ldd_result.stdout
        for i in result.split('\n'):
            path = i.partition("=>")[2]
            if path.lower().find("not found") > 0:
                logger.warning(f"Cuda {self.version} version(need by mindspore-gpu) is not found, please confirm "
                               "that the path of cuda is set to the env LD_LIBRARY_PATH, please refer to the "
                               "installation guidelines: https://www.mindspore.cn/install")
                continue
            path = path.partition(lib_name)[0]
            if path:
                path_list.append(os.path.abspath(path.strip() + "../"))
        return list(set(path_list))

    def _read_version(self, file_path):
        """Get gpu version info in version.txt."""
        with open(file_path, 'r') as f:
            all_info = f.readlines()
            for line in all_info:
                if line.startswith("CUDA Version"):
                    self.v = line.strip().split("CUDA Version")[1]
                    return self.v
        return self.v


def check_version_and_env_config(device_type):
    """check version and env config"""
    if device_type == "Ascend":
        env_checker = AscendEnvChecker()
        try:
            env_checker.set_env()
        except ImportError as e:
            env_checker.check_env(e)
    elif device_type == "Gpu":
        env_checker = GPUEnvChecker()
        env_checker.check_env()
    elif device_type == "Cpu":
        pass


def check_version_and_try_set_env_lib():
    """check version and try set env LD_LIBRARY_PATH"""
    env_checker = AscendEnvChecker()
    env_checker.try_set_env_lib()
