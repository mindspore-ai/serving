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
from pathlib import Path
from mindspore_serving import log as logger


class AscendEnvChecker:
    """ascend environment check"""

    def __init__(self):
        atlas_nnae_version = "/usr/local/Ascend/nnae/latest/fwkacllib/version.info"
        atlas_toolkit_version = "/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/version.info"
        hisi_fwk_version = "/usr/local/Ascend/fwkacllib/version.info"
        hisi_atc_version = "/usr/local/Ascend/atc/version.info"
        if os.path.exists(atlas_nnae_version):
            # atlas default path
            self.fwk_path = "/usr/local/Ascend/nnae/latest/fwkacllib"
            self.op_impl_path = "/usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe"
            self.tbe_path = self.fwk_path + "/lib64"
            self.cce_path = self.fwk_path + "/ccec_compiler/bin"
            self.fwk_version = atlas_nnae_version
            self.op_path = "/usr/local/Ascend/nnae/latest/opp"
        elif os.path.exists(atlas_toolkit_version):
            # atlas default path
            self.fwk_path = "/usr/local/Ascend/ascend-toolkit/latest/fwkacllib"
            self.op_impl_path = "/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe"
            self.tbe_path = self.fwk_path + "/lib64"
            self.cce_path = self.fwk_path + "/ccec_compiler/bin"
            self.fwk_version = atlas_toolkit_version
            self.op_path = "/usr/local/Ascend/ascend-toolkit/latest/opp"
        elif os.path.exists(hisi_fwk_version):
            # hisi default path
            self.fwk_path = "/usr/local/Ascend/fwkacllib"
            self.op_impl_path = "/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe"
            self.tbe_path = self.fwk_path + "/lib64"
            self.cce_path = self.fwk_path + "/ccec_compiler/bin"
            self.fwk_version = hisi_fwk_version
            self.op_path = "/usr/local/Ascend/opp"
        elif os.path.exists(hisi_atc_version):
            # hisi 310 default path
            self.fwk_path = "/usr/local/Ascend/atc"
            self.op_impl_path = "/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe"
            self.tbe_path = self.fwk_path + "/lib64"
            self.cce_path = self.fwk_path + "/ccec_compiler/bin"
            self.fwk_version = hisi_fwk_version
            self.op_path = "/usr/local/Ascend/opp"
        else:
            # custom or unknown environment
            self.fwk_path = ""
            self.op_impl_path = ""
            self.tbe_path = ""
            self.cce_path = ""
            self.fwk_version = ""
            self.op_path = ""

        # env
        self.path = os.getenv("PATH")
        self.python_path = os.getenv("PYTHONPATH")
        self.ld_lib_path = os.getenv("LD_LIBRARY_PATH")
        self.ascend_opp_path = os.getenv("ASCEND_OPP_PATH")

        # check content
        self.path_check = "/fwkacllib/ccec_compiler/bin"
        self.python_path_check = "opp/op_impl/built-in/ai_core/tbe"
        self.ld_lib_path_check_fwk = "/fwkacllib/lib64"
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

        try:
            # pylint: disable=unused-import
            import te
        # pylint: disable=broad-except
        except Exception:
            if Path(self.tbe_path).is_dir():
                if os.getenv('LD_LIBRARY_PATH'):
                    os.environ['LD_LIBRARY_PATH'] = self.tbe_path + ":" + os.environ['LD_LIBRARY_PATH']
                else:
                    os.environ['LD_LIBRARY_PATH'] = self.tbe_path
            else:
                logger.warning(f"No such directory: {self.tbe_path}, Please check if Ascend 910 AI software package is "
                               f"installed correctly.")

        if Path(self.op_impl_path).is_dir():
            sys.path.append(self.op_impl_path)
        else:
            logger.warning(f"No such directory: {self.op_impl_path}, Please check if Ascend 910 AI software package is "
                           f"installed correctly.")

        if Path(self.cce_path).is_dir():
            os.environ['PATH'] = self.cce_path + ":" + os.environ['PATH']
        else:
            logger.warning(f"No such directory: {self.cce_path}, Please check if Ascend 910 AI software package is "
                           f"installed correctly.")

        if self.op_path is None:
            pass
        elif Path(self.op_path).is_dir():
            os.environ['ASCEND_OPP_PATH'] = self.op_path
        else:
            logger.warning(f"No such directory: {self.op_path}, Please check if Ascend 910 AI software package is "
                           f"installed correctly.")

    def try_set_env_lib(self):
        """try set env but with no warning: LD_LIBRARY_PATH"""
        try:
            # pylint: disable=unused-import
            import te
        # pylint: disable=broad-except
        except Exception:
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


def check_version_and_env_config(device_type):
    """check version and env config"""
    if device_type == "Ascend":
        env_checker = AscendEnvChecker()
        try:
            env_checker.set_env()
        except ImportError as e:
            env_checker.check_env(e)
    elif device_type == "Gpu":
        pass
    elif device_type == "Cpu":
        pass


def check_version_and_try_set_env_lib():
    """check version and try set env LD_LIBRARY_PATH"""
    env_checker = AscendEnvChecker()
    env_checker.try_set_env_lib()
