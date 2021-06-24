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
"""export Lenet for mnist dataset"""

import os
from shutil import copyfile
from lenet.export import export_lenet

if __name__ == '__main__':
    export_lenet()
    dst_dir = '../lenet/1'
    try:
        os.mkdir(dst_dir)
    except OSError:
        pass
    dst_file = os.path.join(dst_dir, 'lenet.mindir')
    copyfile('lenet.mindir', dst_file)
