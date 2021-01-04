#!/bin/bash
# Copyright 2019 Huawei Technologies Co., Ltd
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
set -e
BASEPATH=$(cd "$(dirname "$0")"; pwd)
PROJECT_PATH=${BASEPATH}/../../..
if [ $BUILD_PATH ];then
  echo "BUILD_PATH = $BUILD_PATH"
else
  BUILD_PATH=${PROJECT_PATH}/build
  echo "BUILD_PATH = $BUILD_PATH"
fi
cd ${BUILD_PATH}/mindspore_serving/tests/ut/cpp

if [ $# -gt 0 ]; then
  ./serving_ut --gtest_filter=$1
else
  ./serving_ut
fi
RET=$?
cd -

exit ${RET}
