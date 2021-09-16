#!/bin/bash
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
set -e
BASEPATH=$(
  cd "$(dirname "$0")"
  pwd
)

PROJECT_PATH=$(
  cd ${BASEPATH}/../../..
  pwd
)

BUILD_PKG=${PROJECT_PATH}/build/package

export PYTHONPATH=${BUILD_PKG}:${PROJECT_PATH}/tests/ut/python:$PYTHONPATH
export LD_LIBRARY_PATH=${BUILD_PKG}/tests/mindspore/lib:${LD_LIBRARY_PATH}

echo "PYTHONPATH=$PYTHONPATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
export GLOG_v=1

unset http_proxy
unset https_proxy

rm -rf cov_output htmlcov .coverage

# run python ut
pytest -v ${PROJECT_PATH}/tests/ut/python/tests/ --cov=${BUILD_PKG}/mindspore_serving --cov-config=${BASEPATH}/cov_config --cov-report=html --cov-branch
# run cpp ut
bash ../cpp/runtest.sh

mkdir cov_output && cd cov_output
lcov --capture --directory ${PROJECT_PATH}/build/mindspore_serving/ --output-file coverage.info;
lcov --extract coverage.info '*/ccsrc/*' -o coverage.info;
genhtml coverage.info --output-directory ./ --sort --legend
