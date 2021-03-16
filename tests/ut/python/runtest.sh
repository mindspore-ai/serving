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
BASEPATH=$(
  cd "$(dirname "$0")"
  pwd
)
PROJECT_PATH=${BASEPATH}/../../..
if [ $BUILD_PATH ]; then
  echo "BUILD_PATH = $BUILD_PATH"
else
  BUILD_PATH=${PROJECT_PATH}/build
  echo "BUILD_PATH = $BUILD_PATH"
fi
cd ${BUILD_PATH}/mindspore_serving/tests/ut/python
rm -rf mindspore_serving
mkdir -p mindspore_serving/proto
cp ../mindspore_serving/proto/ms_service*.py mindspore_serving/proto/
cp _mindspore_serving*.so mindspore_serving/
cp -r ${PROJECT_PATH}/mindspore_serving/master mindspore_serving/
cp -r ${PROJECT_PATH}/mindspore_serving/worker mindspore_serving/
cp -r ${PROJECT_PATH}/mindspore_serving/common mindspore_serving/
cp -r ${PROJECT_PATH}/mindspore_serving/client mindspore_serving/
cp ${PROJECT_PATH}/mindspore_serving/*.py mindspore_serving/

export PYTHONPATH=${BUILD_PATH}/mindspore_serving/tests/ut/python:${PROJECT_PATH}/tests/ut/python:$PYTHONPATH
export LD_LIBRARY_PATH=${BUILD_PATH}/mindspore_serving/tests/ut/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${BUILD_PATH}/mindspore_serving/tests/ut/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${BUILD_PATH}/mindspore_serving/tests/ut/python:${LD_LIBRARY_PATH}

echo "PYTHONPATH=$PYTHONPATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

unset http_proxy
unset https_proxy

function clear_port()
{
  PROCESS=`netstat -nlp | grep :$1 | awk '{print $7}' | awk -F"/" '{print $1}'`
  for i in $PROCESS
     do
     echo "Kill the process [ $i ]"
     kill -9 $i
  done
}

port_list=(5500 6200 7000 7001 7002 7003 7004 7005 7006 7007)
for port in ${port_list[*]}; do
  clear_port ${port}
done

cd -
cd ${PROJECT_PATH}/tests/ut/python/tests/
if [ $# -gt 0 ]; then
  pytest -v . -k "$1"
else
  pytest -v .
fi

RET=$?
cd -

exit ${RET}
