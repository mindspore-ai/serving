#!/bin/bash

export GLOG_v=1
export DEVICE_ID=1

MINDSPORE_INSTALL_PATH=$1
ENV_DEVICE_ID=$DEVICE_ID
CURRPATH=$(cd "$(dirname $0)" || exit; pwd)
CURRUSER=$(whoami)
PROJECT_PATH=${CURRPATH}/../../../
echo "MINDSPORE_INSTALL_PATH:"  ${MINDSPORE_INSTALL_PATH}
echo "ENV_DEVICE_ID:" ${ENV_DEVICE_ID}
echo "CURRPATH:"  ${CURRPATH}
echo "CURRUSER:"  ${CURRUSER}
echo "PROJECT_PATH:"  ${PROJECT_PATH}

export LD_LIBRARY_PATH=${MINDSPORE_INSTALL_PATH}/lib:/usr/local/python/python375/lib/:${LD_LIBRARY_PATH}
#export PYTHONPATH=${MINDSPORE_INSTALL_PATH}/:${PYTHONPATH}

echo "LD_LIBRARY_PATH: " ${LD_LIBRARY_PATH}
echo "PYTHONPATH: " ${PYTHONPATH}
echo "-------------show MINDSPORE_INSTALL_PATH----------------"
ls -l ${MINDSPORE_INSTALL_PATH}
echo "------------------show /usr/lib64/----------------------"
ls -l /usr/local/python/python375/lib/

clean_pid()
{
  ps aux | grep 'master_with_worker.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
  if [ $? -ne 0 ]
  then
    echo "clean pip failed"
  fi
  sleep 6
}

prepare_model()
{
  echo "### begin to generate mode for serving test ###"
  python3 add_model.py &> add_model.log
  echo "### end to generate mode for serving test ###"
  result=`find . -name  tensor_add.mindir | wc -l`
  if [ ${result} -ne 1 ]
  then
    cat add_model.log
    echo "### generate model for serving test failed ###" && exit 1
    clean_pid
  fi
  rm -rf add
  mkdir add
  mkdir add/1
  mv *.mindir ${CURRPATH}/add/1/
  cp servable_config.py add/
}

start_service()
{
  echo "### start serving service ###"
  unset http_proxy https_proxy
  python3 master_with_worker.py > service.log 2>&1 &
  if [ $? -ne 0 ]
  then
    echo "server faile to start."
  fi

  result=`grep -E 'Serving gRPC server start success, listening on 127.0.0.1:5500' service.log | wc -l`
  count=0
  while [[ ${result} -ne 1 && ${count} -lt 150 ]]
  do
    sleep 1
    count=$(($count+1))
    result=`grep -E 'Serving gRPC server start success, listening on 127.0.0.1:5500' service.log | wc -l`
  done

  if [ ${count} -eq 150 ]
  then
    clean_pid
    cat service.log
    echo "start serving service failed!" && exit 1
  fi
  echo "### start serving service end ###"
}

pytest_serving()
{
  unset http_proxy https_proxy
  echo "###  client start ###"
  python3  client.py > client.log 2>&1
  if [ $? -ne 0 ]
  then
    clean_pid
    cat client.log
    echo "client faile to start." && exit 1
  fi
  echo "### client end ###"
}

test_bert_model()
{
  start_service
  pytest_serving
  cat client.log
  clean_pid
}

echo "-----serving start-----"
rm -rf serving *.log *.mindir *.dat ${CURRPATH}/add ${CURRPATH}/kernel_meta
prepare_model
test_bert_model
