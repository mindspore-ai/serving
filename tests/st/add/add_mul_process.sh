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

export LD_LIBRARY_PATH=${MINDSPORE_INSTALL_PATH}/lib:${LD_LIBRARY_PATH}
#export PYTHONPATH=${MINDSPORE_INSTALL_PATH}/:${PYTHONPATH}

echo "LD_LIBRARY_PATH: " ${LD_LIBRARY_PATH}
echo "PYTHONPATH: " ${PYTHONPATH}
echo "-------------show MINDSPORE_INSTALL_PATH----------------"
ls -l ${MINDSPORE_INSTALL_PATH}
echo "------------------show /usr/lib64/----------------------"
ls -l /usr/local/python/python375/lib/

clean_master_pid()
{
  ps aux | grep 'master.py' | grep ${CURRUSER} | grep -v grep
  if [ $? -eq 0 ]
  then
    ps aux | grep 'master.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -15
    if [ $? -ne 0 ]
    then
      echo "clean master pip failed"
    fi
    sleep 6
    ps aux | grep 'master.py' | grep ${CURRUSER} | grep -v grep
    if [ $? -eq 0 ]
    then
      ps aux | grep 'master.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
      echo "### master pid exist, clean master pip failed ###" & exit 1
    fi
    ps aux | grep 'worker.py' | grep ${CURRUSER} | grep -v grep
    if [ $? -eq 0 ]
    then
      ps aux | grep 'worker.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
      echo "### master pid is killed but worker pid exist ###" & exit 1
    fi
  fi
}

clean_worker_pid()
{
  ps aux | grep 'worker.py' | grep ${CURRUSER} | grep -v grep
  if [ $? -eq 0 ]
  then
    ps aux | grep 'worker.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
    if [ $? -eq 0 ]
    then
      echo "clean work pid failed"
    fi
  fi
}

prepare_model()
{
  echo "### begin to generate mode for serving test ###"
  cd export_model
  python3 add_model.py &> add_model.log
  echo "### end to generate mode for serving test ###"
  result=`find . -name  tensor_add.mindir | wc -l`
  if [ ${result} -ne 1 ]
  then
    cat add_model.log
    echo "### generate model for serving test failed ###" && exit 1
    clean_master_pid
    clean_worker_pid
    cd -
  fi
  cd -
}

start_service()
{
  echo "### start serving service ###"
  unset http_proxy https_proxy
  python3 master.py > start_master.log 2>&1 &
  if [ $? -ne 0 ]
  then
    echo "master server failed to start." && exit 1
  fi

  result=`grep -E 'Serving gRPC server start success, listening on 127.0.0.1:5500' start_master.log | wc -l`
  count=0
  while [[ ${result} -ne 1 && ${count} -lt 150 ]]
  do
    sleep 1
    count=$(($count+1))
    result=`grep -E 'Serving gRPC server start success, listening on 127.0.0.1:5500' start_master.log | wc -l`
  done

  if [ ${count} -eq 150 ]
  then
    clean_master_pid
    cat service.log
    echo "start serving service failed!" && exit 1
  fi

  echo "### start serving service end ###"

  python3 worker.py > start_worker.log 2>&1 &
  if [ $? -ne 0 ]
  then
    echo "worker server failed to start." && exit 1
  fi

  result=`grep -E 'gRPC server start success, listening on 127.0.0.1:6600' start_worker.log | wc -l`
  count=0
  while [[ ${result} -ne 1 && ${count} -lt 150 ]]
  do
    sleep 1
    count=$(($count+1))
    result=`grep -E 'gRPC server start success, listening on 127.0.0.1:6600' start_worker.log | wc -l`
  done

  if [ ${count} -eq 150 ]
  then
    clean_master_pid
    clean_worker_pid
    cat start_worker.log
    echo "start worker service failed!" && exit 1
  fi

  echo "### start worker service end ###"
}

pytest_serving()
{
  unset http_proxy https_proxy
  echo "###  client start ###"
  python3  client_mul_process.py > client_mul_process.log 2>&1
  if [ $? -ne 0 ]
  then
    clean_master_pid
    clean_worker_pid
    cat client_mul_process.log
    echo "client failed to start." && exit 1
  fi
  echo "### client end ###"
}

test_add_model()
{
  start_service
  pytest_serving
  cat client_mul_process.log
  clean_master_pid
  clean_worker_pid
}

echo "-----serving start-----"
rm -rf serving *.log *.mindir *.dat ${CURRPATH}/add ${CURRPATH}/kernel_meta
rm -rf add  client.py client_mul_process.py export_model  master.py worker.py master_with_worker.py
cp -r ../../../example/add/* .
prepare_model
test_add_model
