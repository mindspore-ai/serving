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

clean_pid()
{
  is_sleep=0
  num=`ps -ef | grep master.py | grep -v grep | wc -l`
  if [ $num -ne 0 ]
  then
    ps aux | grep 'master.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
    is_sleep=1
  fi
  num=`ps -ef | grep worker.py | grep -v grep | wc -l`
  if [ $num -ne 0 ]
  then
    ps aux | grep 'worker.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
    is_sleep=1
  fi 
  if [ $is_sleep -ne 0 ]
  then
    sleep 5
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
    clean_pid
    cd -
  fi
  cd -
}

start_master()
{
  echo "### start serving master ###"
  unset http_proxy https_proxy
  python3 master.py > master.log 2>&1 &
  if [ $? -ne 0 ]
  then
    echo "server master failed to start."
  fi

  result=`grep -E 'Serving gRPC server start success, listening on 127.0.0.1:5500' master.log | wc -l`
  count=0
  while [[ ${result} -ne 1 && ${count} -lt 50 ]]
  do
    sleep 1
    count=$(($count+1))
    result=`grep -E 'Serving gRPC server start success, listening on 127.0.0.1:5500' master.log | wc -l`
  done

  if [ ${count} -eq 50 ]
  then
    clean_pid
    cat master.log 
    echo "start serving master failed!" && exit 1
  fi
  echo "### start serving master end ###"
}
start_worker()
{
  echo "### start serving worker ###"
  unset http_proxy https_proxy
  python3 worker.py > worker.log 2>&1 &
  if [ $? -ne 0 ]
  then
    echo "server worker failed to start."
  fi

  result=`grep -E 'Begin to send pong' worker.log | wc -l`
  count=0
  while [[ ${result} -ne 1 && ${count} -lt 100 ]]
  do
    sleep 1
    count=$(($count+1))
    result=`grep -E 'Begin to send pong' worker.log | wc -l`
  done

  if [ ${count} -eq 100 ]
  then
    clean_pid
    cat worker.log 
    echo "start serving worker failed!" && exit 1
  fi
  echo "### start serving worker end ###"
}

kill_master()
{
  num=`ps -ef | grep master.py | grep -v grep | wc -l`
  if [ $num -ne 1 ]
  then
    echo "master start failed"
    echo $num
    clean_pid && exit 1
  fi
  num=`ps -ef | grep worker.py | grep -v grep | wc -l`
  if [ $num -ne 1 ]
  then
    echo "worker start failed"
    echo $num
    clean_pid && exit 1
  fi
  num=`grep -E 'Recv Ping Time Out from' worker.log | wc -l`
  if [ $num -ne 0 ]
  then
    echo "master has exited"
    echo $num
    clean_pid && exit 1
  fi
  ps aux | grep 'master.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
  if [ $? -ne 0 ]
  then
    echo "kill master failed"
  fi
  sleep 25
  num=`ps -ef | grep master.py | grep -v grep | wc -l`
  if [ $num -ne 0 ]
  then
    echo "master exit failed"
    echo $num
    clean_pid && exit 1
  fi
  num=`grep -E 'Recv Ping Time Out from' worker.log | wc -l`
  if [ $num -ne 1 ]
  then
    echo "catch master exit failed"
    echo $num
    clean_pid && exit 1
  fi
}

test_master_fault_model()
{
  start_master
  start_worker
  kill_master
  clean_pid
}

echo "-----serving start-----"
rm -rf serving *.log *.mindir *.dat ${CURRPATH}/add ${CURRPATH}/kernel_meta
rm -rf add  client.py client_mul_process.py  export_model  master_with_worker.py master.py worker.py
cp -r ../../../example/add/* .
prepare_model
test_master_fault_model
echo "### end to serving test ###"
