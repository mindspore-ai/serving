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
  num=`ps -ef | grep master_with_worker.py | grep -v grep | wc -l`
  if [ $num -ne 0 ]
  then
    ps aux | grep 'master_with_worker.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
    is_sleep=1
  fi
  num=`ps -ef | grep agent.py | grep -v grep | wc -l`
  if [ $num -ne 0 ]
  then
    ps aux | grep 'agent.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
    is_sleep=1
  fi 
  if [ $is_sleep -ne 0 ]
  then
    sleep 5
  fi 
}

prepare_model()
{
  model_path=${CURRPATH}/../model
  if [ -d $model_path ]
  then
    echo "copy model path"
    cp -r ../model .
  else
    echo "### begin to generate mode for serving test ###"
    cd export_model
    sh export_model.sh &> model.log
    echo "### end to generate mode for serving test ###"
    result=`find ../ -name  model | wc -l`
    if [ ${result} -ne 1 ]
    then
      cat device0/inference.log0
      cat model.log
      echo "### generate model for serving test failed ###" && exit 1
      clean_pid
      cd -
    fi
    cd -
    cp -r model ../
  fi
}
start_master_with_worker()
{
  echo "### start serving master_with_worker ###"
  unset http_proxy https_proxy
  python3 master_with_worker.py > master_with_worker.log 2>&1 &
  if [ $? -ne 0 ]
  then
    echo "server master_with_worker failed to start."
  fi

  result=`grep -E 'Begin waiting ready of all agents' master_with_worker.log | wc -l`
  count=0
  while [[ ${result} -ne 1 && ${count} -lt 100 ]]
  do
    sleep 1
    count=$(($count+1))
    result=`grep -E 'Begin waiting ready of all agents' master_with_worker.log | wc -l`
  done

  if [ ${count} -eq 100 ]
  then
    clean_pid
    cat master_with_worker.log 
    echo "start serving master_with_worker failed!" && exit 1
  fi
  echo "### start serving master_with_worker end ###"
}
start_agent()
{
  echo "### start serving agent ###"
  unset http_proxy https_proxy
  python3 agent.py > agent.log 2>&1 &
  if [ $? -ne 0 ]
  then
    echo "server agent failed to start."
  fi

  result=`grep -E 'Child 0: Receive success' agent.log | wc -l`
  count=0
  while [[ ${result} -ne 1 && ${count} -lt 100 ]]
  do
    sleep 1
    count=$(($count+1))
    result=`grep -E 'Child 0: Receive success' agent.log | wc -l`
  done

  if [ ${count} -eq 100 ]
  then
    clean_pid
    cat agent.log 
    echo "start serving agent failed!" && exit 1
  fi
  echo "### start serving agent end ###"
}
kill_agent()
{
  num=`ps -ef | grep master_with_worker.py | grep -v grep | wc -l`
  if [ $num -ne 1 ]
  then
    echo "master_with_worker start failed"
    echo $num
    clean_pid && exit 1
  fi
  num=`ps -ef | grep agent.py | grep -v grep | wc -l`
  if [ $num -ne 9 ]
  then
    echo "agent start failed"
    echo $num
    clean_pid && exit 1
  fi
  num=`grep -E 'Recv Pong Time Out from' master_with_worker.log | wc -l`
  if [ $num -ne 0 ]
  then
    echo "agent has exited"
    echo $num
    clean_pid && exit 1
  fi
  ps aux | grep 'agent.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
  if [ $? -ne 0 ]
  then
    echo "kill agent failed"
  fi
  sleep 25
  num=`ps -ef | grep agent.py | grep -v grep | wc -l`
  if [ $num -ne 0 ]
  then
    echo "agent exit failed"
    echo $num
    clean_pid && exit 1
  fi
  num=`ps -ef | grep master_with_worker.py | grep -v grep | wc -l`
  if [ $num -ne 1 ]
  then
    echo "master_with_worker start failed"
    echo $num
    clean_pid && exit 1
  fi
  num=`grep -E 'Recv Pong Time Out from' master_with_worker.log | wc -l`
  if [ $num -ne 8 ]
  then
    echo "catch agent exit failed"
    echo $num
    clean_pid && exit 1
  fi
}

test_agent_fault_model()
{
  start_master_with_worker
  start_agent
  kill_agent
  clean_pid
}

echo "-----serving start-----"
rm -rf serving *.log *.mindir *.dat ${CURRPATH}/matmul ${CURRPATH}/kernel_meta
rm -rf client.py *.json export_model  master_with_worker.py master.py worker.py agent.py
cp -r ../../../example/matmul_distributed/* .
prepare_model
test_agent_fault_model
echo "### end to serving test ###"
