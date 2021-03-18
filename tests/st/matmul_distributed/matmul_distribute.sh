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

    num=`ps -ef | grep agent.py | grep -v grep | wc -l`
    count=0
    while [[ ${num} -ne 0 && ${count} -lt 10 ]]
    do
      sleep 1
      count=$(($count+1))
      num=`ps -ef | grep agent.py | grep -v grep | wc -l`
    done

    if [ ${count} -eq 10 ]
    then
      echo "agent exit failed"
      echo $num
      ps -ef | grep agent.py | grep -v grep
      echo "------------------------------ agent failed log begin: "
      cat agent.log
      echo "------------------------------ agent failed log end"
      clean_pid && exit 1
    fi
    sleep 1

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
    ps aux | grep 'agent.py' | grep ${CURRUSER} | grep -v grep
    if [ $? -eq 0 ]
    then
      ps aux | grep 'agent.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
      echo "### worker pid is killed but agent pid exist ###" & exit 1
    fi
  fi
}

clean_worker_pid()
{
  ps aux | grep 'worker.py' | grep ${CURRUSER} | grep -v grep
  if [ $? -eq 0 ]
  then
    ps aux | grep 'worker.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -15
    if [ $? -ne 0 ]
    then
      echo "clean worker pip failed"
    fi
    sleep 6
    ps aux | grep 'worker.py' | grep ${CURRUSER} | grep -v grep
    if [ $? -eq 0 ]
    then
      ps aux | grep 'worker.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
      echo "### worker pid exist, clean worker pip failed ###" & exit 1
    fi
    ps aux | grep 'agent.py' | grep ${CURRUSER} | grep -v grep
    if [ $? -eq 0 ]
    then
      ps aux | grep 'agent.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
      echo "### worker pid is killed but agent pid exist ###" & exit 1
    fi
  fi
}

clean_agent_pid()
{
  ps aux | grep 'agent.py' | grep ${CURRUSER} | grep -v grep
  if [ $? -eq 0 ]
  then
    ps aux | grep 'agent.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
    if [ $? -eq 0 ]
    then
      echo "clean agent pid failed"
    fi
  fi
}

prepare_model()
{
  echo "### begin to generate mode for serving matmul distribute test ###"
  cd export_model
  bash export_model.sh &> export_model.log
  if [ $? -ne 0 ]
  then
    cat export_model.log
    echo "### generate model for serving matmul distribute test failed ###" && exit 1
    clean_master_pid
    clean_worker_pid
    clean_agent_pid
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
  while [[ ${result} -ne 1 && ${count} -lt 50 ]]
  do
    sleep 1
    count=$(($count+1))
    result=`grep -E 'Serving gRPC server start success, listening on 127.0.0.1:5500' start_master.log | wc -l`
  done

  if [ ${count} -eq 50 ]
  then
    clean_master_pid
    cat start_master.log
    echo "start serving service failed!" && exit 1
  fi

  echo "### start serving master service end ###"

  python3 worker.py > start_worker.log 2>&1 &
  if [ $? -ne 0 ]
  then
    echo "worker server failed to start." && exit 1
  fi

  result=`grep -E 'gRPC server start success, listening on 127.0.0.1:6200' start_worker.log | wc -l`
  count=0
  while [[ ${result} -ne 1 && ${count} -lt 50 ]]
  do
    sleep 1
    count=$(($count+1))
    result=`grep -E 'gRPC server start success, listening on 127.0.0.1:6200' start_worker.log | wc -l`
  done

  if [ ${count} -eq 50 ]
  then
    clean_master_pid
    clean_worker_pid
    cat start_worker.log
    echo "start worker service failed!" && exit 1
  fi

  echo "### start worker service end ###"

    python3 agent.py > start_agent.log 2>&1 &
  if [ $? -ne 0 ]
  then
    echo "agent server failed to start." && exit 1
  fi

  result=`grep -E 'Agent server start success, listening on 127.0.0.1:' start_agent.log | grep -E '7000|7001|7002|7003|7004|7005|7006|7007'| wc -l`
  count=0
  while [[ ${result} -ne 8 && ${count} -lt 150 ]]
  do
    sleep 1
    count=$(($count+1))
    result=`grep -E 'Agent server start success, listening on 127.0.0.1:' start_agent.log | grep -E '7000|7001|7002|7003|7004|7005|7006|7007'| wc -l`
  done

  if [ ${count} -eq 150 ]
  then
    clean_master_pid
    clean_worker_pid
    clean_agent_pid
    cat start_agent.log
    echo "start agent service failed!" && exit 1
  fi

  echo "### start agent service end ###"
}

pytest_serving()
{
  unset http_proxy https_proxy
  echo "###  client start ###"
  python3  client.py > client.log 2>&1
  if [ $? -ne 0 ]
  then
    clean_master_pid
    clean_worker_pid
    clean_agent_pid
    cat client.log
    echo "client failed to start." && exit 1
  fi
  echo "### client end ###"
}

test_add_model()
{
  start_service
  pytest_serving
  cat client.log
  clean_master_pid
  clean_worker_pid
  clean_agent_pid
}

echo "-----serving start-----"
rm -rf serving *.log *.dat ${CURRPATH}/matmul ${CURRPATH}/model ${CURRPATH}/kernel_meta ${CURRPATH}/somas_meta
rm -rf client.py  export_model temp_rank_table master.py worker.py agent.py master_with_worker.py  rank_table_8pcs.json
cp -r ../../../example/matmul_distributed/* .
prepare_model
test_add_model
