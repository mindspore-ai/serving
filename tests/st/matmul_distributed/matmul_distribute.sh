#!/bin/bash

export GLOG_v=1

cd "$(dirname $0)" || exit
CURRPATH=$(pwd)
CURRUSER=$(whoami)
PROJECT_PATH=${CURRPATH}/../../../
echo "CURRPATH:"  ${CURRPATH}
echo "CURRUSER:"  ${CURRUSER}
echo "PROJECT_PATH:"  ${PROJECT_PATH}

echo "LD_LIBRARY_PATH: " ${LD_LIBRARY_PATH}
echo "PYTHONPATH: " ${PYTHONPATH}

clean_server_pid()
{
  num=`ps -ef | grep serving_server.py | grep -v grep | wc -l`
  if [ ${num} -ne 0 ]
  then
    ps aux | grep 'serving_server.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -15
    if [ $? -ne 0 ]
    then
      echo "clean master pid failed"
    fi
  fi

  num=`ps -ef | grep start_distributed_worker.py | grep -v grep | wc -l`
  count=0
  while [[ ${num} -ne 0 && ${count} -lt 10 ]]
  do
    sleep 1
    count=$(($count+1))
    num=`ps -ef | grep start_distributed_worker.py | grep -v grep | wc -l`
  done

  if [ ${count} -eq 10 ]
  then
    echo "worker exit failed"
    echo $num
    ps -ef | grep start_distributed_worker.py | grep -v grep

    echo "------------------------------ worker failed master log begin: "
    cat serving_server.log
    echo "------------------------------ worker failed master log end"

    echo "------------------------------ worker failed log begin: "
    cat serving_logs/*.log
    echo "------------------------------ worker failed log end"
    clean_pid && exit 1
  fi

  num=`ps -ef | grep serving_agent.py | grep -v grep | wc -l`
  count=0
  while [[ ${num} -ne 0 && ${count} -lt 10 ]]
  do
    sleep 1
    count=$(($count+1))
    num=`ps -ef | grep serving_agent.py | grep -v grep | wc -l`
  done

  if [ ${count} -eq 10 ]
  then
    echo "agent exit failed"
    echo $num
    ps -ef | grep serving_agent.py | grep -v grep
    echo "------------------------------ agent failed log begin: "
    cat serving_agent.log
    echo "------------------------------ agent failed log end"
    clean_pid && exit 1
  fi
}

clean_pid()
{
  ps aux | grep 'serving_server.py' | grep ${CURRUSER} | grep -v grep
  if [ $? -eq 0 ]
  then
    ps aux | grep 'serving_server.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
    echo "### master pid exist, clean master pip failed ###"
  fi
  ps aux | grep 'start_distributed_worker.py' | grep ${CURRUSER} | grep -v grep
  if [ $? -eq 0 ]
  then
    ps aux | grep 'start_distributed_worker.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
    echo "### master pid is killed but worker pid exist ###"
  fi
  ps aux | grep 'serving_agent.py' | grep ${CURRUSER} | grep -v grep
  if [ $? -eq 0 ]
  then
    ps aux | grep 'serving_agent.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
    echo "### worker pid is killed but agent pid exist ###"
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
    cd export_model || exit
    sh export_model.sh &> model.log
    echo "### end to generate mode for serving test ###"
    result=`find ../ -name  model | wc -l`
    if [ ${result} -ne 1 ]
    then
      echo "### begin model generation log ###"
      cat model.log
      echo "### end model generation log ###"
      clean_pid
      echo "### generate model for serving test failed ###" && exit 1
    fi
    cd - || exit
    cp -r model ../
  fi
}

start_serving_server()
{
  echo "### start serving server ###"
  unset http_proxy https_proxy
  python3 serving_server.py > serving_server.log 2>&1 &
  if [ $? -ne 0 ]
  then
    echo "serving server failed to start."
  fi

  result=`grep -E 'Master server start success, listening on' serving_server.log | wc -l`
  count=0
  while [[ ${result} -eq 0 && ${count} -lt 150 ]]
  do
    sleep 1
    num=`ps -ef | grep serving_server.py | grep -v grep | wc -l`
    if [ ${num} -eq 0 ]
    then
      echo "serving server log begin-------------------"
      cat serving_server.log
      echo "serving server log end-------------------"

      echo "serving worker log begin-------------------"
      cat serving_logs/*.log
      echo "serving worker log end-------------------"
      clean_pid
      echo "start serving server failed!" && exit 1
    fi
    count=$(($count+1))
    result=`grep -E 'Master server start success, listening on' serving_server.log | wc -l`
  done

  if [ ${count} -eq 150 ]
  then
    echo "serving server log begin-------------------"
    cat serving_server.log
    echo "serving server log end-------------------"

    echo "serving worker log begin-------------------"
    cat serving_logs/*.log
    echo "serving worker log end-------------------"
    clean_pid
    echo "start serving server failed!" && exit 1
  fi
  echo "### start serving server end ###"
}

start_serving_agent()
{
  echo "### start serving agent ###"
  unset http_proxy https_proxy
  python3 serving_agent.py > serving_agent.log 2>&1 &
  if [ $? -ne 0 ]
  then
    echo "server agent failed to start."
  fi

  result=`grep -E 'Child 0: Receive success' serving_agent.log | wc -l`
  count=0
  while [[ ${result} -ne 1 && ${count} -lt 150 ]]
  do
    sleep 1
    num=`ps -ef | grep serving_agent.py | grep -v grep | wc -l`
    if [ ${num} -eq 0 ]
    then
      clean_pid
      cat serving_agent.log
      echo "start serving agent failed!" && exit 1
    fi
    count=$(($count+1))
    result=`grep -E 'Child 0: Receive success' serving_agent.log | wc -l`
  done

  if [ ${count} -eq 150 ]
  then
    clean_pid
    cat serving_agent.log
    echo "start serving agent failed!" && exit 1
  fi
  echo "### start serving agent end ###"
}

pytest_serving()
{
  unset http_proxy https_proxy
  echo "###  client start ###"
  python3  serving_client.py > serving_client.log 2>&1
  if [ $? -ne 0 ]
  then
    cat serving_client.log
    clean_server_pid
    clean_pid
    echo "client failed to start." && exit 1
  fi
  echo "### client end ###"
}

test_matmul_distribute()
{
  start_serving_server
  start_serving_agent
  pytest_serving
  cat serving_client.log
  clean_server_pid
  clean_pid
}

echo "-----serving start-----"
rm -rf serving *.log *.dat matmul model kernel_meta somas_meta
rm -rf unix_socket_files serving_logs
rm -rf serving_client.py  export_model temp_rank_table serving_server.py serving_agent.py rank_table_8pcs.json
cp -r ../../../example/matmul_distributed/* .
prepare_model
test_matmul_distribute
