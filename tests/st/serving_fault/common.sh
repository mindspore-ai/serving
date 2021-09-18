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

clean_pid()
{
  get_master_count
  if [ $? -ne 0 ]
  then
    ps aux | grep 'serving_server.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -15
  fi

  count=0
  get_master_count
  while [[ $? -ne 0 && ${count} -lt 5 ]]
  do
    sleep 1
    get_master_count
  done

  get_master_count
  if [ $? -ne 0 ]
  then
    ps aux | grep 'serving_server.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
  fi
  get_worker_count
  if [ $? -ne 0 ]
  then
    ps aux | grep 'start_worker.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
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

start_serving_server()
{
  echo "### start serving server ###"
  unset http_proxy https_proxy
  python3 serving_server.py > serving_server.log 2>&1 &
  if [ $? -ne 0 ]
  then
    echo "server server failed to start."
  fi

  result=`grep -E 'Serving gRPC server start success, listening on 127.0.0.1:5500' serving_server.log | wc -l`
  count=0
  while [[ ${result} -eq 0 && ${count} -lt 50 ]]
  do
    sleep 1
    get_master_count
    if [ $? -eq 0 ]
    then
      echo "---------------------------------- server server log begin"
      cat serving_server.log
      echo "---------------------------------- server server log end"

      echo "---------------------------------- server worker log begin"
      cat serving_logs/*.log
      echo "---------------------------------- server worker log end"
      echo "start serving server failed!" && exit 1
    fi
    count=$(($count+1))
    result=`grep -E 'Serving gRPC server start success, listening on 127.0.0.1:5500' serving_server.log | wc -l`
  done

  if [ ${count} -eq 50 ]
  then
    clean_pid
    echo "---------------------------------- server server log begin"
    cat serving_server.log
    echo "---------------------------------- server server log end"

    echo "---------------------------------- server worker log begin"
    cat serving_logs/*.log
    echo "---------------------------------- server worker log end"
    echo "start serving server failed!" && exit 1
  fi
  echo "### start serving server end ###"
}

get_master_count()
{
  num=`ps -ef | grep serving_server.py | grep -v grep | wc -l`
  return ${num}
}

get_worker_count()
{
  num=`ps -ef | grep start_worker.py | grep -v grep | wc -l`
  return ${num}
}

wait_master_exit()
{
    get_master_count
    count=0
    while [[ $? -ne 0 && ${count} -lt 15 ]]
    do
      sleep 1
      count=$(($count+1))
      get_master_count
    done

    if [ ${count} -eq 15 ]
    then
      echo "serving master exit failed"
      ps -ef | grep serving_server.py | grep -v grep
      echo "---------------------------------- server server log begin"
      cat serving_server.log
      echo "---------------------------------- server server log end"

      echo "---------------------------------- server worker log begin"
      cat serving_logs/*.log
      echo "---------------------------------- server worker log end"
      clean_pid && exit 1
    fi
}

wait_worker_exit()
{
    get_worker_count
    count=0
    while [[ $? -ne 0 && ${count} -lt 15 ]]
    do
      sleep 1
      count=$(($count+1))
      get_worker_count
    done

    if [ ${count} -eq 15 ]
    then
      echo "serving worker exit failed"
      ps -ef | grep start_worker.py | grep -v grep
      echo "---------------------------------- server server log begin"
      cat serving_server.log
      echo "---------------------------------- server server log end"

      echo "---------------------------------- server worker log begin"
      cat serving_logs/*.log
      echo "---------------------------------- server worker log end"
      clean_pid && exit 1
    fi
}

init()
{
  rm -rf serving *.log *.mindir *.dat kernel_meta
  rm -rf unix_socket_files serving_logs
  rm -rf add export_model  serving_server.py serving_client.py serving_client_with_check.py
  cp -r ../../../example/tensor_add/* .
  prepare_model
  clean_pid
}

