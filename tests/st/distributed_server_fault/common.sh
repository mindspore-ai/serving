#!/bin/bash

export GLOG_v=1

cd "$(dirname $0)" || exit;
CURRPATH=$(pwd)
CURRUSER=$(whoami)
PROJECT_PATH=${CURRPATH}/../../../
echo "CURRPATH:"  ${CURRPATH}
echo "CURRUSER:"  ${CURRUSER}
echo "PROJECT_PATH:"  ${PROJECT_PATH}

echo "LD_LIBRARY_PATH: " ${LD_LIBRARY_PATH}
echo "PYTHONPATH: " ${PYTHONPATH}

get_serving_server_count()
{
  num=`ps -ef | grep serving_server.py | grep -v grep | wc -l`
  return ${num}
}

get_serving_agent_count()
{
  num=`ps -ef | grep serving_agent.py | grep -v grep | wc -l`
  return ${num}
}

clean_pid()
{
  get_serving_server_count
  if [ $? -ne 0 ]
  then
    ps aux | grep 'serving_server.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -15
  fi

  count=0
  get_serving_server_count
  while [[ $? -ne 0 && ${count} -lt 5 ]]
  do
    sleep 1
    get_serving_server_count
  done

  get_serving_server_count
  if [ $? -ne 0 ]
  then
    ps aux | grep 'serving_server.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
  fi
  get_serving_agent_count
  if [ $? -ne 0 ]
  then
    ps aux | grep 'serving_agent.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
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
      cat model.log
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
  while [[ ${result} -eq 0 && ${count} -lt 100 ]]
  do
    sleep 1
    get_serving_server_count
    if [ $? -eq 0 ]
    then
      clean_pid

      echo "serving server log begin-------------------"
      cat serving_server.log
      echo "serving server log end-------------------"

      echo "serving worker log begin-------------------"
      cat serving_logs/*.log
      echo "serving worker log end-------------------"

      echo "start serving server failed!" && exit 1
    fi
    count=$(($count+1))
    result=`grep -E 'Master server start success, listening on' serving_server.log | wc -l`
  done

  if [ ${count} -eq 100 ]
  then
    clean_pid

    echo "serving server log begin-------------------"
    cat serving_server.log
    echo "serving server log end-------------------"

    echo "serving worker log begin-------------------"
    cat serving_logs/*.log
    echo "serving worker log end-------------------"

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
  while [[ ${result} -ne 1 && ${count} -lt 100 ]]
  do
    sleep 1
    get_serving_agent_count
    if [ $? -eq 0 ]
    then
      clean_pid
      cat serving_agent.log
      echo "start serving agent failed!" && exit 1
    fi
    count=$(($count+1))
    result=`grep -E 'Child 0: Receive success' serving_agent.log | wc -l`
  done

  if [ ${count} -eq 100 ]
  then
    clean_pid
    cat serving_agent.log
    echo "start serving agent failed!" && exit 1
  fi
  echo "### start serving agent end ###"
}

wait_server_exit()
{
  get_serving_server_count
  count=0
  while [[ $? -ne 0 && ${count} -lt 15 ]]
  do
    sleep 1
    count=$(($count+1))
    get_serving_server_count
  done

  if [ ${count} -eq 15 ]
  then
    echo "serving server exit failed"
    ps -ef | grep serving_server.py | grep -v grep
    echo "------------------------------ serving server failed log begin: "
    cat serving_server.log
    echo "------------------------------ serving server failed log end"
    clean_pid && exit 1
  fi
}

wait_agent_exit()
{
  get_serving_agent_count
  count=0
  while [[ $? -ne 0 && ${count} -lt 15 ]]
  do
    sleep 1
    count=$(($count+1))
    get_serving_agent_count
  done

  if [ ${count} -eq 15 ]
  then
    echo "serving agent exit failed"
    ps -ef | grep serving_agent.py | grep -v grep
    echo "------------------------------ serving agent failed log begin: "
    cat serving_agent.log
    echo "------------------------------ serving agent failed log end"
    clean_pid && exit 1
  fi
}

init()
{
  rm -rf serving *.log *.mindir *.dat matmul kernel_meta
  rm -rf unix_socket_files serving_logs
  rm -rf *.json export_model  serving_server.py serving_agent.py serving_client.py
  cp -r ../../../example/matmul_distributed/* .
  prepare_model
}

