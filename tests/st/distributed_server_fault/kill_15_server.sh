#!/bin/bash

CURRPATH=$(cd "$(dirname $0)" || exit; pwd)
source ${CURRPATH}/common.sh

kill_serving_server()
{
  get_serving_server_count
  if [ $? -ne 1 ]
  then
    echo "master_with_worker start failed"
    echo $?
    clean_pid && exit 1
  fi
  get_serving_agent_count
  if [ $? -ne 9 ]
  then
    echo "agent start failed"
    echo $?
    clean_pid && exit 1
  fi
  ps aux | grep 'serving_server.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -15
  if [ $? -ne 0 ]
  then
    echo "kill master_with_worker failed"
  fi

  get_serving_agent_count
  count=0
  while [[ $? -ne 0 && ${count} -lt 10 ]]
  do
    sleep 1
    count=$(($count+1))
    get_serving_agent_count
  done

  if [ ${count} -eq 10 ]
  then
    echo "serving agent exit failed"
    ps -ef | grep serving_agent.py | grep -v grep
    echo "------------------------------ serving agent failed log begin: "
    cat serving_agent.log
    echo "------------------------------ serving agent failed log end"
    clean_pid && exit 1
  fi
  sleep 1

  get_serving_server_count
  if [ $? -ne 0 ]
  then
    echo "serving server exit failed"
    echo $?
    clean_pid && exit 1
  fi
}

test_kill_serving_server()
{
  start_serving_server
  start_serving_agent
  kill_serving_server
  clean_pid
}

echo "-----serving start-----"
init
test_kill_serving_server
echo "### end to serving test ###"
