#!/bin/bash

CURRPATH=$(cd "$(dirname $0)" || exit; pwd)
source ${CURRPATH}/common.sh

kill_serving_agent()
{
  get_serving_server_count
  if [ $? -ne 1 ]
  then
    echo "serving server start failed"
    echo $?
    clean_pid && exit 1
  fi
  get_serving_agent_count
  if [ $? -ne 9 ]
  then
    echo "serving agent start failed"
    echo $?
    clean_pid && exit 1
  fi
  ps aux | grep 'serving_agent.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -15
  if [ $? -ne 0 ]
  then
    echo "kill agent failed"
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
    echo "agent exit failed"
    echo $?
    ps -ef | grep serving_agent.py | grep -v grep
    echo "------------------------------ agent failed log begin: "
    cat serving_agent.log
    echo "------------------------------ agent failed log end"
    clean_pid && exit 1
  fi
  sleep 5

  get_serving_server_count
  if [ $? -ne 0 ]
  then
    echo "serving server exit failed"
    echo $?
    clean_pid && exit 1
  fi
}

test_kill_serving_agent()
{
  start_serving_server
  start_serving_agent
  kill_serving_agent
  clean_pid
}

echo "-----serving start-----"
init
test_kill_serving_agent
echo "### end to serving test ###"
