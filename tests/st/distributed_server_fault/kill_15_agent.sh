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

  wait_agent_exit
  wait_server_exit
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
