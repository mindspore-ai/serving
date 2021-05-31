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
  num=`grep -E 'Recv Pong Time Out from' serving_logs/log_matmul*.log | wc -l`
  if [ $num -ne 0 ]
  then
    echo "serving agent has exited"
    echo $num
    clean_pid && exit 1
  fi
  ps aux | grep 'serving_agent.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
  if [ $? -ne 0 ]
  then
    echo "kill serving agent failed"
  fi
  sleep 25
  get_serving_agent_count
  if [ $? -ne 0 ]
  then
    echo "agent exit failed"
    echo $?
    clean_pid && exit 1
  fi
  get_serving_server_count
  if [ $? -ne 1 ]
  then
    echo "serving server start failed"
    echo $?
    clean_pid && exit 1
  fi
  num=`grep -E 'Recv Pong Time Out from' serving_logs/log_matmul*.log | wc -l`
  if [ $num -ne 8 ]
  then
    echo "catch agent exit failed"
    echo $num
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
