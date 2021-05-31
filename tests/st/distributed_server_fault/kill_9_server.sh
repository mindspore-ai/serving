#!/bin/bash

CURRPATH=$(cd "$(dirname $0)" || exit; pwd)
source ${CURRPATH}/common.sh

kill_serving_server()
{
  get_serving_server_count
  if [ $? -ne 1 ]
  then
    echo "serving server start failed"
    echo $?
    clean_pid && exit 1
  fi
  num=`ps -ef | grep start_distributed_worker.py | grep -v grep | wc -l`
  if [ ${num} -ne 1 ]
  then
    echo "serving worker start failed"
    echo ${num}
    clean_pid && exit 1
  fi
  get_serving_agent_count
  if [ $? -ne 9 ]
  then
    echo "serving agent start failed"
    echo $?
    clean_pid && exit 1
  fi
  num=`grep -E 'Recv Ping Time Out from' serving_server.log | wc -l`
  if [ $num -ne 0 ]
  then
    echo "serving agent has exited"
    echo $num
    clean_pid && exit 1
  fi
  ps aux | grep 'start_distributed_worker.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
  if [ $? -ne 0 ]
  then
    echo "kill serving worker failed"
  fi
  sleep 25
  num=`grep -E 'Recv Ping Time Out from' serving_agent.log | wc -l`
  if [ $num -ne 8 ]
  then
    echo "catch serving server exit failed"
    echo $num
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
