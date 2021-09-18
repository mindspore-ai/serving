#!/bin/bash

CURRPATH=$(cd "$(dirname $0)" || exit; pwd)
source ${CURRPATH}/common.sh

kill_master()
{
  get_master_count
  if [ $? -ne 1 ]
  then
    echo "serving server start failed"
    echo $?
    clean_pid && exit 1
  fi
  get_worker_count
  if [ $? -eq 0 ]
  then
    echo "worker start failed"
    echo $?
    clean_pid && exit 1
  fi
  ps aux | grep 'serving_server.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
  if [ $? -ne 0 ]
  then
    echo "kill master failed"
  fi
  wait_worker_exit
}

test_master_fault_model()
{
  start_serving_server
  kill_master
  clean_pid
}

echo "-----serving start-----"
init
test_master_fault_model
echo "### end to serving test ###"
