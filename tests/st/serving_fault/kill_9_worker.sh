#!/bin/bash

CURRPATH=$(cd "$(dirname $0)" || exit; pwd)
source ${CURRPATH}/common.sh

kill_worker()
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

  ps aux | grep 'start_worker.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
  if [ $? -ne 0 ]
  then
    echo "kill worker failed"
  fi
  sleep 5
  get_master_count
  if [ $? -ne 0 ]
  then
    echo "master exit failed"
    echo $?
    clean_pid && exit 1
  fi
}

test_worker_fault_model()
{
  start_serving_server
  kill_worker
  clean_pid
}

echo "-----serving start-----"
init
test_worker_fault_model
echo "### end to serving test ###"
