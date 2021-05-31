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
  ps aux | grep 'serving_server.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -15
  if [ $? -ne 0 ]
  then
    echo "kill master failed"
  fi
  sleep 5
  get_master_count
  if [ $? -ne 0 ]
  then
    echo "master exit failed"
    echo $?
    clean_pid && exit 1
  fi
  get_worker_count
  if [ $? -ne 0 ]
  then
    echo "worker exit failed"
    echo $?
    clean_pid && exit 1
  fi
}

test_master_fault_model()
{
  start_serving_server
  kill_master
  clean_pid
}

echo "-----serving start-----"
rm -rf serving *.log *.mindir *.dat ${CURRPATH}/add ${CURRPATH}/kernel_meta
rm -rf ${CURRPATH}/unix_socket_files ${CURRPATH}/serving_logs
rm -rf add export_model serving_server.py serving_client.py serving_client_with_check.py
cp -r ../../../example/add/* .
prepare_model
test_master_fault_model
echo "### end to serving test ###"
