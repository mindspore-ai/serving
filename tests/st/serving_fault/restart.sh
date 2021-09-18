#!/bin/bash

CURRPATH=$(cd "$(dirname $0)" || exit; pwd)
source ${CURRPATH}/common.sh

unset http_proxy https_proxy

run_client()
{
  echo "###  client start ###"
  python3  serving_client_with_check.py > client.log 2>&1
  if [ $? -ne 0 ]
  then
    clean_pid
    cat client.log
    echo "client failed to start." && exit 1
  fi
  cat client.log
  echo "### client end ###"
}

listening_worker_restart()
{
  start_count=$1
  echo "### serving server worker restart begin ###"
  result=`grep -E 'Register success: worker address' serving_server.log | wc -l`
  count=0
  while [[ ${result} -le $start_count && ${count} -lt 30 ]]
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
      echo "serving server worker restart failed! start count $start_count" && exit 1
    fi
    count=$(($count+1))
    result=`grep -E 'Register success: worker address' serving_server.log | wc -l`
  done

  if [ ${count} -eq 30 ]
  then
    clean_pid
    echo "---------------------------------- server server log begin"
    cat serving_server.log
    echo "---------------------------------- server server log end"

    echo "---------------------------------- server worker log begin"
    cat serving_logs/*.log
    echo "---------------------------------- server worker log end"
    echo "serving server worker restart failed! start count $start_count" && exit 1
  fi
  echo "### serving server worker restart end ###"
}

test_restart()
{
  start_serving_server
  # shellcheck disable=SC2207
  worker_pids=($(ps aux | grep 'start_worker.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}'))
  if [ ${#worker_pids[*]} -ne 2 ]; then
    echo "worker process number is not 2, real count " ${#worker_pids[*]}
    ps -ef | grep start_worker.py
    clean_pid && exit 1
  fi

  echo "before restart"
  ps -ef | grep 'start_worker.py'

  # test kill -9 and restart
  run_client

  echo "kill first worker " ${worker_pids[0]}
  kill -s 9 ${worker_pids[0]}
  echo "after first kill"
  ps -ef | grep 'start_worker.py'

  run_client
  listening_worker_restart 2  # current has 2 Register success log
  run_client

  echo "kill second worker " ${worker_pids[1]}
  kill -s 9 ${worker_pids[1]}
  echo "after second kill"
  ps -ef | grep 'start_worker.py'

  listening_worker_restart 3  # current has 3 Register success log
  # test kill -15
  run_client
  # shellcheck disable=SC2207
  worker_pids=($(ps aux | grep 'start_worker.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}'))
  if [ ${#worker_pids[*]} -ne 2 ]; then
    echo "restarted worker process number is not 2, real count " ${#worker_pids[*]}
    ps -ef | grep start_worker.py
    clean_pid && exit 1
  fi

  echo "end restart"
  ps -ef | grep 'start_worker.py'

  kill -s 15 ${worker_pids[0]}
  kill -s 15 ${worker_pids[1]}
  wait_master_exit
  clean_pid
}

echo "-----serving start-----"
init
test_restart
echo "-----serving end-----"
