#!/bin/bash

export GLOG_v=1
cd "$(dirname $0)" || exit
CURRPATH=$(pwd)
CURRUSER=$(whoami)
PROJECT_PATH=${CURRPATH}/../../../
echo "CURRPATH:"  ${CURRPATH}
echo "CURRUSER:"  ${CURRUSER}
echo "PROJECT_PATH:"  ${PROJECT_PATH}

echo "LD_LIBRARY_PATH: " ${LD_LIBRARY_PATH}
echo "PYTHONPATH: " ${PYTHONPATH}

clean_pid()
{
  ps aux | grep 'serving_server.py' | grep ${CURRUSER} | grep -v grep | awk '{print $2}' | xargs kill -9
  if [ $? -ne 0 ]
  then
    echo "clean pip failed"
  fi
  sleep 6
}

prepare_model()
{
  echo "### begin to generate mode for serving test ###"
  cd export_model
  python3 export_resnet.py &> export_resnet.log
  echo "### end to generate mode for serving test ###"
  result=`find . -name resnet50_1b_imagenet.mindir | wc -l`
  if [ ${result} -ne 1 ]
  then
    cat export_resnet.log
    echo "### generate model for serving test failed ###" && exit 1
    clean_pid
    cd -
  fi
  cd -
}

start_service()
{
  echo "### start serving service ###"
  unset http_proxy https_proxy
  python3 serving_server.py > serving_server.log 2>&1 &
  if [ $? -ne 0 ]
  then
    echo "server failed to start."
  fi

  result=`grep -E 'Serving gRPC server start success, listening on 127.0.0.1:5500' serving_server.log | wc -l`
  count=0
  while [[ ${result} -eq 0 && ${count} -lt 150 ]]
  do
    sleep 1
    count=$(($count+1))
    result=`grep -E 'Serving gRPC server start success, listening on 127.0.0.1:5500' serving_server.log | wc -l`
  done

  if [ ${count} -eq 150 ]
  then
    clean_pid
    cat serving_server.log
    echo "start serving service failed!" && exit 1
  fi
  echo "### start serving service end ###"
}

pytest_serving()
{
  unset http_proxy https_proxy
  echo "###  client start ###"
  python3  serving_client.py > serving_client.log 2>&1
  if [ $? -ne 0 ]
  then
    clean_pid
    cat serving_client.log
    echo "client failed to start." && exit 1
  fi
  echo "### client end ###"
}

test_renet_model()
{
  start_service
  pytest_serving
  cat serving_client.log
  clean_pid
}

echo "-----serving start-----"
rm -rf serving *.log *.mindir *.dat kernel_meta
rm -rf unix_socket_files serving_logs
rm -rf serving_client.py  export_model  serving_server.py  resnet50  test_image
cp -r ../../../example/resnet/* .
prepare_model
test_renet_model
echo "### end to serving test ###"
