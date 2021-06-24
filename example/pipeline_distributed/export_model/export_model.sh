#!/bin/bash

EXEC_PATH=$(pwd)

export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
export RANK_SIZE=8

rm -rf device*
for ((i = 1; i < ${RANK_SIZE}; i++)); do
  mkdir device$i
  cp *.py ./device$i
  cd ./device$i
  export DEVICE_ID=$i
  export RANK_ID=$i
  echo "start inference for device $i"
  pytest -sv ./distributed_inference.py::test_inference >inference.log$i 2>&1 &
  cd ../
done

mkdir device0
cp *.py ./device0
cd ./device0
export DEVICE_ID=0
export RANK_ID=0
echo "start inference for device 0"
pytest -sv ./distributed_inference.py::test_inference >inference.log0 2>&1
if [ $? -eq 0 ]; then
  echo "inference success"
else
  echo "inference failed"
  cat inference.log0
  exit 2
fi
cd ../

output_dir=../model
rm -rf ${output_dir}/device*
for ((i = 0; i < ${RANK_SIZE}; i++)); do
  mkdir -p ${output_dir}/device${i}
  cp device${i}/*.mindir ${output_dir}/device${i}/
  cp device${i}/*.pb ${output_dir}/device${i}/
done

echo "copy models success"
