#!/bin/bash

# ========================每次启动需要更改的变量��?==========================
# 指定batch size
bs=16
# 指定端口
port=61177
description="bs16 pa224 z30001750" # 涓烘湰娆℃祴璇曟坊鍔犳弿杩?
serving_log_dir="/path to/serving/examples/output"
# ==========================================================================

# 指定结果存放路径
save_dir=$(date +"%Y_%m_%d_%H_%M_%S")
echo "start test, description: $description"
echo "result files are dumped to dir: $save_dir"
# 创建测试结果保存目录
mkdir $save_dir
test_client_log_path=$save_dir/test_client.log
touch $test_client_log_path
echo $description >> $test_client_log_path

# 指定log目录
agent_log_path="$serving_log_dir/agent_0.log"
server_log_path="$serving_log_dir/server_app.log"

# 指定npu total time保存路径
npu_time_path=$save_dir/npu_total_time.log

# 指定swap保存路径
swap_path=$save_dir/swap.log

test_throughput() {
    input_path=$1
    output_path=$2
    multiply=$3
    # agent日志文件起始行数
    if [ -f $agent_log_path ]; then
        start_lines=$(wc -l < $agent_log_path)
    else
        start_lines=0
    fi
    # server日志文件起始行数
    if [ -f $server_log_path ]; then
	server_start_line=$(wc -l < $server_log_path)
    else
	server_start_line=0
    fi
    
    python test_client.py --input_path $input_path --output_path $output_path --multiply $multiply --port $port >> $test_client_log_path
    
    # 计算生成的总行��?    total_lines=$(($(wc -l < $agent_log_path)-$start_lines))
    total_server_lines=$(($(wc -l < $server_log_path)-$server_start_line))

    tail -$total_lines $agent_log_path | grep 'npu_total_time' > $npu_time_path
    tail -$total_server_lines $server_log_path | grep 'swap' > $swap_path
    
    # 统计npu_total_time个数和总时��?    npu_time_lines=$(wc -l < $npu_time_path)
    start_time_str=$(sed -n "1p" $npu_time_path | cut -d ',' -f 1)
    end_time_str=$(sed -n "${npu_time_lines}p" $npu_time_path | cut -d ',' -f 1)
    start_time=$(date -d "$start_time_str" +%s)
    end_time=$(date -d "$end_time_str" +%s)
    cost_time=$((end_time-start_time))
    total_tokens=$(($bs*$npu_time_lines))
    speed=$((${total_tokens}/${cost_time}))

    # 统计swap数量
    swap_lines=$(wc -l < $swap_path)

    # echo "$save_dir tokens: $total_tokens cost_time: $cost_time speed: $speed tokens/s" >> $test_client_log_path
    echo "swap times: $swap_lines" >> $test_client_log_path
}

jsonl_list=(
    "/path to/testdata/dataset/first_round_short.jsonl"
    "/path to/testdata/dataset/first_round_100.jsonl"
    "/path to/testdata/dataset/first_round.jsonl"
    "/path to/testdata/dataset/second_round.jsonl"
    "/path to/testdata/dataset/open_weight_2000_answer.jsonl"
    "/path to/testdata/dataset/99395_ocr_for_npu_test0_2000.jsonl"
)

out_list=(
    "$save_dir/first_round_short_out.jsonl"
    "$save_dir/first_round_100_out.jsonl"
    "$save_dir/first_round_out.jsonl"
    "$save_dir/second_round_out.jsonl"
    "$save_dir/open_weight_2000_answer_out.jsonl"
    "$save_dir/99395_ocr_for_npu_test0_2000_out.jsonl"
)

# single test
single_test_index=4
# test_throughput ${jsonl_list[$single_test_index]} ${out_list[$single_test_index]} 1

# test multiply
req_num_list=(20 100 500)
for (( i=0; i<3; i++ ))
do
    break
    test_throughput ${jsonl_list[$single_test_index]} ${out_list[$single_test_index]} ${req_num_list[i]}
done

# multi round test
for (( i=0; i<${#jsonl_list[@]}; i++ )) # 20
do
    # break
    echo "load jsonl path: ${jsonl_list[i]}, write to: ${out_list[i]}"
    test_throughput ${jsonl_list[i]} ${out_list[i]} 1
    sleep 3
done
