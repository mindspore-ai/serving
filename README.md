# MindSpore Serving

### serving is a fast and easy-to-use inference framework

---
### Features
- model parallel deployment
- token streaming via Server-Sent Events
- custom model inputs
- static/continuous batching of prompts
- do post sampling via npu
- PagedAttention

### Supports the most popular LLMs, including the following architectures:
- LLaMA-2
- InternLM
- baichuan2
- wizardcoder

### Get Started
#### 环境依赖
- python 3.9
- [mindspore](https://www.mindspore.cn/install)
- [mindspore-lite](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html?highlight=%E5%AE%89%E8%A3%85) 
- [mindformers](https://gitee.com/mindspore/mindformers#%E4%BA%8Cmindformers%E5%AE%89%E8%A3%85)
- easydict
- transformers==4.35.0

#### 一键安装whl包
```shell
pip install mindspore-serving-xxx.whl
```

注：后处理当前按照入图的方式进行，使用serving前请使用post_sampling_model.py重新导出后处理模型，保证数据类型与LLM模型的输出类型一致；
```shell
python tools/post_sampling_model.py --output_dir ./target_dir
# args
#   output_dir: 后处理模型生成的目录
```

#### 修改模型对应的配置文件

##### 带PagedAttention配置

###### yaml文件

在模型对应的配置文件`configs/模型名称/xxx.yaml`中，用户可自行修改模型，并通过`page_attention`开启PA的模型训练（True为启动模型PA功能，并在后面添加`pa_config`的设置项，具体参数根据模型来设置）

```
model_path:
    prefill_model: ["/path/to/baichuan2/output_serving/mindir_full_checkpoint/rank_0_graph.mindir"]
    decode_model: ["/path/to/baichuan2/output_serving/mindir_inc_checkpoint/rank_0_graph.mindir"]
    argmax_model: "/path/to/serving/target_dir/argmax.mindir"
    topk_model: "/path/to/target_dir/topk.mindir"
    prefill_ini: ["/path/to/baichuan_ini/910b_default_ctx.cfg"]
    decode_ini: ["/path/to/baichuan_ini/910b_default_inc.cfg"]
    post_model_ini: "/path/to/serving/target_dir/config.ini"
model_config:
    model_name: 'baichuan2'
    max_generate_length: 4096
    end_token: 2
    seq_length: [1024, 2048, 4096]	#支持多分档
    vocab_size: 125696
    prefill_batch_size: [1]     #带PA功能只支持单batch
    decode_batch_size: [16]  	#带PA功能只支持多batch，但不支持多分档
    zactivate_len: [4096]
    model_type: 'dyn'
    page_attention: True   # True为启动模型PA功能
    current_index: False
    model_dtype: "DataType.FLOAT32"
    pad_token_id: 0   
    
pa_config:  #带PA的配置此项，根据模型设置参数
    num_blocks: 512     
    block_size: 16
    decode_seq_length: 4096
    
serving_config:
    agent_ports: [61166]
    start_device_id: 0
    server_ip: 'localhost'
    server_port: 61155    
    
tokenizer:
    type: LlamaTokenizer
    vocab_file: '/path/to/llama_pa_models/output/tokenizer_llama2_13b.model'

basic_inputs:
    type: LlamaBasicInputs

extra_inputs:
    type: LlamaExtraInputs

warmup_inputs:
    type: LlamaWarmupInputs
```

###### prefill_ini

```
[ascend_context]
provider=ge

[ge_session_options]
ge.externalWeight=1
ge.exec.atomicCleanPolicy=1
ge.event=notify
ge.exec.staticMemoryPolicy=2
ge.exec.formatMode=1
ge.exec.precision_mode=must_keep_origin_dtype

[graph_kernel_param]
opt_level=2
enable_cce_lib=true
disable_cce_lib_ops=MatMul
disable_cluster_ops=MatMul,Reshape

[ge_graph_options]
ge.inputShape=batch_valid_length:1;slot_mapping:-1;tokens:1,-1
ge.dynamicDims=64,64;128,128;256,256;512,512;1024,1024;2048,2048;4096,4096	
#	必须包含yaml文件中model_config的seq_length的档位[1024,2048,4096]
ge.dynamicNodeType=1
```

###### decode_ini

```
[ascend_context]
provider=ge

[ge_session_options]
ge.externalWeight=1
ge.exec.atomicCleanPolicy=1
ge.event=notify
ge.exec.staticMemoryPolicy=2
ge.exec.formatMode=1
ge.exec.precision_mode=must_keep_origin_dtype

[graph_kernel_param]
opt_level=2
enable_cce_lib=true
disable_cce_lib_ops=MatMul
disable_cluster_ops=MatMul,Reshape

[ge_graph_options]
ge.inputShape=batch_valid_length:-1;block_tables:-1,256;slot_mapping:-1;tokens:-1,1		
#	block_tables中的256是根据yaml配置中的pa_config下的decode_seq_length/block_size得来
ge.dynamicDims=1,1,1,1;2,2,2,2;8,8,8,8;16,16,16,16;64,64,64,64	
#	ge.inputShape中有几个“-1”，便每组有几个数（有4个-1，所以有4个1、4个2、4个...），且必须包含yaml配置中的pa_config下decode_batch_size的batch数
ge.dynamicNodeType=1
```



##### 不带PagedAttention配置

```
model_path:
    prefill_model: ["/path/to/llama_pa_models/no_act/output_no_act_len/output/mindir_full_checkpoint/rank_0_graph.mindir"]
    decode_model: ["/path/to/llama_pa_models/no_act/output_no_act_len/output/mindir_inc_checkpoint/rank_0_graph.mindir"]
    argmax_model: "/path/to/serving_dev/extends_13b/argmax.mindir"
    topk_model: "/path/to/serving_dev/extends_13b/topk.mindir"
    prefill_ini: ["/path/to/llama_pa_models/no_act/ini/910b_default_prefill.cfg"]
    decode_ini: ["/path/to/llama_pa_models/no_act/ini/910_inc.cfg"]
    post_model_ini: "/path/to/baichuan/congfig/config.ini"
model_config:
    model_name: 'llama_dyn'
    max_generate_length: 8192
    end_token: 2
    seq_length: [512, 1024]
    vocab_size: 32000
    prefill_batch_size: [64]			#不支持多分档，只支持多batch
    decode_batch_size: [1,4,8,16,30,64]	#支持多分档
    zactivate_len: [512]
    model_type: "dyn"	#若无此字段默认为“dyn”，若有此字段需指定model_type
    current_index: False
    model_dtype: "DataType.FLOAT32"
    pad_token_id: 0  
    
serving_config:
    agent_ports: [11330]
    start_device_id: 5
    server_ip: 'localhost'
    server_port: 19200
    
tokenizer:
    type: LlamaTokenizer
    vocab_file: '/path/to/llama_pa_models/output/tokenizer_llama2_13b.model'

basic_inputs:
    type: LlamaBasicInputs

extra_inputs:
    type: LlamaExtraInputs

warmup_inputs:
    type: LlamaWarmupInputs
```

###### prefill_ini

```
[ge_session_options]
ge.externalWeight=1
ge.exec.atomicCleanPolicy=1
ge.event=notify
ge.exec.staticMemoryPolicy=2
ge.exec.formatMode=1
ge.exec.precision_mode=must_keep_origin_dtype
```

###### decode_ini

```
[ascend_context]
provider=ge
[ge_session_options]
ge.externalWeight=1
ge.exec.atomicCleanPolicy=1
ge.event=notify
ge.exec.staticMemoryPolicy=2
ge.exec.formatMode=1
ge.exec.precision_mode=must_keep_origin_dtype
[ge_graph_options]
ge.inputShape=batch_index:-1;batch_valid_length:-1;tokens:-1,1;zactivate_len:-1
ge.dynamicDims=1,1,1,512;4,4,4,512;8,8,8,512;16,16,16,512;30,30,30,512;64,64,64,512
#	根据yaml配置中的model_config下的decode_batch_size确定decode档位；“512”根据yaml中的zactivate_len得到
ge.dynamicNodeType=1
```


##### WizardCoder配置（静态shape）

```
model_path:
    prefill_model: ["/path/to/prefill_model.mindir"]
    decode_model: ["/path/to/decode_model.mindir"]
    argmax_model: "/path/to/argmax.mindir"
    topk_model: "/path/to/topk.mindir"
    prefill_ini : ['/path/to/lite.ini']
    decode_ini: ['/path/to/lite.ini']
    post_model_ini: '/path/to/lite.ini'

model_config:
    model_name: 'llama_dyn'
    max_generate_length: 4096
    end_token: 0
    seq_length: [2048]
    vocab_size: 49153
    prefill_batch_size: [1]
    decode_batch_size: [1]
    zactivate_len: [2048]
    model_type: 'static'
    seq_type: 'static'
    batch_waiting_time: 0.0
    decode_batch_waiting_time: 0.0
    batching_strategy: 'continuous'
    current_index: False
    page_attention: False
    model_dtype: "DataType.FLOAT32"
    pad_token_id: 49152

serving_config:
    agent_ports: [9980]
    start_device_id: 0
    server_ip: 'localhost'
    server_port: 12359

tokenizer:
    type: WizardCoderTokenizer
    vocab_file: '/path/to/transformers_config'

basic_inputs:
    type: LlamaBasicInputs

extra_inputs:
    type: LlamaExtraInputs

warmup_inputs:
    type: LlamaWarmupInputs
```

###### lite_ini

```
[ascend_context]
plugin_custom_ops=All
provider=ge
[ge_session_options]
ge.exec.formatMode=1
ge.exec.precision_mode=must_keep_origin_dtype
ge.externalWeight=1
ge.exec.atomicCleanPolicy=1
```

##### kbk 场景 yaml 配置

###### 带PA yaml 配置

```
model_path:
    prefill_model: ["/path/llama2-13b-mindir/full_graph.mindir"]
    decode_model: ["/path/llama2-13b-mindir/inc_graph.mindir"]
    argmax_model: "/path/post_process/argmax.mindir"
    topk_model: "/path/post_process/topk.mindir"
    prefill_ini : ['/path/llma2_13b_pa_dyn_prefill.ini']
    decode_ini: ['/path/llma2_13b_pa_dyn_decode.ini']
    post_model_ini: '/path/post_config.ini'

model_config:
    model_name: 'llama_7b'
    max_generate_length: 4096
    end_token: 2
    seq_length: [4096]
    vocab_size: 32000
    prefill_batch_size: [1]
    decode_batch_size: [1]
    zactivate_len: [512, 1024, 2048, 4096]
    model_type: 'dyn'
    seq_type: 'static'
    batch_waiting_time: 0.0
    decode_batch_waiting_time: 0.0
    batching_strategy: 'continuous'
    current_index: False
    page_attention: True
    model_dtype: "DataType.FLOAT32"
    pad_token_id: 0
    backend: 'kbk' # 'ge'
    model_cfg_path: 'checkpoint_download/llama/llama_7b.yaml'

serving_config:
    agent_ports: [14002]
    start_device_id: 7
    server_ip: 'localhost'
    server_port: 19291

pa_config:
    num_blocks: 2048
    block_size: 16
    decode_seq_length: 4096

tokenizer:
    type: LlamaTokenizer
    vocab_file: '/home/wsc/llama/tokenizer/tokenizer.model'

basic_inputs:
    type: LlamaBasicInputs

extra_inputs:
    type: LlamaExtraInputs

warmup_inputs:
    type: LlamaWarmupInputs
```

#### 设置环境变量，变量配置如下

######  方式一：使用已有脚本启动

```
source /path/to/xxx-serving.sh
```

```
export PYTHONPATH=/path/to/mindformers-ft-2:/path/to/serving/:$PYTHONPATH
```

###### 	    		方式二：镜像

下载好docker镜像后创建容器

```
# --device用于控制指定容器的运行NPU卡号和范围
# -v 用于映射容器外的目录
# --name 用于自定义容器名称
# /bin/bash前的是镜像ID，可以用指令docker images查看

docker run -it -u root \
--ipc=host \
--network host \
--device=/dev/davinci0 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /etc/localtime:/etc/localtime \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /var/log/npu/:/usr/slog \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
--name {请手动输入容器名称} \
XXX /bin/bash
```

```
export PYTHONPATH=/path/to/mindformers-ft-2:/path/to/serving/:$PYTHONPATH
```

#### 启动

```shell
python examplse/start.py --config configs/xxx.yaml# 先后拉起模型和serving进程
```
启动参数：config: 模型对应的yaml文件, refer to [model.yaml](configs/internLM_dyn.yaml)

#### 发起请求
通过“/models/model_name/generate”和"/models/model_name/generate_stream" 进行请求

```shell
curl 127.0.0.1:9800/models/llama2/generate \
     -X POST \
     -d '{"inputs":"Hello?","parameters":{"max_new_tokens":55, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' \
     -H 'Content-Type: application/json'
```

```shell
curl 127.0.0.1:9800/models/llama2/generate_stream \
     -X POST \
     -d '{"inputs":"Hello?","parameters":{"max_new_tokens":55, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' \
     -H 'Content-Type: application/json'
```

或者通过python API

```python
from mindspore_serving.client import MindsporeInferenceClient

client = MindsporeInferenceClient(model_type="llama2", server_url="http://127.0.0.1:8080")

# 1. test generate
text = client.generate("what is Monetary Policy?").generated_text
print('text: ', text)

# 2. test generate_stream
text = ""
for response in client.generate_stream("what is Monetary Policy?", do_sample=False, max_new_tokens=200):
    print("response 0", response)
    if response.token:
        text += response.token.text
    else:
        text = response.generated_text
print(text)
```
