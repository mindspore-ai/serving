# MindSpore Serving

### serving is a fast and easy-to-use inference framework

---
### Features
- model parallel deployment
- token streaming via Server-Sent Events
- custom model inputs
- static/continuous batching of prompts
- do post sampling via npu

### Supports the most popular LLMs, including the following architectures:
- LLaMA-2
- InternLM

### Get Started
#### 环境依赖
- python 3.7
- [mindspore](https://www.mindspore.cn/install)
- [mindspore-lite](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html?highlight=%E5%AE%89%E8%A3%85) 
- [mindformers](https://gitee.com/mindspore/mindformers#%E4%BA%8Cmindformers%E5%AE%89%E8%A3%85)
- easydict

#### 配置文件(serving_config.py)：
    模型文件路径配置：
    "prefill_model_path" - 全量模型mindir路径，如模型切分成多份，每个mindir路径用','隔开
    "decode_model_path"  - 增量模型mindir路径
    “argmax_model”       - argmax后处理mindir路径
    "topk_model"         - topk后处理mindir路径 （这两个后处理模型可以通过post_sampling_model.py进行一键导出）
    "ctx_path"           - 全量模型对应的ini路径
    "inc_path"           - 增量模型对应的ini路径
    "post_model_ini"     - 后处理模型对应的ini路径
    "tokenizer_path"     - 大模型的分词model路径
    
    模型参数配置：
    "max_generate_length" - 最大token生成长度
    "end_token"           - eos token
    "seq_length"          - 如果模型支持动态分档，需要使用该字段
    "dyn_batch_size"      - 如果模型支持动态batch，需要使用该字段
    "seq_type"            - 如果模型支持纯动态，该字段设置为'dyn'
    "batching_strategy"   - 组batch的策略, 'continuous' / 'static'
    "input_function"      - 如用户想自定义模型入参，该字段设置为'custom'（并修改get_inputs_custom和ExtraInput函数），否则为'common'

具体模型的设置可以参考doc/config
注：后处理当前按照入图的方式进行，使用serving前请使用post_sampling_model.py重新导出后处理模型，保证数据类型与LLM模型的输出类型一致；


#### 启动
```shell
python start_agent.py
python client/server_app_post.py
```

然后可以通过“/models/model_name/generate”和"/models/model_name/generate_stream" 进行请求

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
from client import MindsporeInferenceClient

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
