# LLL-Serving

- High Perfomance LLM Inference Serving based on MindSpore-Lite
- Concurrency

# 功能介绍

# 使用说明
- 基本设置:
  - device: NPU的起始序号; 
  - tokenizer_path: 分词器模型路径; 
  - ModelName：模型名称，用于构造mindspore_lite模型输入; 
  - AgentIP: Agent的IP地址; 
  - TOPP_NUM：top_p的默认参数，5000;
  

- Agent配置:
  - 在config/serving_config.py中配置-AgentConfig:
  - ctx_setting: 全量模型配置的路径;
  - inc_setting: 增量模型配置路径;
  - post_model_setting: 后处理模型配置的路径;
  - npu_nums: 并行所用的卡数;
  - prefill_model: 全量模型MINDIR路径;
  - decode_model: 增量模型MINDIR路径;
  - argmax_model : argmax模型MINDIR路径topk_model: topk（k=100）模型MINDIR路径;
  - AgentPorts: 端口号

- 模型配置：在config/serving_config.py中配置BaseConfig:
  - vocab_size : 词表长度 
  - batch_size：批处理大小 
  - model_type：模型类型，静态输入模型，动态分档模型;
  - end_token: 词表的结束token

- 启动顺序：
  - 0、打开终端，设置环境变量env.sh、激活conda 环境、按需填充serving_config.py中的所有配置;
  - 1、运行post_sampling_model.py导出后处理模型；（包括模型路径、模型配置路径）;
  - 2、导出多卡的llama2 70B模型，写入serving_config.py文件（包括模型路径、模型配置路径）;
  - 3、serving_config.py中设置AgentIP, AgentPorts;
  - 4、启动test_agent.py脚本，拉起MindSpore-lite分布式推理服务;
  - 5、拉起agent服务成功后，打开新终端，设置环境，及conda, python client/server_app_post.py, 拉起LLM-Serving服务;
  - 6、server服务启动成功后，打开新终端，启动python client/test_client_post.py,发送请求;
