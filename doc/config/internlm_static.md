# InternLM static batching config

- Baseconfig配置：
    - seq_length:[2048]
    - vocab_size: 103168
    - batch_size: 1
    - prefill_batch_size: 1
    - model_type: 1
    - batching_strategy: 'static'
    - tokenizer: 'InternLMTokenizer',
    - tokenizer_path: './checkpoint_download/internlm/tokenizer.model'
    - input_function: 'custom'
    
- AgentConfig配置：
    - 模型MindIR路径、配置文件路径、后处理模型及配置路径等按需配置；
    - npu_nums: 1
    - AgentPorts: [7000], List len=1
    
- ModelName: "internlm_7b"