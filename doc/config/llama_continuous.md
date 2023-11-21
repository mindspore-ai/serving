# Llama2 70B continusou batching config

- Baseconfig配置：
    - seq_length:[310, 600, 1024, 2048]
    - vocab_size: 32000
    - batch_size: 8
    - prefill_batch_size: 1
    - model_type: 0
    - batching_strategy: 'continuous'
    - tokenizer: 'LlamaTokenizer',
    - tokenizer_path: './checkpoint_download/llama/tokenizer.model'
    - input_function: 'custom'
    
- AgentConfig配置：
    - 模型MindIR路径、配置文件路径、后处理模型及配置路径等按需配置；
    - npu_nums: 4
    - AgentPorts:[7000, 7001, 7002, 7003] List len=4
    
    
- ModelName: "llama_dyn"