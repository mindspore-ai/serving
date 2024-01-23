# MindSpore Serving Release Notes

## MindSpore Serving 2.1.0 Release Notes

### Major Features and Improvements

#### OS
- [STABLE] add support for Python 3.9
- [STABLE] add support for MindSpore 2.2.10
- [STABLE] add support for MindSpore Lite 2.2.10

#### LLMs
- [STABLE] add support for LLaMA2
- [STABLE] add support for InternLM

#### Inference Performance
- [STABLE] adapt to model with dynamic sequence length to avoid redundant input tokens padding
- [STABLE] adjust the batch size based on the length of request queue to avoid fixed batch padding
- [STABLE] support model parallel on multiple NPUs to make efficient use of device memory and computing power
  
#### Throughput
- [STABLE] continuous batching of incoming requests to make better NPU utilization
  
#### Others
- [STABLE] support token streaming using Server-Sent Events (SSE) for progressive generation
- [BETA] provide launch script for convenience [start.py](./examples/start.py)

#### 

### Contributors

Thanks goes to these wonderful people:

zhoupengyuan, shuchi, guoshipeng, tiankai, zhengyi, pengkang

Contributions of any kind are welcome!
