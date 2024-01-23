# MindSpore Serving Release Notes

[View English](./RELEASE.md)

## MindSpore Serving 2.1.0 Release Notes

### 主要特性和增强

#### OS
- [STABLE] 适配Python 3.9版本
- [STABLE] 适配MindSpore 2.2.10版本
- [STABLE] 适配MindSpore Lite 2.2.10版本

#### LLMs
- [STABLE] 支持LLaMA2
- [STABLE] 支持书生

#### 推理性能
- [STABLE] 适配纯动态sequence length模型，入参不再需要padding到固定长度
- [STABLE] 根据请求队列长度动态调整batch大小，不再需要padding到固定的batch大小
- [STABLE] 支持单机多卡模型并行，高效利用NPU显存和算力
  
#### 吞吐量
- [STABLE] 输入请求调度策略支持continuous batching，更好地利用NPU
  
#### 其他
- [STABLE] 支持基于SSE的流式返回，推理结果可以一个接一个地返回客户端
- [BETA] 提供启动脚本 [start.py](./examples/start.py)，降低使用成本


### 贡献者

感谢以下人员做出的贡献:

zhoupengyuan, shuchi, guoshipeng, tiankai, zhengyi, pengkang

欢迎以任何形式对项目提供贡献！
