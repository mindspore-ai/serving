
# 1. MindSpore Serving 1.2.0 Release Notes

## 1.1. Major Features and Improvements

### 1.1.1. Serving Framework

Support distributed inference, it needs to cooperate with distributed training to export distributed models for super-large-scale neural network parameters.
Support GPU platform, Serving worker nodes can be deployer on GPU, Ascend 310 and Ascend 910.

# 2. MindSpore Serving 1.1.0 Release Notes

## 2.1. Major Features and Improvements

### 2.1.1. Ascend 310 & Ascend 910 Serving Framework

Support gRPC and RESTful API.
Support simple Python API for Client and Server.
Support Model configuration，User can customize preprocessing & postprocessing for model.
Support multiple models，Multiple models can run simultaneously.
Support Model batching，Multiple instances will be split and combined to meet the batch size requirements of the model.
