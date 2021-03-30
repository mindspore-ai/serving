# MindSpore Serving 1.2.0-rc1

## MindSpore Serving 1.2.0 Release Notes

### Major Features and Improvements

- [STABLE] Support distributed inference, it needs to cooperate with distributed training to export distributed models for super-large-scale neural network parameters(Ascend 910).
- [STABLE] Support GPU platform, Serving worker nodes can be deployer on Nvidia GPU, Ascend 310 and Ascend 910.
- This release is based on MindSpore version 1.2.0
- Support Python 3.8 and 3.9.

### API Change

#### API Incompatible Change

##### Python API

Support deployment of distributed model, refer to [distributed inference tutorial](https://www.mindspore.cn/tutorial/inference/en/r1.2/serving_distributed_example.html) for related API.

#### Deprecations

##### Python API

### Bug Fixes

## Contributors

Thanks goes to these wonderful people:

chenweifeng, qinzheng, xujincai, xuyongfei, zhangyinxia, zhoufeng.

Contributions of any kind are welcome!

## MindSpore Serving 1.1.1 Release Notes

## Major Features and Improvements

- Adapts new C++ inference interface for MindSpore version 1.1.1.

## Bug fixes

- [BUGFIX] Fix bug in transforming result of type int16 in python Client.
- [BUGFIX] Fix bytes type misidentified as str type after python preprocess and postprocess.
- [BUGFIX] Fix bug releasing C++ tensor data when it's wrapped as numpy object sometimes.
- [BUGFIX] Update RuntimeError to warning log when check Ascend environment failed.

## MindSpore Serving 1.1.0 Release Notes

### Major Features and Improvements

- [STABLE] Support gRPC and RESTful API.
- [STABLE] Support simple Python API for Client and Server.
- [STABLE] Support Model configuration，User can customize preprocessing & postprocessing for model.
- [STABLE] Support multiple models，Multiple models can run simultaneously.
- [STABLE] Support Model batching，Multiple instances will be split and combined to meet the batch size requirements of the model.
- This release is based on MindSpore version 1.1.0

### Bug Fixes

### Contributors
