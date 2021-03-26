## Release 1.2.0-rc1

### Contributors

### Major Features and Improvements

- [STABLE] Support distributed inference, it needs to cooperate with distributed training to export distributed models for super-large-scale neural network parameters(Ascend 910).
- [STABLE] Support GPU platform, Serving worker nodes can be deployer on Nvidia GPU, Ascend 310 and Ascend 910.
- [STABLE] This release is based on MindSpore version 1.2.0

### API Change

#### API Incompatible Change

##### Python API

#### Deprecations

##### Python API

## Release 1.1.0

### Contributors

### Major Features and Improvements

- [STABLE] Support gRPC and RESTful API.
- [STABLE] Support simple Python API for Client and Server.
- [STABLE] Support Model configuration，User can customize preprocessing & postprocessing for model.
- [STABLE] Support multiple models，Multiple models can run simultaneously.
- [STABLE] Support Model batching，Multiple instances will be split and combined to meet the batch size requirements of the model.
- [STABLE] This release is based on MindSpore version 1.1.0
