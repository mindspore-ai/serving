# MindSpore Serving 1.1.1 Release Notes

## Major Features and Improvements

* Adapts new C++ inference interface for MindSpore version 1.1.1.

## Bug fixes

* [BUGFIX] Fix bug in transforming result of type int16 in python Client.
* [BUGFIX] Fix bytes type misidentified as str type after python preprocess and postprocess.
* [BUGFIX] Fix bug releasing C++ tensor data when it's wrapped as numpy object sometimes.
* [BUGFIX] Update RuntimeError to warning log when check Ascend environment failed.

# MindSpore Serving 1.1.0 Release Notes

## Major Features and Improvements

### Ascend 310 & Ascend 910 Serving Framework

* Support gRPC and RESTful API.
* Support simple Python API for Client and Server.
* Support Model configuration，User can customize preprocessing & postprocessing for model.
* Support multiple models，Multiple models can run simultaneously.
* Support Model batching，Multiple instances will be split and combined to meet the batch size requirements of the model.
