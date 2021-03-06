# MindSpore Serving

[查看中文](./README_CN.md)

<!-- TOC -->

- [MindSpore Serving](#mindspore-serving)
    - [Overview](#overview)
    - [Installation](#installation)
        - [Installing Serving](#installing-serving)
        - [Configuring Environment Variables](#configuring-environment-variables)
    - [Quick Start](#quick-start)
    - [Documents](#documents)
        - [Developer Guide](#developer-guide)
    - [Community](#community)
        - [Governance](#governance)
        - [Communication](#communication)
    - [Contributions](#contributions)
    - [Release Notes](#release-notes)
    - [License](#license)

<!-- /TOC -->

## Overview

MindSpore Serving is a lightweight and high-performance service module that helps MindSpore developers efficiently deploy online inference services in the production environment. After completing model training on MindSpore, you can export the MindSpore model and use MindSpore Serving to create an inference service for the model.  

MindSpore Serving architecture:

MindSpore Serving includes two parts: `Client` and `Server`. On a `Client` node, you can deliver inference service commands through the gRPC or RESTful API. The `Server` consists of a `Main` node and one or more `Worker` nodes. The `Main` node manages all `Worker` nodes and their model information, accepts user requests from `Client`s, and distributes the requests to `Worker` nodes. `Servable` is deployed on a worker node, indicates a single model or a combination of multiple models and can provide different services in various methods. `Client` does not depend on specific hardware platforms. The `Server` node supports GPU, the Ascend 310 and Ascend 910 platforms. CPU will be supported in the future.  

<img src="docs/architecture.png" alt="MindSpore Architecture" width="600"/>

MindSpore Serving provides the following functions:

- gRPC and RESTful APIs on clients
- Pre-processing and post-processing of assembled models
- Batch. Multiple instance requests are split and combined to meet the `batch size` requirement of the model.
- Simple Python APIs on clients
- Distributed model inference

## Installation

MindSpore Serving depends on the MindSpore training and inference framework. Therefore, install [MindSpore](https://gitee.com/mindspore/mindspore/blob/master/README.md#installation) and then MindSpore Serving.

### Installing Serving

Perform the following steps to install Serving:

- If use the pip command, download the .whl package from the [MindSpore Serving page](https://www.mindspore.cn/versions/en) and install it.

    ```python
    pip install mindspore_serving-{version}-cp37-cp37m-linux_{arch}.whl
    ```

    > - `{version}` denotes the version of MindSpore Serving. For example, when you are downloading MindSpore Serving 1.1.0, `{version}` should be 1.1.0.
    > - `{arch}` denotes the system architecture. For example, the Linux system you are using is x86 architecture 64-bit, `{arch}` should be `x86_64`. If the system is ARM architecture 64-bit, then it should be `aarch64`.

- Install Serving using the source code.

    Download the [source code](https://gitee.com/mindspore/serving) and go to the `serving` directory.

    Method 1: Specify the path of the installed or built MindSpore package on which Serving depends and install Serving.

    ```shell
    sh build.sh -p $MINDSPORE_LIB_PATH
    ```

    In the preceding information, `build.sh` is the build script file in the `serving` directory, and `$MINDSPORE_LIB_PATH` is the `lib` directory in the installation path of the MindSpore software package, for example, `softwarepath/mindspore/lib`. This path contains the library files on which MindSpore depends.

    Method 2: Directly build Serving. The MindSpore package is built together with Serving. You need to configure the [environment variables](https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend_install_source_en.md#configuring-environment-variables) for MindSpore building.

    ```shell
    # GPU
    sh build.sh -e gpu
    # Ascend 310 and Ascend 910
    sh build.sh -e ascend
    ```

    In the preceding information, `build.sh` is the build script file in the `serving` directory. After the build is complete, find the .whl installation package of MindSpore in the `serving/third_party/mindspore/build/package/` directory and install it.

    ```python
    pip install mindspore_ascend-{version}-cp37-cp37m-linux_{arch}.whl
    ```

    Find the .whl installation package of Serving in the `serving/build/package/` directory and install it.

    ```python
    pip install mindspore_serving-{version}-cp37-cp37m-linux_{arch}.whl
    ```

Run the following commands to verify the installation. Import the Python module. If no error is reported, the installation is successful.

```python
from mindspore_serving import master
from mindspore_serving import worker
```

### Configuring Environment Variables

To run MindSpore Serving, configure the following environment variables:

- MindSpore Serving depends on MindSpore. You need to configure [environment variables](https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend_install_source_en.md#configuring-environment-variables) to run MindSpore.

## Quick Start

[MindSpore-based Inference Service Deployment](https://www.mindspore.cn/serving/docs/en/master/serving_example.html) is used to demonstrate how to use MindSpore Serving.

## Documents

### Developer Guide

- [gRPC-based MindSpore Serving Access](https://www.mindspore.cn/serving/docs/en/master/serving_grpc.html)
- [RESTful-based MindSpore Serving Access](https://www.mindspore.cn/serving/docs/en/master/serving_restful.html)
- [Servable Provided Through Model Configuration](https://www.mindspore.cn/serving/docs/en/master/serving_model.html)
- [MindSpore Serving-based Distributed Inference Service Deployment](https://www.mindspore.cn/serving/docs/en/master/serving_distributed_example.html)

For more details about the installation guide, tutorials, and APIs, see [MindSpore Python API](https://www.mindspore.cn/serving/api/en/master/index.html).

## Community

### Governance

[MindSpore Open Governance](https://gitee.com/mindspore/community/blob/master/governance.md)

### Communication

- [MindSpore Slack](https://join.slack.com/t/mindspore/shared_invite/zt-dgk65rli-3ex4xvS4wHX7UDmsQmfu8w) developer communication platform

## Contributions

Welcome to MindSpore contribution.

## Release Notes

[RELEASE](RELEASE.md)

## License

[Apache License 2.0](LICENSE)
