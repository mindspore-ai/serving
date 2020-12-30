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

Currently, the MindSpore Serving nodes include client, master, and worker. On a client node, you can directly deliver inference service commands through the gRPC or RESTful API. Servable model service is deployed on a worker node. Servable indicates a single model or a combination of multiple models and provides different services in various methods. A master node manages all worker nodes and their model information, as well as managing and distributing tasks. The master and worker nodes can be deployed in the same process or in different processes. Currently, the client and master nodes do not depend on specific hardware platforms. The worker node supports only the Ascend 310 and Ascend 910 platforms. GPUs and CPUs will be supported in the future.  
<img src="docs/architecture.png" alt="MindSpore Architecture" width="600"/>

MindSpore Serving provides the following functions:

- gRPC and RESTful APIs on clients
- Pre-processing and post-processing of assembled models
- Batch. Multiple instance requests are split and combined to meet the `batch size` requirement of the model.
- Simple Python APIs on clients

## Installation

MindSpore Serving depends on the MindSpore training and inference framework. Therefore, install [MindSpore](https://gitee.com/mindspore/mindspore/blob/master/README.md#installation) and then MindSpore Serving.

### Installing Serving

Use the pip command to install Serving. Perform the following steps:

- Download the .whl package from the [MindSpore Serving page](https://www.mindspore.cn/versions/en) and install it.

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
    # Ascend 310
    sh build.sh -e ascend -V 310
    # Ascend 910
    sh build.sh -e ascend -V 910
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

- MindSpore Serving depends on the MindSpore library file. You need to specify the `lib` path in the build or installation path of the MindSpore software package to LD_LIBRARY_PATH. For the meaning of `$MINDSPORE_LIB_PATH`, see [MindSpore-based Inference Service Deployment](https://gitee.com/mindspore/serving/blob/master/README.md#installation).

    ```shell
    export LD_LIBRARY_PATH=$MINDSPORE_LIB_PATH:${LD_LIBRARY_PATH}
    ```

## Quick Start

[MindSpore-based Inference Service Deployment](https://www.mindspore.cn/tutorial/inference/en/master/serving_example.html) is used to demonstrate how to use MindSpore Serving.

## Documents

### Developer Guide

- [gRPC-based MindSpore Serving Access](https://www.mindspore.cn/tutorial/inference/en/master/serving_grpc.html)
- [RESTful-based MindSpore Serving Access](https://www.mindspore.cn/tutorial/inference/en/master/serving_restful.html)
- [Servable Provided Through Model Configuration](https://www.mindspore.cn/tutorial/inference/en/master/serving_model.html)

For more details about the installation guide, tutorials, and APIs, see [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html).

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