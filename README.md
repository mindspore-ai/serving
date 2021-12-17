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

MindSpore Serving is a lightweight and high-performance service module that helps MindSpore developers efficiently
deploy online inference services in the production environment. After completing model training on MindSpore, you can
export the MindSpore model and use MindSpore Serving to create an inference service for the model.

MindSpore Serving architecture:

<img src="docs/architecture.png" alt="MindSpore Architecture" width="600"/>

MindSpore Serving includes two parts: `Client` and `Server`. On a `Client` node, you can deliver inference service
commands through the gRPC or RESTful API. The `Server` consists of a `Main` node and one or more `Worker` nodes.
The `Main` node manages all `Worker` nodes and their model information, accepts user requests from `Client`s, and
distributes the requests to `Worker` nodes. `Servable` is deployed on a worker node, indicates a single model or a
combination of multiple models and can provide different services in various methods. `

On the server side, when [MindSpore](#https://www.mindspore.cn/) is used as the inference backend,, MindSpore Serving
supports the Ascend 910/710/310 and Nvidia GPU environments. When [MindSpore Lite](#https://www.mindspore.cn/lite) is
used as the inference backend, MindSpore Serving supports Ascend 310, Nvidia GPU and CPU environments. Client` does not
depend on specific hardware platforms.

MindSpore Serving provides the following functions:

- gRPC and RESTful APIs on clients
- Pre-processing and post-processing of assembled models
- Batch. Multiple instance requests are split and combined to meet the `batch size` requirement of the model.
- Simple Python APIs on clients
- The multi-model combination is supported. The multi-model combination and single-model scenarios use the same set of
  interfaces.
- Distributed model inference

## Installation

### Installing MindSpore or MindSpore Lite

MindSpore Serving depends on the MindSpore or MindSpore Lite inference framework. We select one of them as the Serving
Inference backend:

- MindSpore

  [Install MindSpore](https://gitee.com/mindspore/mindspore/blob/master/README.md#installation)，and configure
  [environment variables](https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend_install_source_en.md#configuring-environment-variables).

- MindSpore Lite

  For details about how to compile and install MindSpore Lite, see the [MindSpore Lite Documentation](https://www.mindspore.cn/lite/docs/en/master/index.html).
  We should configure the environment variable `LD_LIBRARY_PATH` to indicates the installation path of `libmindspore-lite.so`.

### Installing Serving

Perform the following steps to install Serving:

- If use the pip command, download the .whl package from
  the [MindSpore Serving page](https://www.mindspore.cn/versions/en) and install it.

    ```shell
    pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/Serving/{arch}/mindspore_serving-{version}-{python_version}-linux_{arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

    > - `{version}` denotes the version of MindSpore Serving. For example, when you are downloading MindSpore Serving 1.1.0, `{version}` should be 1.1.0.
    > - `{arch}` denotes the system architecture. For example, the Linux system you are using is x86 architecture 64-bit, `{arch}` should be `x86_64`. If the system is ARM architecture 64-bit, then it should be `aarch64`.
    > - `{python_version}` specifies the python version for which MindSpore Serving is built. If you wish to use Python3.7.5,`{python_version}` should be `cp37-cp37m`. If Python3.9.0 is used, it should be `cp39-cp39`. Please use the same Python environment whereby MindSpore Serving is installed.

- Install Serving using the [source code](https://gitee.com/mindspore/serving).

    ```shell
    git clone https://gitee.com/mindspore/serving.git -b master
    cd serving
    bash build.sh
    ```

  For the `bash build.sh` above, we can add `-jn`, for example `-j16`, to accelerate compilation. By adding `-S on`
  option, third-party dependencies can be downloaded from gitee instead of github.

  After the build is complete, find the .whl installation package of Serving in the `serving/build/package/` directory
  and install it.

    ```python
    pip install mindspore_serving-{version}-cp37-cp37m-linux_{arch}.whl
    ```

Run the following commands to verify the installation. Import the Python module. If no error is reported, the
installation is successful.

```python
from mindspore_serving import master
from mindspore_serving import worker
```

## Quick Start

[MindSpore-based Inference Service Deployment](https://www.mindspore.cn/serving/docs/en/master/serving_example.html) is
used to demonstrate how to use MindSpore Serving.

## Documents

### Developer Guide

- [gRPC-based MindSpore Serving Access](https://www.mindspore.cn/serving/docs/en/master/serving_grpc.html)
- [RESTful-based MindSpore Serving Access](https://www.mindspore.cn/serving/docs/en/master/serving_restful.html)
- [Services Provided Through Model Configuration](https://www.mindspore.cn/serving/docs/en/master/serving_model.html)
- [Services Composed of Multiple Models](https://www.mindspore.cn/serving/docs/en/master/serving_model.html#services-composed-of-multiple-models)
- [MindSpore Serving-based Distributed Inference Service Deployment](https://www.mindspore.cn/serving/docs/en/master/serving_distributed_example.html)

For more details about the installation guide, tutorials, and APIs,
see [MindSpore Python API](https://www.mindspore.cn/serving/api/en/master/index.html).

## Community

### Governance

[MindSpore Open Governance](https://gitee.com/mindspore/community/blob/master/governance.md)

### Communication

- [MindSpore Slack](https://join.slack.com/t/mindspore/shared_invite/zt-dgk65rli-3ex4xvS4wHX7UDmsQmfu8w) developer
  communication platform

## Contributions

Welcome to MindSpore contribution.

## Release Notes

[RELEASE](RELEASE.md)

## License

[Apache License 2.0](LICENSE)
