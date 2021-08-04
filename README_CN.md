# MindSpore Serving

[View English](./README.md)

<!-- TOC -->

- [MindSpore Serving](#mindspore-serving)
    - [概述](#概述)
    - [安装](#安装)
        - [安装Serving](#安装serving)
        - [配置环境变量](#配置环境变量)
    - [快速入门](#快速入门)
    - [文档](#文档)
        - [开发者教程](#开发者教程)
    - [社区](#社区)
        - [治理](#治理)
        - [交流](#交流)
    - [贡献](#贡献)
    - [版本说明](#版本说明)
    - [许可证](#许可证)

<!-- /TOC -->

## 概述

MindSpore Serving是一个轻量级、高性能的服务模块，旨在帮助MindSpore开发者在生产环境中高效部署在线推理服务。当用户使用MindSpore完成模型训练后，导出MindSpore模型，即可使用MindSpore Serving创建该模型的推理服务。  

MindSpore Serving架构：

MindSpore Serving分为客户端、服务器两个部分。在客户端中，用户通过gRPC或RESTful接口向服务器下发推理服务命令。服务器包括主（`Main`）节点和一个或多个工作（`Worker`）节点，主节点管理所有的工作节点及其部署的模型信息，接受客户端的用户请求，并将请求分发给工作节点。每个工作节点部署了一个可服务对象，即`Servable`，这里的`Servable`可以是单个模型，也可以是多个模型的组合，一个`Servable`可以围绕相同的模型通过多种方法来提供不同的服务。客户端不依赖特定硬件平台，服务器支持GPU、Ascend 310和Ascend 910平台，后续会逐步支持CPU场景。  

<img src="docs/architecture.png" alt="MindSpore Architecture" width="600"/>

MindSpore Serving提供以下功能：

- 支持客户端gRPC和RESTful接口。
- 支持组装模型的前处理和后处理。
- 支持batch功能，多实例请求会被拆分组合以满足模型`batch size`的需要。
- 提供客户端Python简易接口。
- 支持分布式模型推理功能。

## 安装

MindSpore Serving依赖MindSpore训练推理框架，安装完[MindSpore](https://gitee.com/mindspore/mindspore#%E5%AE%89%E8%A3%85) ，再安装MindSpore Serving。

### 安装Serving

安装方式如下：

- 使用pip命令安装，请从[MindSpore Serving下载页面](https://www.mindspore.cn/versions)下载并安装whl包。

    ```python
    pip install mindspore_serving-{version}-cp37-cp37m-linux_{arch}.whl
    ```

    > - `{version}`表示MindSpore Serving版本号，例如下载1.1.0版本MindSpore Serving时，`{version}`应写为1.1.0。
    > - `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为`x86_64`。如果系统是ARM架构64位，则写为`aarch64`。

- 源码编译安装。

    下载[源码](https://gitee.com/mindspore/serving)，下载后进入`serving`目录。

    方式一，指定Serving依赖的已安装或编译的MindSpore包路径，安装Serving：

    ```shell
    sh build.sh -p $MINDSPORE_LIB_PATH
    ```

    其中，`build.sh`为`serving`目录下的编译脚本文件，`$MINDSPORE_LIB_PATH`为MindSpore软件包的安装路径下的`lib`路径，例如，`softwarepath/mindspore/lib`，该路径包含MindSpore运行依赖的库文件。

    方式二，直接编译Serving，编译时会配套编译MindSpore的包，需要配置MindSpore编译时的[环境变量](https://gitee.com/mindspore/docs/blob/r1.3/install/mindspore_ascend_install_source.md#配置环境变量) ：

    ```shell
    # GPU
    sh build.sh -e gpu
    # Ascend 310 and Ascend 910
    sh build.sh -e ascend
    ```

    其中，`build.sh`为`serving`目录下的编译脚本文件，编译完后，在`serving/third_party/mindspore/build/package/`目录下找到MindSpore的whl安装包进行安装：

    ```python
    pip install mindspore_ascend-{version}-cp37-cp37m-linux_{arch}.whl
    ```

    同时在`serving/build/package/`目录下找到Serving的whl安装包进行安装：

    ```python
    pip install mindspore_serving-{version}-cp37-cp37m-linux_{arch}.whl
    ```

执行以下命令，验证安装结果。导入Python模块不报错即安装成功：

```python
from mindspore_serving import server
```

### 配置环境变量

MindSpore Serving运行需要配置以下环境变量：

- MindSpore Serving依赖MindSpore正确运行，运行MindSpore需要完成[环境变量配置](https://gitee.com/mindspore/docs/blob/r1.3/install/mindspore_ascend_install_pip.md#%E9%85%8D%E7%BD%AE%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F)。

## 快速入门

以一个简单的[Add网络示例](https://www.mindspore.cn/serving/docs/zh-CN/r1.3/serving_example.html)，演示MindSpore Serving如何使用。

## 文档

### 开发者教程

- [基于gRPC接口访问MindSpore Serving服务](https://www.mindspore.cn/serving/docs/zh-CN/r1.3/serving_grpc.html)
- [基于RESTful接口访问MindSpore Serving服务](https://www.mindspore.cn/serving/docs/zh-CN/r1.3/serving_restful.html)
- [通过配置模型提供Servable](https://www.mindspore.cn/serving/docs/zh-CN/r1.3/serving_model.html)
- [基于MindSpore Serving部署分布式推理服务](https://www.mindspore.cn/serving/docs/zh-CN/r1.3/serving_distributed_example.html)

有关安装指南、教程和API的更多详细信息，请参阅[用户文档](https://www.mindspore.cn/docs/api/zh-CN/r1.3/index.html)。

## 社区

### 治理

查看MindSpore如何进行[开放治理](https://gitee.com/mindspore/community/blob/master/governance.md)。

### 交流

- [MindSpore Slack](https://join.slack.com/t/mindspore/shared_invite/zt-dgk65rli-3ex4xvS4wHX7UDmsQmfu8w) 开发者交流平台。

## 贡献

欢迎参与贡献。

## 版本说明

版本说明请参阅[RELEASE](RELEASE.md)。

## 许可证

[Apache License 2.0](LICENSE)
