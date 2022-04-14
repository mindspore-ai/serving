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

MindSpore Serving是一个轻量级、高性能的服务模块，旨在帮助MindSpore开发者在生产环境中高效部署在线推理服务。当用户使用MindSpore完成模型训练
后，导出MindSpore模型，即可使用MindSpore Serving创建该模型的推理服务。

MindSpore Serving架构：

<img src="docs/architecture.png" alt="MindSpore Architecture" width="600"/>

MindSpore Serving分为客户端、服务器两个部分。在客户端中，用户通过gRPC或RESTful接口向服务器下发推理服务命令。服务器包括主（`Main`）节点和
一个或多个工作（`Worker`）节点，主节点管理所有的工作节点及其部署的模型信息，接受客户端的用户请求，并将请求分发给工作节点。每个工作节点部署了
一个可服务对象，即`Servable`，这里的`Servable`可以是单个模型，也可以是多个模型的组合，一个`Servable`可以围绕相同的模型通过多种方法来提供
不同的服务。

对于服务端，当以[MindSpore](#https://www.mindspore.cn/)作为推理后端时，MindSpore Serving当前支持Ascend 910/710/310和Nvidia
GPU环境。当以[MindSpore Lite](#https://www.mindspore.cn/lite)作为推理后端时，MindSpore Serving当前支持Ascend 310、Nvidia
GPU和CPU。客户端不依赖特定硬件平台。

MindSpore Serving提供以下功能：

- 支持客户端gRPC和RESTful接口。
- 支持组装模型的前处理和后处理。
- 支持batch功能，多实例请求会被拆分组合以满足模型`batch size`的需要。
- 提供客户端Python简易接口。
- 支持多模型组合，多模型组合和单模型场景使用相同的一套接口。
- 支持分布式模型推理功能。

## 安装

MindSpore Serving安装和配置可以参考[MindSpore Serving安装页面](https://www.mindspore.cn/serving/docs/zh-CN/r1.7/serving_install.html)。

## 快速入门

以一个简单的[Add网络示例](https://www.mindspore.cn/serving/docs/zh-CN/r1.7/serving_example.html)，演示MindSpore Serving如何使用。

## 文档

### 开发者教程

- [基于gRPC接口访问MindSpore Serving服务](https://www.mindspore.cn/serving/docs/zh-CN/r1.7/serving_grpc.html)
- [基于RESTful接口访问MindSpore Serving服务](https://www.mindspore.cn/serving/docs/zh-CN/r1.7/serving_restful.html)
- [配置模型提供服务](https://www.mindspore.cn/serving/docs/zh-CN/r1.7/serving_model.html)
- [配置多模型组合的服务](https://www.mindspore.cn/serving/docs/zh-CN/r1.7/serving_model.html#id9)
- [基于MindSpore Serving部署分布式推理服务](https://www.mindspore.cn/serving/docs/zh-CN/r1.7/serving_distributed_example.html)

有关安装指南、教程和API的更多详细信息，请参阅[用户文档](https://www.mindspore.cn/serving/docs/zh-CN/r1.7/server.html)。

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
