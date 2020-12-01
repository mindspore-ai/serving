# MindSpore Serving

[View English](./README.md)

- [概述](#概述)
- [安装部署](#安装部署)
    - [安装MindSpore Serving](#安装MindSpore-Serving)
    - [配置环境变量](#配置环境变量)
    - [部署MindSpore Serving](#部署MindSpore-Serving)
- [快速入门](#快速入门)
    - [导出模型](#导出模型)
    - [部署Serving推理服务](#部署serving推理服务)
    - [执行推理](#执行推理)
- [文档](#文档)
    - [开发者教程](#开发者教程)
- [社区](#社区)
    - [治理](#治理)
    - [交流](#交流)
- [贡献](#贡献)
- [版本说明](#版本说明)
- [许可证](#许可证)

## 概述

MindSpore Serving是一个轻量级、高性能的服务模块，旨在帮助MindSpore开发者在生产环境中高效部署在线推理服务。当用户使用MindSpore完成模型训练后，导出MindSpore模型，即可使用MindSpore Serving创建该模型的推理服务。  

**MindSpore Serving架构：**   
当前MindSpore Serving服务节点分为client，master和worker。client为用户节点，下发推理服务命令。执行机worker部署了模型服务。当前仅支持Ascend 310和Ascend 910，后续会逐步支持GPU和CPU场景。master节点用来管理所有的执行机worker及其部署的模型信息，并进行任务管理与分发。master和worker可以部署在一个进程中，也可以部署在不同进程中。  
<img src="docs/image/architecture.png" alt="MindSpore Architecture" width="600"/>   

**MindSpore Serving提供以下功能：**
- 支持客户端gRPC和RESTful接口
- 支持组装模型的前处理和后处理
- 支持batch功能
- 提供客户端python简易接口

## 安装部署
MindSpore Serving依赖MindSpore训练推理框架，安装完[MindSpore](https://gitee.com/mindspore/mindspore#%E5%AE%89%E8%A3%85) ，再安装MindSpore Serving。

### 安装MindSpore Serving
使用pip命令安装，安装方式如下：

**1、请从MindSpore Serving下载页面下载并安装whl包。**
```python
pip install mindspore_serving-1.0.0-cp37-cp37m-linux_x86_64.whl
```
**2、源码安装。**  
下载[源码](https://gitee.com/mindspore/serving)。

方式一，使用已安装或编译的MindSpore包：
```shell
sh build.sh -p $MINDSPORE_LIB_PATH
```
$MINDSPORE_LIB_PATH为mindspore软件包的安装路径下的lib路径，例：softwarepath/mindspore/lib，该路径包含mindspore运行依赖的库文件。

方式二，编译Serving时编译配套的MindSpore包，需要配置MindSpore编译时的[环境变量](https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend_install_source.md#配置环境变量) ：
```shell
# ascend 310
sh build.sh -eacl
# ascend 910
sh build.sh -ed
```

编译完后，在output/目录下找到安装包进行安装：
```python
pip install mindspore_serving-0.1.0-cp37-cp37m-linux_x86_64.whl
```
### 配置环境变量
Asend 910环境上安装mindspore，需要完成[环境变量配置](https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend_install_pip.md#%E9%85%8D%E7%BD%AE%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F)。   
运行MindSpore Serving，还需要增加额外mindspore软件包的安装路径下的lib路径到LD_LIBRARY_PATH。
```shell
export LD_LIBRARY_PATH=$MINDSPORE_LIB_PATH:${LD_LIBRARY_PATH}
```

### 部署MindSpore Serving
MindSpore Serving提供两种部署方式，用户可根据需要进行选择部署。

**轻量级部署：**  
服务端调用python接口直接启动推理进程（master和worker共进程），客户端直接连接推理服务后下发推理任务。  
启动服务：
```python
import os
from mindspore_serving import master
from mindspore_serving import worker
servable_dir = os.path.abspath(".")
worker.start_servable_in_master(servable_dir, "xxx", device_id=0) 
master.start_grpc_server("127.0.0.1", 5500)
```

**集群部署：**  
服务端由master进程和worker进程组成，master用来管理集群内所有的worker节点，并进行推理任务的分发。  
启动worker：
```python
import os
from mindspore_serving import worker
servable_dir = os.path.abspath(".")
worker.start_servable(servable_dir, "lenet", device_id=0, 
                       master_ip="127.0.0.1", master_port=5500,
                       host_ip="127.0.0.1", host_port=5600)
```
启动master：
```python
from mindspore_serving import master
master.start_grpc_server("127.0.0.1", 5500)
```
完成服务端部署后，即可启用客户端程序执行推理操作。

## 快速入门
以一个简单的Add网络为例，演示MindSpore Serving如何使用。

### 导出模型
使用[add_model.py](https://gitee.com/mindspore/serving/blob/master/mindspore_serving/example/add/export_model/add_model.py)，构造一个只有Add算子的网络，并导出MindSpore推理部署模型。

```python
python add_model.py
```
执行脚本，生成`tensor_add.mindir`文件，该模型的输入为两个shape为[2,2]的二维Tensor，输出结果是两个输入Tensor之和。

### 部署Serving推理服务
执行以下[python程序](https://gitee.com/mindspore/serving/blob/master/mindspore_serving/example/add/master_with_worker.py)，启动服务：
```bash
import os
from mindspore_serving import master
from mindspore_serving import worker
def start():
    servable_dir = os.path.abspath(".")
    worker.start_servable_in_master(servable_dir, "add", device_id=0)
    master.start_grpc_server("127.0.0.1", 5500)
```
启动过程需要使用servable_dir路径下的模型文件和配置文件，文件目录结果如下图所示：   
<pre><font color="#268BD2"><b>add/</b></font>
├── <font color="#268BD2"><b>1</b></font>
│   └── tensor_add.mindir
└── servable_config.py
</pre>   
其中，模型文件为上一步网络生成的，即`tensor_add.mindir`文件。配置文件为[servable_config.py](https://gitee.com/mindspore/serving/blob/master/mindspore_serving/example/add/add/servable_config.py)，其定义了模型的处理函数，包含前处理和后处理过程。
当服务端打印日志`Serving gRPC start success, listening on 0.0.0.0:5500`时，表示Serving服务已加载推理模型完毕。

### 执行推理
使用[client.py](https://gitee.com/mindspore/serving/blob/master/mindspore_serving/example/add/client.py)，启动Python客户端。
```bash
python client.py
```

显示如下返回值说明Serving服务已正确执行Add网络的推理。
```bash
[{'y': array([[2. , 2.],
        [2.,  2.]], dtype=float32)}]
[{'y': array([[2. , 2.],
        [2.,  2.]], dtype=float32)}]
```

## 文档

### 开发者教程
- [如何使用python接口开发客户端？](docs/GRPC.md)
- [如何启动Restful服务进行推理？](docs/RESTful.md)
- [如何实现模型前处理和后处理？](docs/MODEL.md)

有关安装指南、教程和API的更多详细信息，请参阅[用户文档](https://gitee.com/mindspore/serving/docs)。

## 社区

### 治理

查看MindSpore如何进行[开放治理](https://gitee.com/mindspore/community/blob/master/governance.md)。

### 交流

- [MindSpore Slack](https://join.slack.com/t/mindspore/shared_invite/zt-dgk65rli-3ex4xvS4wHX7UDmsQmfu8w) 开发者交流平台。
- `#mindspore`IRC频道（仅用于会议记录）
- 视频会议：待定
- 邮件列表：<https://mailweb.mindspore.cn/postorius/lists>

## 贡献

欢迎参与贡献。

## 版本说明

版本说明请参阅[RELEASE](RELEASE.md)。

## 许可证

[Apache License 2.0](LICENSE)
