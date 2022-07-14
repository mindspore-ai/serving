# MindSpore Serving Release Notes

[查看中文](./RELEASE_CN.md)

## MindSpore Serving 1.8.0 Release Notes

### Major Features and Improvements

- [STABLE] When deploying a large-scale model with parallel pipelines, Serving supports parallel pipeline processing of multiple inference instances.

### Contributors

Thanks goes to these wonderful people:

qinzheng, xuyongfei, zhangyinxia, zhoufeng.

Contributions of any kind are welcome!

## MindSpore Serving 1.7.0 Release Notes

### Major Features and Improvements

- [DEMO] Ascend 310P can be used as the inference device, for more detail see [MindSpore Serving backend](https://www.mindspore.cn/serving/docs/en/r1.8/serving_install.html#installation).
- [DEMO] Support models of MindIR format when MindSpore Lite is used as the MindSpore Serving inference backend, for more detail see [MindSpore Serving backend](https://www.mindspore.cn/serving/docs/en/r1.8/serving_install.html#installation).

#### Deprecations

##### Python API

- `AclOptions` and `GpuOptions` are removed from version 1.7.0, and use `AscendDeviceInfo` and `GPUDeviceInfo` instead.
- `register.declare_sevable` and `register.call_servable` are removed from version 1.7.0, and use `register.declare_model` and `register.add_stage` instead.
- `register.call_preprocess`, `register.call_preprocess_pipeline`, `register.call_postprocess` and `register.call_postprocess_pipeline` are removed from version 1.7.0, and use `register.add_stage` instead.

### Contributors

Thanks goes to these wonderful people:

qinzheng, xuyongfei, zhangyinxia, zhoufeng.

Contributions of any kind are welcome!

## MindSpore Serving 1.6.0 Release Notes

### Major Features and Improvements

- [STABLE] We can use existing interfaces(`decalre_model` and `add_stage`) that define single-model services to define
  multi-model composite services.
- [STABLE] When the number of occupied devices is fixed, additional worker processes(using parameter
  `num_parallel_workers`) are supported to accelerate Python functions such as preprocessing and postprocessing,
  improving device utilization.
- [STABLE] The interface `Model.call` is a stable feature, and can be used to define complex model invocation processes
  in the Serving server, such as looping and conditional branching.
- [STABLE] The new interfaces `Context`, `CPUDeviceInfo`, `GPUDeviceInfo`, `AscendDeviceInfo` are provided to set
  user-defined device information. The original interfaces `GpuOptions` and `AclOptions` are deprecated.
- [BETA] We support MindSpore Lite as the MindSpore Serving inference backend, for more detail see
  [MindSpore Serving backend](https://www.mindspore.cn/serving/docs/en/r1.8/serving_install.html#installation).

### API Change

#### New features

##### Python API

###### Multi-model composite services

We can use existing interfaces(`decalre_model` and `add_stage`) that define single-model services to define
multi-model composite services. For more detail, see [Services Composed of Multiple Models](https://www.mindspore.cn/serving/docs/en/r1.8/serving_model.html#services-composed-of-multiple-models).

```python
from mindspore_serving.server import register

add_model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
sub_model = register.declare_model(model_file="tensor_sub.mindir", model_format="MindIR", with_batch_dim=False)

@register.register_method(output_names=["y"])
def add_sub_only_model(x1, x2, x3):  # x1+x2-x3
    y = register.add_stage(add_model, x1, x2, outputs_count=1)
    y = register.add_stage(sub_model, y, x3, outputs_count=1)
    return y
```

###### Additional worker processes are supported to accelerate Python functions(preprocessing and postprocessing)

Parameter `num_parallel_workers` in class `ServableStartConfig`  is a stable feature. It's can be used to configure the
total number of workers. The number of workers occupying devices is determined by the length of parameter `device_ids`.
Additional worker processes use worker processes that occupy devices for model inference.  For more detail, see
[Multi-process Concurrency](https://www.mindspore.cn/serving/docs/en/r1.8/serving_model.html#multi-process-concurrency).

```python
class ServableStartConfig:
    def __init__(self, servable_directory, servable_name, device_ids, version_number=0, device_type=None,
                 num_parallel_workers=0, dec_key=None, dec_mode='AES-GCM')
```

Start the serving server that contains the `resnet50` servable. The `resnet50` servable has four worker
processes(`num_parallel_workers`), one of which occupies the device(`device_ids`).

```python
import os
import sys
from mindspore_serving import server

def start():
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    # Total 4 worker, one worker occupy device 0, the model inference tasks of other workers are forwarded to the worker
    # that occupies the device.
    config = server.ServableStartConfig(servable_directory=servable_dir,
                                        servable_name="resnet50", device_ids=0,
                                        num_parallel_workers=4)
    server.start_servables(config)

    server.start_grpc_server("127.0.0.1:5500")
    server.start_restful_server("127.0.0.1:1500")

if __name__ == "__main__":
    start()
```

###### Model.call interface can be used to define complex model invocation processes

The interface `Model.call` is a stable feature, and can be used to define complex model invocation processes
in the Serving server, such as looping and conditional branching.

```python
from mindspore_serving.server import register

import numpy as np
from .tokenizer import create_tokenizer, padding, END_TOKEN

bert_model = register.declare_model(model_file="bert_poetry.mindir", model_format="MindIR")

def calc_new_token(probas):
  ...
  return new_token_id

tokenizer = create_tokenizer()

def generate_sentence(input_sentence):
  input_token_ids = tokenizer.encode(input_sentence)
  target_ids = []
  MAX_LEN = 64
  while len(input_token_ids) + len(target_ids) < MAX_LEN:
    input_ids = padding(np.array(input_token_ids + target_ids), length=128)
    pad_mask = (input_ids != 0).astype(np.float32)
    probas = bert_model.call(input_ids, pad_mask)  # call bert model to generate token id of new word
    new_token_id = calc_new_token(probas[len(input_token_ids)])
    target_ids.append(new_token_id)
    if new_token_id == END_TOKEN:
      break
  output_sentence = tokenizer.decode(input_token_ids + target_ids)
  return output_sentence


@register.register_method(output_names=["output_sentence"])
def predict(input_sentence):
  output_sentence = register.add_stage(generate_sentence, input_sentence, outputs_count=1)
  return output_sentence
```

#### Deprecations

##### Python API

- The parameter `options` in `register.declare_model` is deprecated from version 1.6.0 and will be removed in a future version, use parameter `context` instead.
- `AclOptions` and `GpuOptions` are deprecated from version 1.6.0 and will be removed in a future version, use `AscendDeviceInfo` and `GPUDeviceInfo` instead.

### Contributors

Thanks goes to these wonderful people:

qinzheng, xuyongfei, zhangyinxia, zhoufeng.

Contributions of any kind are welcome!

## MindSpore Serving 1.5.0 Release Notes

### Major Features and Improvements

- [STABLE] To support multi-model orchestration (to be released in version 1.6), a set of APIs (`decalre_model`
  and `add_stage`) is added. The new APIs will be used in single-model and multi-model scenarios. The old
  APIs(`register.declare_servable`,`call_servable`,`call_preprocess`,`call_postprocess`) used in single-model scenarios
  are deprecated.
- [BETA] When the number of occupied devices is fixed, additional worker processes are supported to accelerate Python
  functions such as preprocessing and postprocessing, improving device utilization.
- [BETA]`Model.call` interface is added to support invoking models in Python functions.

### API Change

#### API Incompatible Change

##### Python API

###### New set of APIs for single-model and multi-model scenarios

To support multiple models(will be officially released in version 1.6), a set of APIs (`decalre_model` and `add_stage`)
is added. The single-model and multi-model scenarios will use the same set of APIs.

New APIs are recommended in single-model scenarios. Old APIs (`declare_servable`,`call_servable`,`call_preprocess`,
`call_postprocess`) are deprecated.

<table>
<tr>
<td style="text-align:center"> 1.4 </td> <td style="text-align:center"> 1.5 </td>
</tr>
<tr>
<td>

```python
from mindspore_serving.server import register

register.declare_servable(servable_file="resnet.mindir",
                          model_format="MindIR")

def resnet_preprocess(image):
    ....

def resnet_postprocess(scores):
    ....

@register.register_method(output_names=["label"])
def predict(image):
    x = register.call_preprocess(resnet_preprocess, image)
    x = register.call_servable(x)
    x = register.call_postprocess(resnet_postprocess, x)
    return x
```

</td>
<td>

```python
from mindspore_serving.server import register

resnet_model = register.declare_model(model_file="resnet.mindir",
                                      model_format="MindIR")

def resnet_preprocess(image):
    ....

def resnet_postprocess(scores):
    ....

@register.register_method(output_names=["label"])
def predict(image):
    x = register.add_stage(resnet_preprocess, image, outputs_count=1)
    x = register.add_stage(resnet_model, x, outputs_count=1)
    x = register.add_stage(resnet_postprocess, x, outputs_count=1)
    return x
```

</td>
</tr>
</table>

#### New features

##### Python API

###### Additional worker processes are supported to accelerate Python functions(preprocessing and postprocessing)

Parameter `num_parallel_workers` is added to class `ServableStartConfig` to configure the total number of workers. The
number of workers occupying devices is determined by the length of parameter `device_ids`. Additional worker processes
use worker processes that occupy devices for model inference.

```python
class ServableStartConfig:
    def __init__(self, servable_directory, servable_name, device_ids, version_number=0, device_type=None,
                 num_parallel_workers=0, dec_key=None, dec_mode='AES-GCM')
```

Start the serving server that contains the `resnet50` servable. The `resnet50` servable has four worker
processes(`num_parallel_workers`), one of which occupies the device(`device_ids`).

```python
import os
import sys
from mindspore_serving import server

def start():
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    # Total 4 worker, one worker occupy device 0, the model inference tasks of other workers are forwarded to the worker
    # that occupies the device.
    config = server.ServableStartConfig(servable_directory=servable_dir,
                                        servable_name="resnet50", device_ids=0,
                                        num_parallel_workers=4)
    server.start_servables(config)

    server.start_grpc_server("127.0.0.1:5500")
    server.start_restful_server("127.0.0.1:1500")

if __name__ == "__main__":
    start()
```

###### Model.call interface is added to support invoking models in Python functions

```python
from mindspore_serving.server import register

add_model = register.declare_model(model_file="tensor_add.mindir",
                                   model_format="MindIR")

def add_func(x1, x2, x3, x4):
    instances = []
    instances.append((x1, x2))
    instances.append((x3, x4))
    output_instances = add_model.call(instances)  # for multi instances
    y1 = output_instances[0][0]  # instance 0 output 0
    y2 = output_instances[1][0]  # instance 1 output 0
    y = add_model.call(y1, y2)  # for single instance
    return y

@register.register_method(output_names=["y"])
def predict(x1, x2, x3, x4):
    y = register.add_stage(add_func, x1, x2, x3, x4, outputs_count=1)
    return y
```

#### Deprecations

##### Python API

- `register.declare_servable`,`call_servable`,`call_preprocess`,`call_postprocess`,`call_preprocess_pipeline`
  and`call_postprocess_pipeline` are now deprecated in favor of`register.declare_model` and`add_stage`, as shown above.
  Deprecated interfaces will be deleted in the future.
- Beta interfaces`PipelineServable` and`register_pipeline` introduced in version 1.3 will be deleted and replaced
  with`Model.call`.

### Contributors

Thanks goes to these wonderful people:

chenweifeng, qinzheng, xuyongfei, zhangyinxia, zhoufeng.

Contributions of any kind are welcome!

## MindSpore Serving 1.3.0 Release Notes

### Major Features and Improvements

- [STABLE] Enhances and simplifies the deployment and startup of single-chip models. Multiple models can be loaded by a
  single script. Each model can have multiple copies on multiple chips. Requests can be split and distributed to these
  copies for concurrent execution.
- [STABLE] The `master`+`worker` interface of the Serving server is changed to the `server` interface.
- [STABLE] The client and server support Unix Domain Socket-based gRPC communication.
- [STABLE] gRPC and RESTful interfaces support TLS/SSL security authentication.
- [STABLE] The MindIR encryption model is supported.
- [BETA] Incremental inference models consisting of multiple static graphs are supported, including single-card models
  and distributed models.

### API Change

#### API Incompatible Change

##### Python API

###### Enhances and simplifies the deployment and startup of single-chip models

Multiple models can be loaded by a single script. Each model can have multiple copies on multiple chips. Requests can be
split and distributed to these copies for concurrent execution.

Interface `worker.start_servable_in_master` that can start only a single servables is changed to
interface `server.start_servables` that can start multiple servables, and each servable can correspond to multiple
copies. In addition, related interface `server.ServableStartConfig` is added.

<table>
<tr>
<td style="text-align:center"> 1.2.x </td> <td style="text-align:center"> 1.3.0 </td>
</tr>
<tr>
<td>

```python
import os
import sys
from mindspore_serving import master
from mindspore_serving import worker

def start():
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    # deploy model add on device 0
    worker.start_servable_in_master(servable_dir, "add", device_id=0)

    master.start_grpc_server("127.0.0.1", 5500)
    master.start_restful_server("127.0.0.1", 1500)

if __name__ == "__main__":
    start()
```

</td>
<td>

```python
import os
import sys
from mindspore_serving import server

def start():
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    # deploy model add on devices 0 and 1
    add_config = server.ServableStartConfig(servable_directory=servable_dir,
                                            servable_name="add",
                                            device_ids=(0, 1))
    # deploy model resnet50 on devices 2 and 3
    resnet50_config = server.ServableStartConfig(servable_directory=servable_dir,
                                                 servable_name="resnet50 ",
                                                 device_ids=(2, 3))
    server.start_servables(servable_configs=(add_config, resnet50_config))

    server.start_grpc_server(address="127.0.0.1:5500")
    server.start_restful_server(address="127.0.0.1:1500")

if __name__ == "__main__":
    start()
```

</td>
</tr>
</table>

###### `mindspore_serving.worker.register` is updated to `mindspore_serving.server.register`

<table>
<tr>
<td style="text-align:center"> 1.2.x </td> <td style="text-align:center"> 1.3.0 </td>
</tr>
<tr>
<td>

```python
from mindspore_serving.worker import register
```

</td>
<td>

```python
from mindspore_serving.server import register
```

</td>
</tr>
</table>

###### The gRPC and RESTful startup interfaces are updated. The namespace is changed from master to server, and the input parameters `ip` and `port` are changed to `address` only

<table>
<tr>
<td style="text-align:center"> 1.2.x </td> <td style="text-align:center"> 1.3.0 </td>
</tr>
<tr>
<td>

```python
from mindspore_serving import master

master.start_grpc_server("127.0.0.1", 5500)
master.start_restful_server("127.0.0.1", 1500)
master.stop()
```

</td>
<td>

```python
from mindspore_serving import server

server.start_grpc_server("127.0.0.1:5500")
server.start_restful_server("127.0.0.1:1500")
server.stop()
```

</td>
</tr>
</table>

###### The name of the distributed interface function is simplified, and the namespace is changed from `worker` to `server`

In `servable_config.py` of distributed model:

<table>
<tr>
<td style="text-align:center"> 1.2.x </td> <td style="text-align:center"> 1.3.0 </td>
</tr>
<tr>
<td>

```python
from mindspore_serving.worker import distributed

distributed.declare_distributed_servable(
    rank_size=8, stage_size=1, with_batch_dim=False)
```

</td>
<td>

```python
from mindspore_serving.server import distributed

distributed.declare_servable(
    rank_size=8, stage_size=1, with_batch_dim=False)
```

</td>
</tr>
</table>

In startup script of distributed model:

<table>
<tr>
<td style="text-align:center"> 1.2.x </td> <td style="text-align:center"> 1.3.0 </td>
</tr>
<tr>
<td>

```python
import os
import sys
from mindspore_serving import master
from mindspore_serving.worker import distributed

def start():
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    distributed.start_distributed_servable_in_master(
        servable_dir, "matmul",
        rank_table_json_file="rank_table_8pcs.json",
        version_number=1,
        worker_ip="127.0.0.1", worker_port=6200)

    master.start_grpc_server("127.0.0.1", 5500)
    master.start_restful_server("127.0.0.1", 1500)

if __name__ == "__main__":
    start()
```

</td>
<td>

```python
import os
import sys
from mindspore_serving import server
from mindspore_serving.server import distributed

def start():
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    distributed.start_servable(
        servable_dir, "matmul",
        rank_table_json_file="rank_table_8pcs.json",
        version_number=1,
        distributed_address="127.0.0.1:6200")

    server.start_grpc_server("127.0.0.1:5500")
    server.start_restful_server("127.0.0.1:1500")

if __name__ == "__main__":
    start()
```

</td>
</tr>
</table>

In agent startup script of distributed model:

<table>
<tr>
<td style="text-align:center"> 1.2.x </td> <td style="text-align:center"> 1.3.0 </td>
</tr>
<tr>
<td>

```python
from mindspore_serving.worker import distributed

def start_agents():
    """Start all the worker agents in current machine"""
    model_files = []
    group_configs = []
    for i in range(8):
        model_files.append(f"model/device{i}/matmul.mindir")
        group_configs.append(f"model/device{i}/group_config.pb")

    distributed.startup_worker_agents(
        worker_ip="127.0.0.1", worker_port=6200,
        model_files=model_files,
        group_config_files=group_configs)

if __name__ == '__main__':
    start_agents()
```

</td>
<td>

```python
from mindspore_serving.server import distributed

def start_agents():
    """Start all the agents in current machine"""
    model_files = []
    group_configs = []
    for i in range(8):
        model_files.append(f"model/device{i}/matmul.mindir")
        group_configs.append(f"model/device{i}/group_config.pb")

    distributed.startup_agents(
        distributed_address="127.0.0.1:6200",
        model_files=model_files,
        group_config_files=group_configs)

if __name__ == '__main__':
    start_agents()
```

</td>
</tr>
</table>

###### The input parameters `ip`+`port` of the gRPC client are changed to `address`

In addition to the {ip}:{port} address format, the Unix Domain Socket in the unix:{unix_domain_file_path} format is
supported.

<table>
<tr>
<td style="text-align:center"> 1.2.x </td> <td style="text-align:center"> 1.3.0 </td>
</tr>
<tr>
<td>

```python
import numpy as np
from mindspore_serving.client import Client

def run_add_cast():
    """invoke servable add method add_cast"""
    client = Client("localhost", 5500, "add", "add_cast")
    instances = []
    x1 = np.ones((2, 2), np.int32)
    x2 = np.ones((2, 2), np.int32)
    instances.append({"x1": x1, "x2": x2})
    result = client.infer(instances)
    print(result)

if __name__ == '__main__':
    run_add_cast()
```

</td>
<td>

```python
import numpy as np
from mindspore_serving.client import Client

def run_add_cast():
    """invoke servable add method add_cast"""
    client = Client("127.0.0.1:5500", "add", "add_cast")
    instances = []
    x1 = np.ones((2, 2), np.int32)
    x2 = np.ones((2, 2), np.int32)
    instances.append({"x1": x1, "x2": x2})
    result = client.infer(instances)
    print(result)

if __name__ == '__main__':
    run_add_cast()
```

</td>
</tr>
</table>

#### New features

##### Python API

###### Support Unix Domain Socket

The Serving server:

```python
import os
import sys
from mindspore_serving import server


def start():
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    servable_config = server.ServableStartConfig(servable_directory=servable_dir, servable_name="resnet50",
                                                 device_ids=(0, 1))
    server.start_servables(servable_configs=servable_config)
    server.start_grpc_server(address="unix:/tmp/serving_resnet50_test_temp_file")


if __name__ == "__main__":
    start()
```

The Serving client:

```python
import os
from mindspore_serving.client import Client


def run_classify_top1():
    client = Client("unix:/tmp/serving_resnet50_test_temp_file", "resnet50", "classify_top1")
    instances = []
    for path, _, file_list in os.walk("./test_image/"):
        for file_name in file_list:
            image_file = os.path.join(path, file_name)
            print(image_file)
            with open(image_file, "rb") as fp:
                instances.append({"image": fp.read()})
    result = client.infer(instances)
    print(result)


if __name__ == '__main__':
    run_classify_top1()
```

###### Support SSL/TLS

The Serving server:

```python
import os
import sys
from mindspore_serving import server


def start():
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    servable_config = server.ServableStartConfig(servable_directory=servable_dir, servable_name="add",
                                                 device_ids=(0, 1))
    server.start_servables(servable_configs=servable_config)
    ssl_config = server.SSLConfig(certificate="server.crt", private_key="server.key", custom_ca=None,
                                  verify_client=False)
    server.start_grpc_server(address="127.0.0.1:5500", ssl_config=ssl_config)
    server.start_restful_server(address="127.0.0.1:1500", ssl_config=ssl_config)


if __name__ == "__main__":
    start()
```

The gRPC Serving client:

```python
from mindspore_serving.client import Client
from mindspore_serving.client import SSLConfig
import numpy as np


def run_add_common():
    """invoke Servable add method add_common"""
    ssl_config = SSLConfig(custom_ca="ca.crt")
    client = Client("localhost:5500", "add", "add_common", ssl_config=ssl_config)
    instances = []

    # instance 1
    x1 = np.asarray([[1, 1], [1, 1]]).astype(np.float32)
    x2 = np.asarray([[1, 1], [1, 1]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2})

    result = client.infer(instances)
    print(result)


if __name__ == '__main__':
    run_add_common()
```

The RESTful client

```shell
>>> curl -X POST -d '{"instances":{"x1":[[1.0, 1.0], [1.0, 1.0]], "x2":[[1.0, 1.0], [1.0, 1.0]]}}' --insecure https://127.0.0.1:1500/model/add:add_common
{"instances":[{"y":[[2.0,2.0],[2.0,2.0]]}]}
```

###### Support encryption MindIR model

```python
# export model
import mindspore as ms

# define add network
# export encryption model
ms.export(add, ms.Tensor(x), ms.Tensor(y), file_name='tensor_add_enc', file_format='MINDIR',
          enc_key="asdfasdfasdfasgwegw12310".encode(), enc_mode='AES-GCM')
```

```python
# start Serving server
import os
import sys
from mindspore_serving import server


def start():
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    servable_config = server.ServableStartConfig(servable_directory=servable_dir, servable_name="add",
                                                 device_ids=(0, 1),
                                                 dec_key='asdfasdfasdfasgwegw12310'.encode(), dec_mode='AES-CBC')
    server.start_servables(servable_configs=servable_config)

    server.start_grpc_server(address="127.0.0.1:5500")
    server.start_restful_server(address="127.0.0.1:1500")


if __name__ == "__main__":
    start()
```

###### [BETA] Support incremental inference models consisting of multiple static graphs

A Incremental inference models can include a full input graph and an incremental input graph, and the Serving
orchestrates the two static graphs using a user-defined Python script. For more details, please refer
to [Serving pangu alpha](https://gitee.com/mindspore/models/tree/master/official/nlp/pangu_alpha/serving_increment)
.

#### Deprecations

##### Python API

- `mindspore_serving.master` and `mindspore_serving.worker` are now deprecated in favor of `mindspore_serving.server`,
  as shown above. Deprecated interfaces will be deleted in the next iteration.

- The following interfaces are directly deleted. That is, workers of one serving server can no longer be deployed on
  othe machines. Users are no longer aware of workers at the interface layer.

```python
mindspore_serving.worker.start_servable
mindspore_serving.worker.distributed.start_distributed_servable
mindspore_serving.master.start_master_server
```

### Contributors

Thanks goes to these wonderful people:

chenweifeng, qinzheng, xuyongfei, zhangyinxia, zhoufeng.

Contributions of any kind are welcome!

## MindSpore Serving 1.2.0 Release Notes

### Major Features and Improvements

- [STABLE] Support distributed inference, it needs to cooperate with distributed training to export distributed models
  for super-large-scale neural network parameters(Ascend 910).
- [STABLE] Support GPU platform, Serving worker nodes can be deployer on Nvidia GPU, Ascend 310 and Ascend 910.
- This release is based on MindSpore version 1.2.0
- Support Python 3.8 and 3.9.

### API Change

#### API Incompatible Change

##### Python API

Support deployment of distributed model, refer
to [distributed inference tutorial](https://www.mindspore.cn/serving/docs/en/r1.8/serving_distributed_example.html)
for related API.

#### Deprecations

##### Python API

### Bug Fixes

### Contributors

Thanks goes to these wonderful people:

chenweifeng, qinzheng, xujincai, xuyongfei, zhangyinxia, zhoufeng.

Contributions of any kind are welcome!

## MindSpore Serving 1.1.1 Release Notes

### Major Features and Improvements

- Adapts new C++ inference interface for MindSpore version 1.1.1.

### Bug fixes

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
- [STABLE] Support Model batching，Multiple instances will be split and combined to meet the batch size requirements of
  the model.
- This release is based on MindSpore version 1.1.0

### Bug Fixes

### Contributors
