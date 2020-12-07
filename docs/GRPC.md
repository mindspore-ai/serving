# gRPC接口使用说明

## 概述
MindSpore Serving提供gRPC接口访问Serving服务。在Python环境下，我们提供[mindspore_serving.client](../mindspore_serving/client/python/client.py) 接口填写请求、解析回复。接下来我们详细说明`mindspore_serving.client`如何使用。

## 样例
在详细说明接口之前，我们先看几个样例。

### add样例
样例来源于[add example](../mindspore_serving/example/add/client.py)
```
from mindspore_serving.client import Client
import numpy as np


def run_add_common():
    """invoke Servable add method add_common"""
    client = Client("localhost", 5500, "add", "add_common")
    instances = []

    # instance 1
    x1 = np.asarray([[1, 1], [1, 1]]).astype(np.float32)
    x2 = np.asarray([[1, 1], [1, 1]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2})

    # instance 2
    x1 = np.asarray([[2, 2], [2, 2]]).astype(np.float32)
    x2 = np.asarray([[2, 2], [2, 2]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2})

    # instance 3
    x1 = np.asarray([[3, 3], [3, 3]]).astype(np.float32)
    x2 = np.asarray([[3, 3], [3, 3]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2})

    result = client.infer(instances)
    print(result)


if __name__ == '__main__':
    run_add_common()
```
按照[入门流程](../README_CN.md/#%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8) 导出模型，启动Serving服务器，并执行客户端代码。当运行正常后，将打印以下结果，为了展示方便，格式作了调整：
```
[{'y': array([[2., 2.], [2., 2.]], dtype=float32)},
 {'y': array([[4., 4.], [4., 4.]], dtype=float32)},
 {'y': array([[6., 6.], [6., 6.]], dtype=float32)}]
```

以下将对其中的细节进行说明。
1. 构造`Client`

    构造`Client`时，指示Serving的ip和端口号，并给定Servable名称和它提供的方法。这里的Servable可以是单个模型，也可以是多个模型的组合，一个Servable可以提供多种方法以提供不同的服务。

    上面的`add`样例， Serving运行在本地（`localhost`），指定的gRPC端口号为`5500`，运行了`add` Servable，`add` Servable提供了`add_common`方法。

2. 添加实例

   每次请求可包括一个或多个实例，每个实例之间相互独立，结果互不影响。

   比如：`add` Servable提供的`add_common`方法提供两个2x2 Tensor相加功能，即一个实例包含两个2x2 Tensor输入，一个2x2 Tensor输出。一次请求可包括一个、两个或者多个这样的实例，针对每个实例返回一个结果。上述`add`样例提供了三个实例，预期将返回三个实例的结果。
    ```
    Given Request:
    instance1:
    x1 = [[1, 1], [1, 1]]
    x2 = [[1, 1], [1, 1]]

    instance2:
    x1 = [[2, 2], [2, 2]]
    x2 = [[2, 2], [2, 2]]

    instance3:
    x1 = [[3, 3], [3, 3]]
    x2 = [[3, 3], [3, 3]]

    Expected Relpy:
    instance1:
    y = [[2., 2.], [2., 2.]] # instance1 x1 + x2

    instance2:
    y = [[4., 4.], [4., 4.]] # instance2 x1 + x2

    instance3:
    y = [[6., 6.], [6., 6.]] # instance3 x1 + x2
    ```

   `Client.infer`接口入参可为实例的list、tuple或者单个实例。每个实例的输入由dict表示，dict的key即为输入的名称，value为输入的值。
   
   value可以是以下格式的值：

    |  值类型   | 说明  |  举例  |
    |  ----  | ----  |  ---- |
    | numpy array  | 用以表示Tensor | np.ones((3,224), np.float32) |
    | numpy scalar | 用以表示Scalar | np.int8(5)  |
    | python bool int float | 用以表示Scalar, 当前int将作为int32, float将作为float32 | 32.0  |
    | python str | 用以表示字符串 | "this is a text"  |
    | python bytes | 用以表示二进制数据 | 图片数据  |
    
    上面的add样例，`add` Servable提供的`add_common`方法入参名为`x1`和`x2`，添加每个实例时指定每个输入的值。

3. 获取推理结果
    通过`Client.infer`填入一个或多个实例。
    返回可能有以下形式：
    
    所有实例推理正确：

    ```
    [{'y': array([[2., 2.], [2., 2.]], dtype=float32)},
     {'y': array([[4., 4.], [4., 4.]], dtype=float32)},
     {'y': array([[6., 6.], [6., 6.]], dtype=float32)}]
    ```

    针对所有实例共同的错误，返回一个包含`error`的dict。将例子中Client构造时填入的`add_common`改为`add_common2`，将返回结果：

    ```
    {'error', 'Request Servable(add) method(add_common2), method is not available'}
    ```

    部分实例推理错误，出错的推理实例将返回包含`error`的dict。将instance2一个输入的`dtype`改为`np.int32`，将返回结果：

    ```
    [{'y': array([[2., 2.], [2., 2.]], dtype=float32)},
     {'error': 'Given model input 1 data type kMSI_Int32 not match ...'},
     {'y': array([[6., 6.], [6., 6.]], dtype=float32)}]
    ```
    每个实例返回一个dict，key的值来自于Servable的方法定义，例如本例子中，`add` Servable提供的`add_common`方法输出仅有一个，为`y`。value为以下格式：

    |  Serving输出类型 | Client返回类型   | 说明  |  举例  |
    |  ----  | ----  |  ---- | ---- |
    | Tensor | numpy array | tensor array | np.ones((3,224), np.float32) |
    | Scalar: <br>int8, int16, int32, int64, <br>uint8, uint16, uint32, uint64, <br>bool, float16, float32, float64 | numpy scalar | Scalar格式的数据转为numpy scalar | np.int8(5)  |
    | String | python str | 字符串格式输出转为python str | "news_car"  |
    | Bytes | python bytes | 二进制格式输出转为python bytes | 图片数据  |


### lenet样例
样例来源于[lenet example](../mindspore_serving/example/lenet/client.py) 。通过lenet样例来说明二进制的输入，lenet输入为二进制，输出为Scalar。
```
import os
from mindspore_serving.client import Client


def run_predict():
    client = Client("localhost", 5500, "lenet", "predict")
    instances = []
    for path, _, file_list in os.walk("./test_image/"):
        for file_name in file_list:
            image_file = os.path.join(path, file_name)
            print(image_file)
            with open(image_file, "rb") as fp:
                instances.append({"image": fp.read()})
    result = client.infer(instances)
    if "error" in result:
        print("error happen:", result["error"])
        return
    print(result)


if __name__ == '__main__':
    run_predict()
```
上面lenet例子中，输入的每个实例的唯一的输入`image`为文件二进制方式读取的bytes。
正常结束执行后，可能将会有以下打印：
```
[{'result': 4}, {'result': 4}, {'result': 4}, {'result': 4}]
```

