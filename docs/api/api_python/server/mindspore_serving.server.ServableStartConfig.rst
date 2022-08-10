
.. py:class:: mindspore_serving.server.ServableStartConfig(servable_directory, servable_name, device_ids=None, version_number=0, device_type=None, num_parallel_workers=0, dec_key=None, dec_mode='AES-GCM')

    启动一个服务的配置。详情请查看
    `基于MindSpore Serving部署推理服务 <https://www.mindspore.cn/serving/docs/zh-CN/master/serving_example.html>`_ 和
    `通过配置模型提供Servable <https://www.mindspore.cn/serving/docs/zh-CN/master/serving_model.html>`_ 。

    参数：
        - **servable_directory** (str) - 服务所在的目录。预期有一个名为 `servable_name` 的目录。
        - **servable_name** (str) - 服务名称。
        - **device_ids** (Union[int, list[int], tuple[int]], optional) - 模型部署和运行的设备列表，列表中的每个会设备将部署和运行一个服务副本。当设备类型为Nvidia GPU、Ascend 310/310P/910时使用。默认值：None。
        - **version_number** (int, optional) - 要加载的服务的版本号。版本号应为正整数，从1开始，0表示加载最新版本。默认值：0。
        - **device_type** (str, optional) - 模型部署的目标设备类型，目前支持"Ascend"、"GPU"、"CPU"和None。默认值：None。

          - "Ascend"：目标设备为Ascend 310/310P/910等。
          - "GPU"：目标设备为Nvidia GPU。
          - "CPU"：目标设备为CPU。
          - None：系统根据实际的后端设备和MindSpor推理包决定目标设备，推荐使用默认值None。

        - **num_parallel_workers** (int, optional) - 处理Python任务的进程数，用于提高预处理、后处理等Python任务的处理能力。值小于 `device_ids` 的长度时，处理Python任务的进程数为 `device_ids` 的长度。值的范围为[0,64]。默认值：0。
        - **dec_key** (bytes, optional) - 用于解密的字节类型密钥。有效长度为16、24或32。默认值：None。
        - **dec_mode** (str, optional) - 指定解密模式，设置 `dec_key` 时生效。值可为： `'AES-GCM'` 或 `'AES-CBC'` 。默认值： `'AES-GCM'` 。

    异常：
        - **RuntimeError** - 参数的类型或值无效。

