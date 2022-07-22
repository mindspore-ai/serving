
.. py:class:: mindspore_serving.server.register.GPUDeviceInfo(**kwargs)

    用于GPU设备配置。

    参数：
        - **precision_mode** (str, optional) - 推理精度选项，值可以是 `"origin"` 或 `"fp16"` ， `"origin"` 表示以模型中指定精度进行推理， `"fp16"` 表示以FP16精度进行推理。默认值： `"origin"` 。

    异常：
        - **RuntimeError** - 选项无效，或值类型不是字符串。
