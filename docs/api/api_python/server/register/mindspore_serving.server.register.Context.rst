
.. py:class:: mindspore_serving.server.register.Context(**kwargs)

    Context用于自定义设备配置，如果不指定Context，MindSpore Serving将使用默认设备配置。当使用推理后端为MindSpore Lite，且目标设备为Ascend或Nvidia GPU时，模型部分算子可能运行在CPU设备上，将额外配置 `CPUDeviceInfo` 。

    参数：
        - **thread_num** (int, 可选) - 设置运行时的CPU线程数量，该选项仅当推理后端为MindSpore Lite有效。
        - **thread_affinity_core_list** (tuple[int], list[int], 可选) - 设置运行时的CPU绑核列表，该选项仅当推理后端为MindSpore Lite有效。
        - **enable_parallel** (bool, 可选) - 设置运行时是否支持并行，该选项仅当推理后端为MindSpore Lite有效。

    异常：
        - **RuntimeError** - 输入参数的类型或值无效。

    .. py:method:: append_device_info(device_info)

       用于添加一个用户自定义的设备配置。

       参数：
           - **device_info** (Union[CPUDeviceInfo, GPUDeviceInfo, AscendDeviceInfo]) - 用户自定义设备配置，用户不指定设备配置时将使用默认值。可以为每个可能的设备自定义设备配置，系统根据实际的后端设备和推理包选择所需的设备信息。

       异常：
           - **RuntimeError** - 输入参数的类型或值无效。
