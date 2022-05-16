
.. py:function:: mindspore_serving.server.register.declare_model(model_file, model_format, with_batch_dim=True, options=None, without_batch_dim_inputs=None, context=None, config_file=None)

    在服务的servable_config.py配置文件中使用，用于声明一个模型。

    .. note:: 该接口应在import servable_config.py时生效。因此，建议在servable_config.py中全局使用此接口。

    .. warning:: 参数 `options` 从1.6.0版本中已弃用，并将在未来版本中删除，请改用参数 `context` 。

    **参数：**

    - **model_file** (Union[str, list[str]]) - 模型文件名。
    - **model_format** (str) - 模型格式，"OM"、"MindIR" 或"MindIR_Lite"，忽略大小写。
    - **with_batch_dim** (bool, optional) - 模型输入和输出的shape第一个维度是否是batch维度。默认值：True。
    - **options** (Union[AclOptions, GpuOptions], optional) - 模型的选项，支持 `AclOptions` 或 `GpuOptions` ，从1.6.0版本中已弃用，并将在未来版本中删除，请改用参数 `context` 。默认值：None。
    - **context** (Context) - 用于配置设备环境的上下文信息，值为None时，Serving将依据运行的目标设备配置默认的设备上下文。默认值：None。
    - **without_batch_dim_inputs** (Union[int, tuple[int], list[int]], optional) - 当 `with_batch_dim` 为True时，用于指定shape不包括batch维度的模型输入的索引，比如模型输入0的shape不包括batch维度，则 `without_batch_dim_inputs=(0,)` 。默认值：None。
    - **config_file** (str, optional) - 用于设置混合精度推理的配置文件。文件路径可以是servable_config.py所在目录的绝对路径或相对路径。默认值：None。

    **异常：**

    - **RuntimeError** - 参数的类型或值无效。

    **返回：**

    `Model` ，此模型的标识，用作 `Model.call` 和 `add_stage` 的输入。
