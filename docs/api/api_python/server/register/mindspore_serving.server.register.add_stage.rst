
.. py:function:: mindspore_serving.server.register.add_stage(stage, *args, outputs_count, batch_size=None, tag=None)

    在服务的 `servable_config.py` 中，通过 `register_method` 装饰（wrap）Python函数定义服务的一个方法（method），
    本接口用于定义这个方法中的一个运行步骤（stage），可以是一个Python函数或者模型。

    .. note:: 入参 `args` 的长度应等于函数或模型的输入个数。

    **参数：**

    - **stage** (Union(function, Model)) - 用户定义的Python函数或由 `declare_model` 返回 `Model` 对象。
    - **outputs_count** (int) - 用户定义的Python函数或模型的输出个数。
    - **batch_size** (int, optional) - 仅当stage是Python函数，且函数一次可以处理多实例时，此参数有效。默认值：None。

        - None，函数的输入将是一个实例的输入。
        - 0，函数的输入将是实例的元组对象，实例元组的最大长度由服务器根据模型的batch大小确定。
        - int value >= 1，函数的输入将是实例的元组对象，实例元组的最大长度是 `batch_size` 指定的值。

    - **args** - stage输入占位符，可以是 `register_method` 装饰（wrap）的函数的输入或其他 `add_stage` 的输出。 `args` 的长度应等于Python函数或模型的输入数量。
    - **tag** (str, optional) - stage的自定义标签，如"preprocess"，默认值：None。

    **异常：**

    - **RuntimeError** - 参数的类型或值无效，或发生其他错误。
