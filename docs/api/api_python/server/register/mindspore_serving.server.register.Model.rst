
.. py:class:: mindspore_serving.server.register.Model(model_key)

    用于表示一个声明的模型。用户不应该直接构造 `Model` 对象，而是来自于 `declare_model` 或 `declare_servable` 的返回。

    参数：
        - **model_key** (str) - 模型的唯一标志。

    .. py:method:: call(*args, subgraph=0)

        调用模型推理接口。

        参数：
            - **args** - 实例的元组/列表，或一个实例的输入。
            - **subgraph** (int, 可选) - 子图索引，当一个模型中存在多个子图时使用。默认值：``0``。

        返回：
            当输入参数 `args` 为元组/列表时，返回为instances的元组，当前输入 `args` 为一个实例的输入时，输出为这个实例的输出。

        异常：
            - **RuntimeError** - 输入无效。
