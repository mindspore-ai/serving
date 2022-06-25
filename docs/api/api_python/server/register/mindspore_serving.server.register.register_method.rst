
.. py:function:: mindspore_serving.server.register.register_method(output_names)

    在服务的servable_config.py配置文件中使用，用于注册服务的方法，一个服务可以包括一个或多个方法，每个方法可基于模型提供不同的功能，客户端访问服务时需要指定服务和方法。MindSpore Serving支持由多个Python函数和多个模型组合串接提供服务。

    .. note:: 本接口需要在Serving服务器导入servable_config.py时生效。因此，建议在servable_config.py中全局使用此接口。

    此接口将定义方法的签名和处理流程。

    签名包括方法名称、方法的输入和输出名称。当Serving客户端访问服务时，客户端需要指定服务名称、方法名称，并提供一个或多个推理实例。每个实例通过输入名称指定输入数据，并通过输出名称获取输出结果。

    处理流程由一个或多个阶段（stage）组成，每个阶段可以是一个Python函数或模型。即，一个方法的处理流程可以包括一个或多个Python函数和一个或多个模型。此外，接口还定义了这些阶段之间的数据流。

    **参数：**

    - **output_names** (Union[str, tuple[str], list[str]]) - 指定方法的输出名称。输入名称通过注册函数的参数名称指定。

    **异常：**

    - **RuntimeError** - 参数的类型或值无效，或发生其他错误。
