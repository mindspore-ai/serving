
.. py:class:: mindspore_serving.client.Client(address, servable_name, method_name, version_number=0, ssl_config=None)

    通过Client访问Serving服务器gRPC接口，可用于创建请求、访问服务和解析结果。

    .. note:: Serving客户端在一个请求中可以发送的最大数据量为512MB，Serving服务器可以接收的最大数据量可以配置为1~512MB，默认为100MB。

    参数：
        - **address** (str) - Serving服务器gRPC接口地址。
        - **servable_name** (str) - Serving服务器提供的服务的名称。
        - **method_name** (str) - 服务中方法的名称。
        - **version_number** (int, optional) - 服务的版本号，``0`` 表示指定所有正在运行的一个或多个版本的服务中最大的版本号。默认值：``0``。
        - **ssl_config** (mindspore_serving.client.SSLConfig, optional) - SSL配置，如果 ``None``，则禁用SSL。默认值：``None``。

    异常：
        - **RuntimeError** - 参数的类型或值无效，或发生其他错误。

    .. py:method:: infer(instances)

        用于创建请求、访问服务、解析和返回结果。

        参数：
            - **instances** (Union[dict, tuple[dict]]) - 一个实例或一组实例的输入，每个实例都是dict。dict的key是输入名称，value是输入值。value的类型可以是Python int、float、bool、str、bytes、numpy scalar或numpy array对象。

        异常：
            - **RuntimeError** - 参数的类型或值无效，或发生其他错误。

    .. py:method:: infer_async(instances)

        用于创建请求，异步访问服务。

        参数：
            - **instances** (Union[dict, tuple[dict]]) - 一个实例或一组实例的输入，每个实例都是dict。dict的key是输入名称，value是输入值。value的类型可以是Python int、float、bool、str、bytes、numpy scalar或numpy array对象。

        异常：
            - **RuntimeError** - 参数的类型或值无效，或发生其他错误。
