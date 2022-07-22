
.. py:function:: mindspore_serving.server.start_restful_server(address, max_msg_mb_size=100, ssl_config=None)

    启动RESTful服务器，用于Serving客户端和Serving服务器之间的通信。

    参数：
        - **address** (str) - RESTful服务器地址，地址应为Internet domain socket地址。
        - **max_msg_mb_size** (int, optional) - 最大可接收的RESTful消息大小，以MB为单位，取值范围[1, 512]。默认值：100。
        - **ssl_config** (mindspore_serving.server.SSLConfig, optional) - 服务器的SSL配置，如果是None，则禁用SSL。默认值：None。

    异常：
        - **RuntimeError** - 启动RESTful服务器失败：参数校验失败，RESTful地址错误或端口重复。
