
.. py:function:: mindspore_serving.server.start_grpc_server(address, max_msg_mb_size=100, ssl_config=None)

    启动gRPC服务器，用于Serving客户端和Serving服务器之间的通信。

    参数：
        - **address** (str) - gRPC服务器地址，地址可以是 `{ip}:{port}` 或 `unix:{unix_domain_file_path}` 。

          - `{ip}:{port}` - Internet domain socket地址。
          - `unix:{unix_domain_file_path}` - Unix domain socket地址，用于与同一台计算机上的多个进程通信。 `{unix_domain_file_path}` 可以是相对路径或绝对路径，但文件所在的目录必须已经存在。

        - **max_msg_mb_size** (int, 可选) - 可接收的最大gRPC消息大小（MB），取值范围[1, 512]。默认值：``100``。
        - **ssl_config** (mindspore_serving.server.SSLConfig, 可选) - 服务器的SSL配置，如果 ``None``，则禁用SSL。默认值：``None``。

    异常：
        - **RuntimeError** - 启动gRPC服务器失败：参数校验失败，gRPC地址错误或端口重复。
