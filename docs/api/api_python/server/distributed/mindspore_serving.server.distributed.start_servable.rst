
.. py:function:: mindspore_serving.server.distributed.start_servable(servable_directory, servable_name, rank_table_json_file, version_number=1, distributed_address='0.0.0.0:6200', wait_agents_time_in_seconds=0)

    启动在 `servable_directory` 中定义的名为 `servable_name` 的分布式服务。

    **参数：**

    - **servable_directory** (str) - 服务所在的目录。预期有一个名为 `servable_name` 的目录。详细信息可以查看 `通过配置模型提供Servable <https://www.mindspore.cn/serving/docs/zh-CN/master/serving_model.html>`_ 。
    - **servable_name** (str) - 服务名称。
    - **version_number** (int, optional) - 要加载的服务版本号。版本号应为正整数，从1开始。默认值：1。
    - **rank_table_json_file** (str) - rank table json文件名。
    - **distributed_address** (str, optional) - Worker代理（Agent）连接的分布式Worker服务器地址。默认值："0.0.0.0:6200"。
    - **wait_agents_time_in_seconds** (int, optional) - 等待所有Worker代理就绪的最长时间（以秒为单位），0表示无限时间。默认值：0。

    **异常：**

    - **RuntimeError** - 启动分布式服务失败。
