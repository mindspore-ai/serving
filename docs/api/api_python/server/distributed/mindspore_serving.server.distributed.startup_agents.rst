
.. py:function:: mindspore_serving.server.distributed.startup_agents(distributed_address, model_files, group_config_files=None, agent_start_port=7000, agent_ip=None, rank_start=None, dec_key=None, dec_mode='AES-GCM')

    在当前计算机上启动所有所需的Worker代理（Agent），这组Worker代理进程将负责本机器设备上的推理任务，详细可参考
    `基于MindSpore Serving部署分布式推理服务 <https://www.mindspore.cn/serving/docs/zh-CN/master/serving_distributed_example.html>`_ 。

    参数：
        - **distributed_address** (str) - Worker代理连接分布式Worker服务器地址。
        - **model_files** (Union[list[str], tuple[str]]) - 当前计算机中需要的所有模型文件，为绝对路径或相对于此启动Python脚本的路径。
        - **group_config_files** (Union[list[str], tuple[str]], optional) - 当前计算机中需要的所有组配置文件，相对于此启动Python脚本的绝对路径或相对路径，为None时表示没有配置文件。默认值：None。
        - **agent_start_port** (int, optional) - Worker代理连接Worker服务器的起始端口号。默认值：7000。
        - **agent_ip** (str, optional) - 本地Worker代理ip，如果为无，则代理ip将从rank table文件中获取。参数 `agent_ip` 和参数 `rank_start` 必须同时有值，或者同时是None。默认值：None。
        - **rank_start** (int, optional) - 此计算机的起始rank id，如果为None，则将从rank table文件中获取rank id。参数 `agent_ip` 和参数 `rank_start` 必须同时有值，或者同时是None。默认值：None。
        - **dec_key** (bytes, optional) - 用于解密的密钥，类型为字节。有效长度为16、24或32。默认值：None。
        - **dec_mode** (str, optional) - 指定解密模式，在设置了 `dec_key` 时生效。值可为： `'AES-GCM'` 或 `'AES-CBC'` 。默认值： `'AES-GCM'` 。

    异常：
        - **RuntimeError** - 启动Worker代理失败。
