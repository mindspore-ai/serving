
.. py:function:: mindspore_serving.server.start_servables(servable_configs, enable_lite=False)

    用于Serving服务器中启动一个或多个服务，一个模型可结合预处理、后处理提供一个服务，多个模型也可串接组合提供一个服务。

    本接口可以用来启动多个不同的服务。一个服务可以部署在多个设备上，其中每个设备运行一个服务副本。

    在Ascend 910硬件平台上，每个服务的每个副本都独占一个设备。不同的服务或同一服务的不同版本需要部署在不同的芯片上。
    在Ascend 310/310P和GPU硬件平台上，一个设备可以被多个服务共享，不同服务或同一服务的不同版本可以部署在同一设备上，实现设备重用。

    如何配置模型提供服务请查看
    `基于MindSpore Serving部署推理服务 <https://www.mindspore.cn/serving/docs/zh-CN/master/serving_example.html>`_ 和
    `通过配置模型提供Servable <https://www.mindspore.cn/serving/docs/zh-CN/master/serving_model.html>`_ 。

    **参数：**

    - **servable_configs** (Union[ServableStartConfig, list[ServableStartConfig], tuple[ServableStartConfig]]) - 一个或多个服务的启动配置。
    - **enable_lite** (bool) - 是否使用MindSpore Lite推理后端。 默认False。

    **异常：**

    - **RuntimeError** - 启动一个或多个服务失败。相关日志可查看本Serving服务器启动脚本所在目录的子目录serving_logs。
