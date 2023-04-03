
.. py:function:: mindspore_serving.server.distributed.declare_servable(rank_size, stage_size, with_batch_dim=True, without_batch_dim_inputs=None, enable_pipeline_infer=False)

    用于在servable_config.py中声明分布式服务，详细可参考
    `基于MindSpore Serving部署分布式推理服务 <https://www.mindspore.cn/serving/docs/zh-CN/r2.0/serving_distributed_example.html>`_ 。

    参数：
        - **rank_size** (int) - 分布式模型的rank大小。
        - **stage_size** (int) - 分布式模型的stage大小。
        - **with_batch_dim** (bool, optional) - 模型输入和输出shape的第一个维度是否是batch维度。默认值：True。
        - **without_batch_dim_inputs** (Union[int, tuple[int], list[int]], optional) - 当 `with_batch_dim` 为True时，用于指定shape不包括batch维度的模型输入的索引，比如模型输入0的shape不包括batch维度，则 `without_batch_dim_inputs=(0,)` 。默认值：None。
        - **enable_pipeline_infer** (bool, optional) - 是否开启流水线并行推理，流水线并行可有效提升推理性能，详情可参考 `流水线并行 <https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0/parallel/pipeline_parallel.html>`_ 。默认值：False。

    返回：
        `Model` ，此模型的标识，可以用来调用 `Model.call` 或作为 `add_stage` 的输入。

    异常：
        - **RuntimeError** - 参数的类型或值无效。
