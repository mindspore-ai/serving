﻿
.. py:class:: mindspore_serving.server.register.AscendDeviceInfo(**kwargs)

    用于设置Ascend设备配置。

    参数：
        - **insert_op_cfg_path** (str, 可选) - AIPP配置文件的路径。
        - **input_format** (str, 可选) - 模型输入格式，取值可以是 ``"ND"`` 、 ``"NCHW"`` 、 ``"NHWC"`` 、 ``"CHWN"`` 、 ``"NC1HWC0"`` 或 ``"NHWC1C0"`` 。
        - **input_shape** (str, 可选) - 模型输入形状，如 ``"input_op_name1: n1,c2,h3,w4;input_op_name2: n4,c3,h2,w1"`` 。
        - **output_type** (str, 可选) - 模型输出类型，值可以是 ``"FP16"`` 、 ``"UINT8"`` 或 ``"FP32"`` ，默认值： ``"FP32"`` 。
        - **precision_mode** (str, 可选) - 模型精度模式，取值可以是 ``"force_fp16"`` 、 ``"allow_fp32_to_fp16"`` 、 ``"must_keep_origin_dtype"`` 或者 ``"allow_mix_precision"`` 。默认值： ``"force_fp16"`` 。
        - **op_select_impl_mode** (str, 可选) - 运算符选择模式，值可以是 ``"high_performance"`` 或 ``"high_precision"`` 。默认值： ``"high_performance"`` 。
        - **fusion_switch_config_path** (str, 可选) - 融合配置文件路径，包括图融合和UB融合。系统内置图融合和UB融合规则，默认启用。您可以通过设置此参数禁用指定的融合规则。
        - **buffer_optimize_mode** (str, 可选) - 数据缓存优化策略，值可以是 ``"l1_optimize"`` 、 ``"l2_optimize"`` 、 ``"off_optimize"`` 或者 ``"l1_and_l2_optimize"`` 。默认 ``"l2_optimize"`` 。

    异常：
        - **RuntimeError** - Ascend设备配置无效。
