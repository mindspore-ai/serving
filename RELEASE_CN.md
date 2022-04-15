# MindSpore Serving Release Notes

[View English](./RELEASE.md)

## MindSpore Serving 1.7.0 Release Notes

### 主要特性和增强

- [DEMO] Ascend 710可以作为MindSpore Serving的硬件后端，详情可参考[MindSpore Serving后端](https://www.mindspore.cn/serving/docs/zh-CN/master/serving_install.html#installation).
- [DEMO] MindSpore Lite作为MindSpore Serving推理后端时，支持MindIR模型格式，详情可参考[MindSpore Serving后端](https://www.mindspore.cn/serving/docs/zh-CN/master/serving_install.html#installation).

#### 不建议使用

##### Python API

- `AclOptions`和 `GpuOptions`从1.7.0版本开始被移除，使用 `AscendDeviceInfo`和 `GPUDeviceInfo`替代。
- `register.declare_sevable`和 `register.call_servable`从1.7.0版本开始被移除，使用 `register.declare_model`和 `register.add_stage`替代.
- `register.call_preprocess`，`register.call_preprocess_pipeline`，`register.call_postprocess`和 `register.call_postprocess_pipeline`从1.7.0版本开始被移除，使用 `register.add_stage`替代.

### 贡献者

感谢以下人员做出的贡献:

qinzheng, xuyongfei, zhangyinxia, zhoufeng.

欢迎以任何形式对项目提供贡献！
