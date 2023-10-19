# MindSpore Serving Release Notes

[View English](./RELEASE.md)

## MindSpore Serving 2.0.2 Release Notes

### 主要特性和增强

- 配套MindSpore 2.2.0版本接口。
- 修复第三方库OpenSSL漏洞CVE-2023-3446、CVE-2023-4807。

### 贡献者

感谢以下人员做出的贡献:

qinzheng, xuyongfei, zhangyinxia, zhoufeng.

欢迎以任何形式对项目提供贡献！

## MindSpore Serving 2.0.0 Release Notes

### 主要特性和增强

- 配套MindSpore 2.0.0rc1版本接口。
- 修复第三方库OpenSSL漏洞CVE-2022-4304、CVE-2022-4450、CVE-2022-4450、CVE-2023-0286、CVE-2023-0464、CVE-2023-0465、CVE-2023-0466。

### 贡献者

感谢以下人员做出的贡献:

qinzheng, xuyongfei, zhangyinxia, zhoufeng.

欢迎以任何形式对项目提供贡献！

## MindSpore Serving 1.8.0 Release Notes

### 主要特性和增强

- [STABLE] Serving部署流水线并行的大模型时，支持流水线并行处理多个推理实例。

### 贡献者

感谢以下人员做出的贡献：

qinzheng, xuyongfei, zhangyinxia, zhoufeng.

欢迎以任何形式对项目提供贡献！

## MindSpore Serving 1.7.0 Release Notes

### 主要特性和增强

- [DEMO] Ascend 310P可以作为MindSpore Serving的硬件后端，详情可参考[MindSpore Serving后端](https://www.mindspore.cn/serving/docs/zh-CN/master/serving_install.html#installation)。
- [DEMO] MindSpore Lite作为MindSpore Serving推理后端时，支持MindIR模型格式，详情可参考[MindSpore Serving后端](https://www.mindspore.cn/serving/docs/zh-CN/master/serving_install.html#installation)。

#### 不建议使用

##### Python API

- `AclOptions`和 `GpuOptions`从1.7.0版本开始被移除，使用 `AscendDeviceInfo`和 `GPUDeviceInfo`替代。
- `register.declare_sevable`和 `register.call_servable`从1.7.0版本开始被移除，使用 `register.declare_model`和 `register.add_stage`替代。
- `register.call_preprocess`，`register.call_preprocess_pipeline`，`register.call_postprocess`和 `register.call_postprocess_pipeline`从1.7.0版本开始被移除，使用 `register.add_stage`替代。

### 贡献者

感谢以下人员做出的贡献:

qinzheng, xuyongfei, zhangyinxia, zhoufeng.

欢迎以任何形式对项目提供贡献！
