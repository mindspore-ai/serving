# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Servable declaration interface"""

from mindspore_serving._mindspore_serving import ModelMeta_, ServableRegister_, ModelContext_

from mindspore_serving import log as logger
from mindspore_serving.server.common import check_type, deprecated
from .utils import get_servable_dir

g_declared_models = []


@deprecated("1.5.0", "mindspore_serving.server.register.declare_model")
def declare_servable(servable_file, model_format, with_batch_dim=True, options=None, without_batch_dim_inputs=None):
    r"""
    declare one model.

    .. warning::
        'register.declare_servable' is deprecated from version 1.5.0 and will be removed in a future version, use
        :class:`mindspore_serving.server.register.declare_model` instead.

    Args:
        servable_file (Union[str, list[str]]): Model files name.
        model_format (str): Model format, "OM" or "MindIR", case ignored.
        with_batch_dim (bool, optional): Whether the first shape dim of the inputs and outputs of model is batch dim.
            Default: True.
        options (Union[AclOptions, GpuOptions], optional): Options of model, supports AclOptions or GpuOptions.
            Default: None.
        without_batch_dim_inputs (Union[int, tuple[int], list[int]], optional): Index of inputs that without batch
            dim when with_batch_dim is True. Default: None.

    Raises:
        RuntimeError: The type or value of the parameters are invalid.

    Return:
        Model, identification of this model, used as input of add_stage.
    """
    return declare_model(servable_file, model_format, with_batch_dim, options, without_batch_dim_inputs)


class Model:
    """Indicate a model. User should not construct Model object directly, it's need to be returned from `declare_model`
    or `declare_servable`

    Args:
        model_key (str): Model key identifies the model.
    """

    def __init__(self, model_key):
        self.model_key = model_key

    def call(self, *args, subgraph=0):
        r"""Invoke the model inference interface based on instances.

        Args:
            subgraph (int, optional): Subgraph index, used when there are multiply sub-graphs in one model.
            args : tuple/list of instances, or inputs of one instance.

        Return:
            Tuple of instances when input parameter 'args' is tuple/list, or outputs of one instance.

        Raises:
            RuntimeError: Inputs are invalid.

        Examples:
            >>> import numpy as np
            >>> from mindspore_serving.server import register
            >>> import mindspore.dataset.vision.c_transforms as VC
            >>> model = register.declare_model(model_file="resnet_bs32.mindir", model_format="MindIR") # batch_size=32
            >>>
            >>> def preprocess(image):
            ...     decode = VC.Decode()
            ...     resize = VC.Resize([224, 224])
            ...     normalize = VC.Normalize(mean=[125.307, 122.961, 113.8575], std=[51.5865, 50.847, 51.255])
            ...     hwc2chw = VC.HWC2CHW()
            ...     image = decode(image)
            ...     image = resize(image) # [3,224,224]
            ...     image = normalize(image) # [3,224,224]
            ...     image = hwc2chw(image) # [3,224,224]
            ...     return input
            >>>
            >>> def postprocess(score):
            >>>     return np.argmax(score)
            >>>
            >>> def call_resnet_model(image):
            ...     image = preprocess(image)
            ...     score = model.call(image)  # for only one instance
            ...     return postprocess(score)
            >>>
            >>> def call_resnet_model_batch(instances):
            ...     input_instances = []
            ...     for instance in instances:
            ...         image = instance[0] # only one input
            ...         image = preprocess(image) # [3,224,224]
            ...         input_instances.append([image])
            ...     output_instances = model.call(input_instances)  # for multiply instances
            ...     for instance in output_instances:
            ...         score = instance[0]  # only one output for each instance
            ...         index = postprocess(score)
            ...         yield index
            >>>
            >>> @register.register_method(output_names=["index"])
            >>> def predict_v1(image):  # without pipeline, call model with only one instance a time
            ...     index = register.add_stage(call_resnet_model, image, outputs_count=1)
            ...     return index
            >>>
            >>> @register.register_method(output_names=["index"])
            >>> def predict_v2(image):  # without pipeline, call model with maximum 32 instances a time
            ...     index = register.add_stage(call_resnet_model_batch, image, outputs_count=1, batch_size=32)
            ...     return index
            >>>
            >>> @register.register_method(output_names=["index"])
            >>> def predict_v3(image):  # pipeline
            ...     image = register.add_stage(preprocess, image, outputs_count=1)
            ...     score = register.add_stage(model, image, outputs_count=1)
            ...     index = register.add_stage(postprocess, score, outputs_count=1)
            ...     return index
        """
        check_type.check_int("subgraph", subgraph, 0)
        subgraph_str = ""
        if subgraph != 0:
            subgraph_str = " ,subgraph=" + str(subgraph)
        if not args:
            raise RuntimeError(f"Model({self.model_key}{subgraph_str}).call() failed: no inputs provided, the inputs "
                               f"can be call(x1, x2) for single instance or call([[x1, x2], [x1, x2]]) for multi "
                               f"instances.")
        instances = []
        instance_format = False
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            instance_format = True
            inputs = args[0]
            for instance in inputs:
                if not isinstance(instance, (tuple, list)):
                    raise RuntimeError(f"Model({self.model_key}{subgraph_str}).call() failed: inputs format invalid, "
                                       f"the inputs can be call(x1, x2) for single instance or "
                                       f" call([[x1, x2], [x1, x2]]) for multi instances.")
                instances.append(tuple(instance))
        else:
            instances.append(tuple(args))

        output = ServableRegister_.run(self.model_key, tuple(instances), subgraph)
        if not instance_format:
            output = output[0]
            if len(output) == 1:
                return output[0]
            return output
        return output


def append_declared_model(model_key):
    global g_declared_models
    model = Model(model_key)
    g_declared_models.append(model)
    return model


def declare_model(model_file, model_format, with_batch_dim=True, options=None, without_batch_dim_inputs=None,
                  context=None, config_file=None):
    r"""
    Declare one model when importing servable_config.py of one servable.

    Note:
        This interface should take effect when importing servable_config.py by the serving server. Therefore, it's
        recommended that this interface be used globally in servable_config.py.

    .. warning::
        The parameter 'options' is deprecated from version 1.6.0 and will be removed in a future version, use
        parameter 'context' instead.

    Args:
        model_file (Union[str, list[str]]): Model files name.
        model_format (str): Model format, "OM", "MindIR" or "MindIR_Lite", case ignored.
        with_batch_dim (bool, optional): Whether the first shape dim of the inputs and outputs of model is batch dim.
            Default: True.
        options (Union[AclOptions, GpuOptions], optional): Options of model, supports AclOptions or GpuOptions.
            Default: None.
        context (Context): Context is used to store environment variables during execution. If the value is None,
            Serving uses the default device context based on the deployed device. Default: None.
        without_batch_dim_inputs (Union[int, tuple[int], list[int]], optional): Index of inputs that without batch
            dim when with_batch_dim is True. For example, if the shape of input 0 does not include the batch dimension,
            `without_batch_dim_inputs` can be set to `(0,)`. Default: None.
        config_file (str, optional): Config file for model to set mix precision inference. The file path can be an
            absolute path or a relative path to the directory in which servable_config.py resides.
            Default: None.

    Return:
        Model, identification of this model, can be used for `Model.call` or as the inputs of `add_stage`.

    Raises:
        RuntimeError: The type or value of the parameters are invalid.
    """

    check_type.check_bool('with_batch_dim', with_batch_dim)

    meta = ModelMeta_()
    model_file = check_type.check_and_as_str_tuple_list('model_file', model_file)
    meta.common_meta.servable_name = get_servable_dir()
    meta.common_meta.model_key = ";".join(model_file)
    meta.common_meta.with_batch_dim = with_batch_dim
    if without_batch_dim_inputs:
        without_batch_dim_inputs = check_type.check_and_as_int_tuple_list('without_batch_dim_inputs',
                                                                          without_batch_dim_inputs, 0)
        meta.common_meta.without_batch_dim_inputs = without_batch_dim_inputs

    # init local servable meta info
    check_type.check_str('model_format', model_format)
    model_format = model_format.lower()
    if model_format not in ("om", "mindir", "mindir_opt", "mindir_lite"):
        raise RuntimeError("model format can only be OM, MindIR or MindIR_Lite, case ignored")

    meta.local_meta.model_file = model_file
    meta.local_meta.set_model_format(model_format)

    if context is not None:
        if not isinstance(context, Context):
            raise RuntimeError(f"Parameter 'context' should be Context, but gotten {type(context)}")
        meta.local_meta.model_context = context.model_context
    elif isinstance(options, (GpuOptions, AclOptions)):
        logger.warning(
            "'options' will be deprecated in the future, we recommend using 'context', if these two parameters "
            "are both set, options will be ignored")
        meta.local_meta.model_context = options.context.model_context
    elif options is not None:
        raise RuntimeError(f"Parameter 'options' should be None, GpuOptions or AclOptions, but "
                           f"gotten {type(options)}")

    if config_file is not None:
        check_type.check_str("config_file", config_file)
        meta.local_meta.config_file = config_file

    ServableRegister_.declare_model(meta)
    logger.info(f"Declare model, model_file: {model_file} , model_format: {model_format},  with_batch_dim: "
                f"{with_batch_dim}, options: {options}, without_batch_dim_inputs: {without_batch_dim_inputs}"
                f", context: {context}, config file: {config_file}")

    return append_declared_model(meta.common_meta.model_key)


class Context:
    """
    Context is used to customize device configurations. If Context is not specified, MindSpore Serving uses the default
    device configurations. When inference backend is MindSpore Lite and the device type is Ascend or Gpu, the extra
    `CPUDeviceInfo` will be used.

    Args:
        thread_num (int, optional): Set the number of threads at runtime. Only valid when using mindspore lite.
        thread_affinity_core_list (tuple[int], list[int], optional): Set the thread lists to CPU cores.
            Only valid when inference backend is MindSpore Lite.
        enable_parallel (bool, optional): Set the status whether to perform model inference or training in parallel.
            Only valid when inference backend is MindSpore Lite.

    Raises:
        RuntimeError: type or value of input parameters are invalid.

    Examples:
            >>> from mindspore_serving.server import register
            >>> import numpy as np
            >>> context = register.Context(thread_num=1, thread_affinity_core_list=[1,2], enable_parallel=True)
            >>> context.append_device_info(register.GPUDeviceInfo(precision_mode="fp16"))
            >>> model = declare_model(model_file="tensor_add.mindir", model_format="MindIR", context=context)
    """

    def __init__(self, **kwargs):
        self.model_context = ModelContext_()
        val_set_fun = {
            "thread_num": self._set_thread_num,
            "thread_affinity_core_list": self._set_thread_affinity_core_list,
            "enable_parallel": self._set_enable_parallel
        }
        for k, v in kwargs.items():
            if k not in val_set_fun:
                raise RuntimeError("Set context failed, unsupported option " + k)
            val_set_fun[k](v)
        self.device_types = []

    def append_device_info(self, device_info):
        """Append one user-defined device info to the context

         Args:
            device_info (Union[CPUDeviceInfo, GPUDeviceInfo, AscendDeviceInfo]): User-defined device info for one
                device, otherwise default values are used. You can customize device info for each device, and the system
                selects the required device info based on the actual backend device and MindSpore inference package.

         Raises:
            RuntimeError: type or value of input parameters are invalid.
        """
        if not isinstance(device_info, DeviceInfoContext):
            raise RuntimeError(f"Parameter 'device_info' should instance of CPUDeviceInfo, GPUDeviceInfo, or "
                               f"AscendDeviceInfo, but actually {type(device_info)}")
        # pylint: disable=protected-access
        info_map = device_info._as_context_map()
        if not info_map["device_type"]:
            raise RuntimeError("Invalid DeviceInfoContext, device_type cannot be empty")
        device_type = info_map["device_type"]
        if device_type in self.device_types:
            raise RuntimeError(f"Device info of type {device_type} has already been appended")
        self.device_types.append(device_type)
        self.model_context.append_device_info(info_map)

    def _set_thread_num(self, val):
        check_type.check_int("thread_num", val, 1)
        self.model_context.thread_num = val

    def _set_thread_affinity_core_list(self, val):
        check_type.check_int_tuple_list("thread_affinity_core_list", val, 0)
        self.model_context.thread_affinity_core_list = val

    def _set_enable_parallel(self, val):
        check_type.check_bool("enable_parallel", val)
        if val:
            self.model_context.enable_parallel = 1
        else:
            self.model_context.enable_parallel = 0

    def __str__(self):
        res = f"thread_num: {self.model_context.thread_num}, thread_affinity_core_list: " \
              f"{self.model_context.thread_affinity_core_list}, enable_parallel: " \
              f"{self.model_context.enable_parallel}, device_list, {self.model_context.device_list}"
        return res


class DeviceInfoContext:
    def __init__(self):
        """ Initialize context"""

    def _as_context_map(self):
        """Transfer device info to dict of str,str"""
        raise NotImplementedError


class CPUDeviceInfo(DeviceInfoContext):
    """
    Helper class to set cpu device info.

    Args:
        precision_mode(str, optional): Option of model precision, and the value can be "origin", "fp16".
            "origin" indicates that inference is performed with the preciesion defined in the model, and
            "fp16" indicates that inference is performed based on FP16 precision.
            Default: "origin".

    Raises:
        RuntimeError: Cpu option is invalid, or value is not str.

    Examples:
        >>> from mindspore_serving.server import register
        >>> context = register.Context()
        >>> context.append_device_info(register.CPUDeviceInfo(precision_mode="fp16"))
        >>> model = register.declare_model(model_file="deeptext.ms", model_format="MindIR_Lite", context=context)
    """

    def __init__(self, **kwargs):
        super(CPUDeviceInfo, self).__init__()
        self.precision_mode = ""
        val_set_fun = {"precision_mode": self._set_precision_mode}
        for k, w in kwargs.items():
            if k not in val_set_fun:
                raise RuntimeError("Set cpu device info failed, unsupported option " + k)
            val_set_fun[k](w)
        self.context_map = self._as_context_map()

    def _set_precision_mode(self, val):
        check_type.check_str("precision_mode", val)
        if val not in ("origin", "fp16"):
            raise RuntimeError(f"Cpu device info 'precision_mode' can only be 'origin', 'fp16'. given '{val}'")
        self.precision_mode = val

    def _as_context_map(self):
        """Transfer cpu device info to dict of str,str"""
        context_map = {}
        if self.precision_mode:
            context_map["precision_mode"] = self.precision_mode
        context_map["device_type"] = "cpu"
        return context_map


class GPUDeviceInfo(DeviceInfoContext):
    """
    Helper class to set gpu device info.

    Args:
        precision_mode(str, optional): Option of model precision, and the value can be "origin", "fp16".
            "origin" indicates that inference is performed with the preciesion defined in the model, and
            "fp16" indicates that inference is performed based on FP16 precision.
            Default: "origin".

    Raises:
        RuntimeError: Gpu option is invalid, or value is not str.

    Examples:
        >>> from mindspore_serving.server import register
        >>> context = register.Context()
        >>> context.append_device_info(register.GPUDeviceInfo(precision_mode="fp16"))
        >>> model = register.declare_model(model_file="deeptext.mindir", model_format="MindIR", context=context)
    """

    def __init__(self, **kwargs):
        super(GPUDeviceInfo, self).__init__()
        self.precision_mode = ""
        val_set_fun = {"precision_mode": self._set_precision_mode}
        for k, w in kwargs.items():
            if k not in val_set_fun:
                raise RuntimeError("Set gpu device info failed, unsupported option " + k)
            val_set_fun[k](w)
        self.context_map = self._as_context_map()

    def _set_precision_mode(self, val):
        """Set option 'precision_mode', which means inference operator selection, and the value can be "origin",
        "fp16", default "origin".

        Args:
            val (str): Value of option 'precision_mode'. "origin" inference with model definition.
            "fp16" enable FP16 operator selection, with FP32 fallback. Default: "origin".

        Raises:
            RuntimeError: The type of value is not str, or the value is invalid.
        """
        check_type.check_str('precision_mode', val)
        if val not in ("origin", "fp16"):
            raise RuntimeError(f"Gpu device info 'precision_mode' can only be 'origin', 'fp16'. given '{val}'")
        self.precision_mode = val

    def _as_context_map(self):
        """Transfer gpu device info to dict of str,str"""
        context_map = {}
        if self.precision_mode:
            context_map["precision_mode"] = self.precision_mode
        context_map["device_type"] = "gpu"
        return context_map


class AscendDeviceInfo(DeviceInfoContext):
    """
    Helper class to set Ascend device infos.

    Args:
        insert_op_cfg_path (str, optional): Path of aipp config file.
        input_format (str, optional): Manually specify the model input format, the value can be "ND", "NCHW", "NHWC",
            "CHWN", "NC1HWC0", or "NHWC1C0".
        input_shape (str, optional): Manually specify the model input shape, such as
            "input_op_name1: n1,c2,h3,w4;input_op_name2: n4,c3,h2,w1".
        output_type (str, optional): Manually specify the model output type, the value can be "FP16", "UINT8" or "FP32",
            Default: "FP32".
        precision_mode (str, optional): Model precision mode, the value can be "force_fp16"，"allow_fp32_to_fp16"，
            "must_keep_origin_dtype" or "allow_mix_precision". Default: "force_fp16".
        op_select_impl_mode (str, optional): The operator selection mode, the value can be "high_performance" or
            "high_precision". Default: "high_performance".
        fusion_switch_config_path (str, optional): Configuration file path of the convergence rule, including graph
             convergence and UB convergence. The system has built-in graph convergence and UB convergence rules, which
             are enableed by default. You can disable the rules specified in the file by setting this parameter.
        buffer_optimize_mode (str, optional): The value can be "l1_optimize", "l2_optimize", "off_optimize" or
            "l1_and_l2_optimize". Default "l2_optimize".
    Raises:
        RuntimeError: Ascend device info is invalid.

    Examples:
        >>> from mindspore_serving.server import register
        >>> context = register.Context()
        >>> context.append_device_info(register.AscendDeviceInfo(input_format="NCHW"))
        >>> model = register.declare_model(model_file="deeptext.ms", model_format="MindIR_Lite", context=context)
    """

    def __init__(self, **kwargs):
        super(AscendDeviceInfo, self).__init__()
        self.insert_op_cfg_path = ""
        self.input_format = ""
        self.input_shape = ""
        self.output_type = ""
        self.precision_mode = ""
        self.op_select_impl_mode = ""
        self.fusion_switch_config_path = ""
        self.buffer_optimize_mode = ""
        val_set_fun = {"insert_op_cfg_path": self._set_insert_op_cfg_path,
                       "input_format": self._set_input_format,
                       "input_shape": self._set_input_shape,
                       "output_type": self._set_output_type,
                       "precision_mode": self._set_precision_mode,
                       "op_select_impl_mode": self._set_op_select_impl_mode,
                       "fusion_switch_config_path": self._set_fusion_switch_config_path,
                       "buffer_optimize_mode": self._set_buffer_optimize_mode}

        for k, w in kwargs.items():
            if k not in val_set_fun:
                raise RuntimeError("Set ascend device info failed, unsupported parameter " + k)
            val_set_fun[k](w)
        self.context_map = self._as_context_map()

    def _set_insert_op_cfg_path(self, val):
        """Set option 'insert_op_cfg_path'

        Args:
            val (str): Value of option 'insert_op_cfg_path'.

        Raises:
            RuntimeError: The type of value is not str.
        """
        check_type.check_str('insert_op_cfg_path', val)
        self.insert_op_cfg_path = val

    def _set_input_format(self, val):
        """Set option 'input_format', manually specify the model input format, and the value can be
        "ND", "NCHW", "NHWC", "CHWN", "NC1HWC0", or "NHWC1C0".

        Args:
            val (str): Value of option 'input_format', and the value can be "ND", "NCHW", "NHWC",
                "CHWN", "NC1HWC0", or "NHWC1C0".

        Raises:
            RuntimeError: The type of value is not str, or the value is invalid.
        """
        check_type.check_str('input_format', val)
        if val not in ("ND", "NCHW", "NHWC", "CHWN", "NC1HWC0", "NHWC1C0"):
            raise RuntimeError(f"Ascend device info 'input_format' can only be 'ND', 'NCHW', 'NHWC', 'CHWN', 'NC1HWC0'"
                               f", or 'NHWC1C0', actually given '{val}'")
        self.input_format = val

    def _set_input_shape(self, val):
        """Set option 'input_shape', manually specify the model input shape, such as
        "input_op_name1: n1,c2,h3,w4;input_op_name2: n4,c3,h2,w1".

        Args:
            val (str): Value of option 'input_shape'.

        Raises:
            RuntimeError: The type of value is not str, or the value is invalid.
        """
        check_type.check_str('input_shape', val)
        self.input_shape = val

    def _set_output_type(self, val):
        """Set option 'output_type', manually specify the model output type, and the value can be "FP16", "UINT8", or
        "FP32", default "FP32".

        Args:
            val (str): Value of option 'output_type', and the value can be "FP16", "UINT8", or "FP32", default "FP32".

        Raises:
            RuntimeError: The type of value is not str, or the value is invalid.
        """
        check_type.check_str('output_type', val)
        if val not in ("FP32", "FP16", "UINT8"):
            raise RuntimeError(f"Ascend device info 'op_select_impl_mode' can only be 'FP32'(default), 'FP16' or "
                               f"'UINT8', actually given '{val}'")
        self.output_type = val

    def _set_precision_mode(self, val):
        """Set option 'precision_mode',  which means operator selection mode, and the value can be "force_fp16"，
        "force_fp16", "must_keep_origin_dtype", or "allow_mix_precision", default "force_fp16".

        Args:
            val (str): Value of option 'precision_mode', and the value can be "force_fp16"， "force_fp16",
                "must_keep_origin_dtype", or "allow_mix_precision", default "force_fp16".

        Raises:
            RuntimeError: The type of value is not str, or the value is invalid.
        """
        check_type.check_str('precision_mode', val)
        if val not in ("force_fp16", "allow_fp32_to_fp16", "must_keep_origin_dtype", "allow_mix_precision"):
            raise RuntimeError(f"Ascend device info 'precision_mode' can only be 'force_fp16'(default), "
                               f"'allow_fp32_to_fp16' 'must_keep_origin_dtype' or 'allow_mix_precision', "
                               f"actually given '{val}'")
        self.precision_mode = val

    def _set_op_select_impl_mode(self, val):
        """Set option 'op_select_impl_mode', which means model precision mode, and the value can be "high_performance"
        or "high_precision",  default "high_performance".

        Args:
            val (str): Value of option 'op_select_impl_mode'，which can be "high_performance" or "high_precision",
                default "high_performance".

        Raises:
            RuntimeError: The type of value is not str, or the value is invalid.
        """
        check_type.check_str('op_select_impl_mode', val)
        if val not in ("high_performance", "high_precision"):
            raise RuntimeError(f"Ascend device info 'op_select_impl_mode' can only be 'high_performance'(default) or "
                               f"'high_precision', actually given '{val}'")
        self.op_select_impl_mode = val

    def _set_fusion_switch_config_path(self, val):
        check_type.check_str('fusion_switch_config_path', val)
        self.fusion_switch_config_path = val

    def _set_buffer_optimize_mode(self, val):
        check_type.check_str('buffer_optimize_mode', val)
        if val not in ("l1_optimize", "l2_optimize", "off_optimize", "l1_and_l2_optimize"):
            raise RuntimeError(f"Ascend device info 'buffer_optimize_mode' can only be 'off_optimize'(default), "
                               f"'l1_optimize', 'l2_optimize' or 'l1_and_l2_optimize', actually given '{val}'")
        self.buffer_optimize_mode = val

    def _as_context_map(self):
        """Transfer acl device info to dict of str,str"""
        context_map = {}
        if self.insert_op_cfg_path:
            context_map["insert_op_cfg_path"] = self.insert_op_cfg_path
        if self.input_format:
            context_map["input_format"] = self.input_format
        if self.input_shape:
            context_map["input_shape"] = self.input_shape
        if self.output_type:
            context_map["output_type"] = self.output_type
        if self.precision_mode:
            context_map["precision_mode"] = self.precision_mode
        if self.op_select_impl_mode:
            context_map["op_select_impl_mode"] = self.op_select_impl_mode
        if self.buffer_optimize_mode:
            context_map["buffer_optimize_mode"] = self.buffer_optimize_mode
        if self.fusion_switch_config_path:
            context_map["fusion_switch_config_path"] = self.fusion_switch_config_path
        context_map["device_type"] = "ascend"
        return context_map


class AclOptions:
    """
    Helper class to set Ascend device infos.

    .. warning::
        'AclOptions' is deprecated from version 1.6.0 and will be removed in a future version, use
        :class:`mindspore_serving.server.register.AscendDeviceInfo` instead.

    Args:
        insert_op_cfg_path (str, optional): Path of aipp config file.
        input_format (str, optional): Manually specify the model input format, the value can be "ND", "NCHW", "NHWC",
            "CHWN", "NC1HWC0", or "NHWC1C0".
        input_shape (str, optional): Manually specify the model input shape, such as
            "input_op_name1: n1,c2,h3,w4;input_op_name2: n4,c3,h2,w1".
        output_type (str, optional): Manually specify the model output type, the value can be "FP16", "UINT8" or "FP32",
            Default: "FP32".
        precision_mode (str, optional): Model precision mode, the value can be "force_fp16"，"allow_fp32_to_fp16"，
            "must_keep_origin_dtype" or "allow_mix_precision". Default: "force_fp16".
        op_select_impl_mode (str, optional): The operator selection mode, the value can be "high_performance" or
            "high_precision". Default: "high_performance".

    Raises:
        RuntimeError: Acl option is invalid, or value is not str.

    Examples:
        >>> from mindspore_serving.server import register
        >>> options = register.AclOptions(op_select_impl_mode="high_precision", precision_mode="allow_fp32_to_fp16")
        >>> register.declare_servable(servable_file="deeptext.mindir", model_format="MindIR", options=options)
    """

    def __init__(self, **kwargs):
        super(AclOptions, self).__init__()
        logger.warning("'AclOptions' is deprecated from version 1.6.0 and will be removed in a future version, "
                       "use 'mindspore_serving.server.register.AscendDeviceInfo' instead.")
        device_info = AscendDeviceInfo(**kwargs)
        self.context = Context()
        self.context.append_device_info(device_info)


class GpuOptions:
    """
    Helper class to set gpu options.

    .. warning::
        'GpuOptions' is deprecated from version 1.6.0 and will be removed in a future version, use
        :class:`mindspore_serving.server.register.GPUDeviceInfo` instead.

    Args:
        precision_mode(str, optional): inference operator selection, and the value can be "origin", "fp16".
            Default: "origin".

    Raises:
        RuntimeError: Gpu option is invalid, or value is not str.

    Examples:
        >>> from mindspore_serving.server import register
        >>> options = register.GpuOptions(precision_mode="origin")
        >>> register.declare_servable(servable_file="deeptext.mindir", model_format="MindIR", options=options)
    """

    def __init__(self, **kwargs):
        super(GpuOptions, self).__init__()
        logger.warning("'GpuOptions' is deprecated from version 1.6.0 and will be removed in a future version, "
                       "use 'mindspore_serving.server.register.GPUDeviceInfo' instead.")
        device_info = GPUDeviceInfo(**kwargs)
        self.context = Context()
        self.context.append_device_info(device_info)
