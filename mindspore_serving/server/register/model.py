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

from mindspore_serving._mindspore_serving import ModelMeta_, ServableRegister_

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
    """Indicate a model. User should not construct Model object directly, it's need to returned from declare_model
    or declare_servable"""

    def __init__(self, model_key):
        self.model_key = model_key

    def call(self, *args, subgraph=0):
        r"""Invoke the model inference interface based on instances.

        Note:
            This is a beta interface and may not function stably.

        Args:
            subgraph (int, optional): Subgraph index, used when there are multiply sub-graphs in one model.
            args : tuple/list of instances, or inputs of one instance.

        Raises:
            RuntimeError: Inputs are invalid.

        Return:
            Tuple of instances when input parameter 'args' is tuple/list, or outputs of one instance.

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


def declare_model(model_file, model_format, with_batch_dim=True, options=None, without_batch_dim_inputs=None):
    r"""
    Declare one model when importing servable_config.py of one servable.

    Note:
        This interface should take effect when importing servable_config.py by the serving server. Therefore, it's
        recommended that this interface be used globally in servable_config.py.

    Args:
        model_file (Union[str, list[str]]): Model files name.
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
        Model, identification of this model.
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
    if model_format not in ("om", "mindir"):
        raise RuntimeError("model format can only be OM or MindIR")

    meta.local_meta.model_file = model_file
    meta.local_meta.set_model_format(model_format)
    if isinstance(options, dict):
        for k, w in options.items():
            check_type.check_str("options key", k)
            check_type.check_str(k + " value", w)
    elif isinstance(options, _Options):
        # pylint: disable=protected-access
        options = options._as_options_map()
    elif options is not None:
        raise RuntimeError(f"Parameter 'options' should be None, dict of <str,str> or AclOptions, but "
                           f"gotten {type(options)}")
    if options:
        meta.local_meta.options = options

    ServableRegister_.declare_model(meta)
    logger.info(f"Declare model, model_file: {model_file} , model_format: {model_format},  with_batch_dim: "
                f"{with_batch_dim}, options: {options}, without_batch_dim_inputs: {without_batch_dim_inputs}")

    return append_declared_model(meta.common_meta.model_key)


class _Options:
    """ Abstract base class used to build a Options class. """

    def __init__(self, **kwargs):
        """ Initialize Options"""

    def _as_options_map(self):
        """Transfer Options to dict of str,str"""


class AclOptions(_Options):
    """
    Helper class to set acl options.

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
        self.insert_op_cfg_path = ""
        self.input_format = ""
        self.input_shape = ""
        self.output_type = ""
        self.precision_mode = ""
        self.op_select_impl_mode = ""
        val_set_fun = {"insert_op_cfg_path": self._set_insert_op_cfg_path,
                       "input_format": self._set_input_format,
                       "input_shape": self._set_input_shape,
                       "output_type": self._set_output_type,
                       "precision_mode": self._set_precision_mode,
                       "op_select_impl_mode": self._set_op_select_impl_mode}
        for k, w in kwargs.items():
            if k not in val_set_fun:
                raise RuntimeError("Set acl option failed, unsupported option " + k)
            val_set_fun[k](w)

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
            raise RuntimeError(f"Acl option 'input_format' can only be 'ND', 'NCHW', 'NHWC', 'CHWN', 'NC1HWC0', or "
                               f"'NHWC1C0', actually given '{val}'")
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
            raise RuntimeError(f"Acl option 'op_select_impl_mode' can only be 'FP32'(default), 'FP16' or "
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
            raise RuntimeError(f"Acl option 'precision_mode' can only be 'force_fp16'(default), "
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
            raise RuntimeError(f"Acl option 'op_select_impl_mode' can only be 'high_performance'(default) or "
                               f"'high_precision', actually given '{val}'")
        self.op_select_impl_mode = val

    def _as_options_map(self):
        """Transfer AclOptions to dict of str,str"""
        options = {}
        if self.insert_op_cfg_path:
            options['acl_option.insert_op_config_file_path'] = self.insert_op_cfg_path
        if self.input_format:
            options['acl_option.input_format'] = self.input_format
        if self.input_shape:
            options['acl_option.input_shape'] = self.input_shape
        if self.output_type:
            options['acl_option.output_type'] = self.output_type
        if self.precision_mode:
            options['acl_option.precision_mode'] = self.precision_mode
        if self.op_select_impl_mode:
            options['acl_option.op_select_impl_mode'] = self.op_select_impl_mode
        return options


class GpuOptions(_Options):
    """
    Helper class to set gpu options.

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
        self.precision_mode = "origin"
        val_set_fun = {"precision_mode": self._set_precision_mode}
        for k, w in kwargs.items():
            if k not in val_set_fun:
                raise RuntimeError("Set gpu option failed, unsupported option " + k)
            val_set_fun[k](w)

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
            raise RuntimeError(f"Gpu Options 'precision_mode' can only be 'origin', 'fp16'. given '{val}'")
        self.precision_mode = val

    def _as_options_map(self):
        """Transfer GpuOptions to dict of str,str"""
        options = {}
        options['gpu_option.precision_mode'] = self.precision_mode
        return options
