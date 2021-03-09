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

from mindspore_serving._mindspore_serving import ServableMeta_, ServableStorage_
from mindspore_serving.worker import check_type
from mindspore_serving.worker.common import get_servable_dir
from mindspore_serving import log as logger


def declare_servable(servable_file, model_format, with_batch_dim=True, options=None, without_batch_dim_inputs=None):
    r"""
    declare the servable info.

    Args:
        servable_file (str): Model file name.
        model_format (str): Model format, "OM" or "MindIR", case ignored.
        with_batch_dim (bool): Whether the first shape dim of the inputs and outputs of model is batch dim,
             default True.
        options (None, AclOptions, GpuOptions, map): Options of model, currently AclOptions, GpuOptions works.
        without_batch_dim_inputs (None, int, tuple or list of int): Index of inputs that without batch dim
            when with_batch_dim is True.
    Raises:
        RuntimeError: The type or value of the parameters is invalid.
    """

    check_type.check_bool('with_batch_dim', with_batch_dim)

    meta = ServableMeta_()
    meta.common_meta.servable_name = get_servable_dir()
    meta.common_meta.with_batch_dim = with_batch_dim
    if without_batch_dim_inputs:
        without_batch_dim_inputs = check_type.check_and_as_int_tuple_list('without_batch_dim_inputs',
                                                                          without_batch_dim_inputs, 0)
        meta.common_meta.without_batch_dim_inputs = without_batch_dim_inputs

    # init local servable meta info
    check_type.check_str('servable_file', servable_file)
    check_type.check_str('model_format', model_format)
    model_format = model_format.lower()
    if model_format not in ("om", "mindir"):
        raise RuntimeError("model format can only be OM or MindIR")

    meta.local_meta.servable_file = servable_file
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

    ServableStorage_.declare_servable(meta)
    logger.info(f"Declare servable, servable_name: {meta.common_meta.servable_name} "
                f", servable_file: {servable_file} , model_format: {model_format},  with_batch_dim: {with_batch_dim} "
                f", options: {options}, without_batch_dim_inputs: {without_batch_dim_inputs}")


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
        insert_op_cfg_path (str): Path of aipp config file.
        input_format (str): Manually specify the model input format, the value can be "ND", "NCHW", "NHWC",
            "CHWN", "NC1HWC0", or "NHWC1C0".
        input_shape (str): Manually specify the model input shape, such as
            "input_op_name1: n1,c2,h3,w4;input_op_name2: n4,c3,h2,w1",
        output_type (str): Manually specify the model output type, the value can be "FP16", "UINT8"，or "FP32",
            default "FP32".
        precision_mode (str): Model precision mode, the value can be "force_fp16"，"allow_fp32_to_fp16"，
            "must_keep_origin_dtype" or "allow_mix_precision", default "force_fp16".
        op_select_impl_mode (str): The operator selection mode, the value can be "high_performance" or "high_precision",
            default "high_performance".

    Raises:
        RuntimeError: Acl option is invalid, or value is not str.

    Examples:
        >>> from mindspore_serving.worker import register
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
        val_set_fun = {"insert_op_cfg_path": self.set_insert_op_cfg_path,
                       "input_format": self.set_input_format,
                       "input_shape": self.set_input_shape,
                       "output_type": self.set_output_type,
                       "precision_mode": self.set_precision_mode,
                       "op_select_impl_mode": self.set_op_select_impl_mode}
        for k, w in kwargs.items():
            if k not in val_set_fun:
                raise RuntimeError("Set acl option failed, unsupported option " + k)
            val_set_fun[k](w)

    def set_insert_op_cfg_path(self, val):
        """Set option 'insert_op_cfg_path'

        Args:
            val (str): Value of option 'insert_op_cfg_path'.

        Raises:
            RuntimeError: The type of value is not str.
        """
        check_type.check_str('insert_op_cfg_path', val)
        self.insert_op_cfg_path = val

    def set_input_format(self, val):
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

    def set_input_shape(self, val):
        """Set option 'input_shape', manually specify the model input shape, such as
        "input_op_name1: n1,c2,h3,w4;input_op_name2: n4,c3,h2,w1".

        Args:
            val (str): Value of option 'input_shape'.

        Raises:
            RuntimeError: The type of value is not str, or the value is invalid.
        """
        check_type.check_str('input_shape', val)
        self.input_shape = val

    def set_output_type(self, val):
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

    def set_precision_mode(self, val):
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

    def set_op_select_impl_mode(self, val):
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
        enable_trt_infer (bool): Whether enable inference with TensorRT.

    Raises:
        RuntimeError: Gpu option is invalid, or value is not str.

    Examples:
        >>> from mindspore_serving.worker import register
        >>> options = register.GpuOptions(enable_trt_infer=True)
        >>> register.declare_servable(servable_file="deeptext.mindir", model_format="MindIR", options=options)
    """

    def __init__(self, **kwargs):
        super(GpuOptions, self).__init__()
        self.enable_trt_infer = False
        val_set_fun = {"enable_trt_infer": self.set_trt_infer_mode}
        for k, w in kwargs.items():
            if k not in val_set_fun:
                raise RuntimeError("Set gpu option failed, unsupported option " + k)
            val_set_fun[k](w)

    def set_trt_infer_mode(self, val):
        """Set option 'enable_trt_infer'

        Args:
            val (bool): Value of option 'enable_trt_infer'.

        Raises:
            RuntimeError: The type of value is not bool.
        """
        check_type.check_bool('enable_trt_infer', val)
        self.enable_trt_infer = val

    def _as_options_map(self):
        """Transfer GpuOptions to dict of str,str"""
        options = {}
        if self.enable_trt_infer:
            options['gpu_option.enable_trt_infer'] = str(self.enable_trt_infer)
        return options
