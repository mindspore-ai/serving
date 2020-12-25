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

from mindspore_serving._mindspore_serving import ServableMeta_
from mindspore_serving.worker import check_type
from mindspore_serving.worker.common import get_servable_dir
from .method import _servable_storage


def declare_servable(servable_file, model_format, with_batch_dim=True, options=None, without_batch_dim_inputs=None):
    r"""
    declare the servable info.

    Args:
        servable_file (str): Model file name.
        model_format (str): Model format, "OM" or "MindIR", case ignored.
        with_batch_dim (bool): Whether the first shape dim of the inputs and outpus of model is batch dim, default True.
        options (None, AclOptions, map): Options of model, currently AclOptions works.
        without_batch_dim_inputs (None, int, tuple or list of int): Index of inputs that without batch dim
            when with_batch_dim is True.
    Raises:
        RuntimeError: The type or value of the parameters is invalid.
    """

    check_type.check_str('servable_file', servable_file)
    check_type.check_str('model_format', model_format)
    check_type.check_bool('with_batch_dim', with_batch_dim)

    model_format = model_format.lower()
    if model_format not in ("om", "mindir"):
        raise RuntimeError("model format can only be OM or MindIR")

    meta = ServableMeta_()
    meta.servable_name = get_servable_dir()
    meta.servable_file = servable_file
    meta.set_model_format(model_format)
    meta.with_batch_dim = with_batch_dim
    if isinstance(options, dict):
        for k, w in options.items():
            check_type.check_str("options key", k)
            check_type.check_str(k + " value", w)
    elif isinstance(options, AclOptions):
        options = _as_options_map(options)
    elif options is not None:
        raise RuntimeError(f"Parameter 'options' should be None, dict of <str,str> or AclOptions, but "
                           f"gotten {type(options)}")
    if options:
        meta.options = options
    if without_batch_dim_inputs:
        without_batch_dim_inputs = check_type.check_and_as_int_tuple_list('without_batch_dim_inputs',
                                                                          without_batch_dim_inputs, 0)
        meta.without_batch_dim_inputs = without_batch_dim_inputs

    _servable_storage.declare_servable(meta)
    print("------------Declare servable, servable_name:", meta.servable_name,
          ", servable_file:", servable_file, ", model_format:", model_format, ", with_batch_dim:", with_batch_dim,
          ", options:", options, ", without_batch_dim_inputs:", without_batch_dim_inputs)


class AclOptions:
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


def _as_options_map(acl_options):
    """Transfer AclOptions to dict of str,str"""
    options = {}
    if acl_options.insert_op_cfg_path:
        options['mindspore.option.insert_op_config_file_path'] = acl_options.insert_op_cfg_path
    if acl_options.input_format:
        options['mindspore.option.input_format'] = acl_options.input_format
    if acl_options.input_shape:
        options['mindspore.option.input_shape'] = acl_options.input_shape
    if acl_options.output_type:
        options['mindspore.option.output_type'] = acl_options.output_type
    if acl_options.precision_mode:
        options['mindspore.option.precision_mode'] = acl_options.precision_mode
    if acl_options.op_select_impl_mode:
        options['mindspore.option.op_select_impl_mode'] = acl_options.op_select_impl_mode
    return options
