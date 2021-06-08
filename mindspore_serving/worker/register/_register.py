# Copyright 2021 Huawei Technologies Co., Ltd
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
"""MindSpore Serving Worker, for servable config."""

from mindspore_serving.server import register
from mindspore_serving.server.common.decorator import deprecated


@deprecated("1.3.0", "mindspore_serving.server.register.call_preprocess_pipeline")
def call_preprocess_pipeline(preprocess_fun, *args):
    r"""For method registration, define the preprocessing pipeline function and its' parameters.

    A single request can include multiple instances, and multiple queued requests will also have multiple instances.
    If you need to process multiple instances through multi thread or other parallel processing capability
    in `preprocess` or `postprocess`, such as using MindData concurrency ability to process multiple input
    images in `preprocess`, MindSpore Serving provides 'call_preprocess_pipeline' and 'call_pstprocess_pipeline'
    to register such preprocessing and postprocessing. For more detail,
    please refer to [Resnet50 model configuration example]
    <https://gitee.com/mindspore/serving/blob/master/example/resnet/resnet50/servable_config.py>`_ .

    Args:
        preprocess_fun (function): Python pipeline function for preprocess.
        args: Preprocess inputs. The length of 'args' should equal to the input parameters number
            of implemented python function.

    Raises:
        RuntimeError: The type or value of the parameters is invalid, or other error happened.

    Examples:
        >>> from mindspore_serving.worker import register
        >>> import numpy as np
        >>> def add_trans_datatype(instances):
        ...     for instance in instances:
        ...         x1 = instance[0]
        ...         x2 = instance[0]
        ...         yield x1.astype(np.float32), x2.astype(np.float32)
        >>>
        >>> register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
        >>>
        >>> @register.register_method(output_names=["y"]) # register add_cast method in add
        >>> def add_cast(x1, x2):
        ...     x1, x2 = register.call_preprocess_pipeline(add_trans_datatype, x1, x2)  # cast input to float32
        ...     y = register.call_servable(x1, x2)
        ...     return y
    """
    return register.call_preprocess_pipeline(preprocess_fun, *args)


@deprecated("1.3.0", "mindspore_serving.server.register.call_preprocess")
def call_preprocess(preprocess_fun, *args):
    r"""For method registration, define the preprocessing function and its' parameters.

    Args:
        preprocess_fun (function): Python function for preprocess.
        args: Preprocess inputs. The length of 'args' should equal to the input parameters number
            of implemented python function.

    Raises:
        RuntimeError: The type or value of the parameters is invalid, or other error happened.

    Examples:
        >>> from mindspore_serving.worker import register
        >>> import numpy as np
        >>> def add_trans_datatype(x1, x2):
        ...     return x1.astype(np.float32), x2.astype(np.float32)
        >>>
        >>> register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
        >>>
        >>> @register.register_method(output_names=["y"]) # register add_cast method in add
        >>> def add_cast(x1, x2):
        ...     x1, x2 = register.call_preprocess(add_trans_datatype, x1, x2)  # cast input to float32
        ...     y = register.call_servable(x1, x2)
        ...     return y
    """
    return register.call_preprocess(preprocess_fun, *args)


@deprecated("1.3.0", "mindspore_serving.server.register.call_servable")
def call_servable(*args):
    r"""For method registration, define the inputs data of model inference

    Note:
        The length of 'args' should be equal to the inputs number of model

    Args:
        args: Model's inputs, the length of 'args' should be equal to the inputs number of model.

    Raises:
        RuntimeError: The type or value of the parameters is invalid, or other error happened.

    Examples:
        >>> from mindspore_serving.worker import register
        >>> register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
        >>>
        >>> @register.register_method(output_names=["y"]) # register add_common method in add
        >>> def add_common(x1, x2):
        ...     y = register.call_servable(x1, x2)
        ...     return y
    """
    return register.call_servable(*args)


@deprecated("1.3.0", "mindspore_serving.server.register.call_postprocess_pipeline")
def call_postprocess_pipeline(postprocess_fun, *args):
    r"""For method registration, define the postprocessing pipeline function and its' parameters.

    A single request can include multiple instances, and multiple queued requests will also have multiple instances.
    If you need to process multiple instances through multi thread or other parallel processing capability
    in `preprocess` or `postprocess`, such as using MindData concurrency ability to process multiple input
    images in `preprocess`, MindSpore Serving provides 'call_preprocess_pipeline' and 'call_pstprocess_pipeline'
    to register such preprocessing and postprocessing. For more detail,
    please refer to [Resnet50 model configuration example]
    <https://gitee.com/mindspore/serving/blob/master/example/resnet/resnet50/servable_config.py>`_ .

    Args:
        postprocess_fun (function): Python pipeline function for postprocess.
        args: Preprocess inputs. The length of 'args' should equal to the input parameters number
            of implemented python function.

    Raises:
        RuntimeError: The type or value of the parameters is invalid, or other error happened.
    """
    return register.call_postprocess_pipeline(postprocess_fun, *args)


@deprecated("1.3.0", "mindspore_serving.server.register.call_postprocess")
def call_postprocess(postprocess_fun, *args):
    r"""For method registration, define the postprocessing function and its' parameters.

    Args:
        postprocess_fun (function): Python function for postprocess.
        args: Preprocess inputs. The length of 'args' should equal to the input parameters number
            of implemented python function.

    Raises:
        RuntimeError: The type or value of the parameters is invalid, or other error happened.
    """
    return register.call_postprocess(postprocess_fun, *args)


@deprecated("1.3.0", "mindspore_serving.server.register.register_method")
def register_method(output_names):
    """register method for servable.

    Define the data flow of preprocess, model inference and postprocess in the method.
    Preprocess and postprocess are optional.

    Args:
        output_names (str, tuple or list of str): The output names of method. The input names is
            the args names of the registered function.

    Raises:
        RuntimeError: The type or value of the parameters is invalid, or other error happened.

    Examples:
        >>> from mindspore_serving.worker import register
        >>> import numpy as np
        >>> def add_trans_datatype(x1, x2):
        ...      return x1.astype(np.float32), x2.astype(np.float32)
        >>>
        >>> register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
        >>>
        >>> @register.register_method(output_names=["y"]) # register add_cast method in add
        >>> def add_cast(x1, x2):
        ...     x1, x2 = register.call_preprocess(add_trans_datatype, x1, x2)  # cast input to float32
        ...     y = register.call_servable(x1, x2)
        ...     return y
    """
    return register.register_method(output_names=output_names)


@deprecated("1.3.0", "mindspore_serving.server.register.declare_servable")
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
        RuntimeError: The type or value of the parameters are invalid.
    """
    return register.declare_servable(servable_file=servable_file, model_format=model_format,
                                     with_batch_dim=with_batch_dim, options=options,
                                     without_batch_dim_inputs=without_batch_dim_inputs)


AclOptions = register.AclOptions
GpuOptions = register.GpuOptions
