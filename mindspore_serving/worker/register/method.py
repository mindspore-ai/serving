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
"""Method registration interface"""

import inspect
import ast
from functools import wraps
from easydict import EasyDict

from mindspore_serving._mindspore_serving import ServableStorage_, MethodSignature_, PredictPhaseTag_
from mindspore_serving.worker.common import get_func_name, get_servable_dir
from mindspore_serving.common import check_type
from mindspore_serving import log as logger
from .preprocess import register_preprocess, check_preprocess
from .postprocess import register_postprocess, check_postprocess

method_def_context_ = MethodSignature_()
method_def_ast_meta_ = EasyDict()

method_tag_input = PredictPhaseTag_.kPredictPhaseTag_Input
method_tag_preprocess = PredictPhaseTag_.kPredictPhaseTag_Preproces
method_tag_predict = PredictPhaseTag_.kPredictPhaseTag_Predict
method_tag_postprocess = PredictPhaseTag_.kPredictPhaseTag_Postprocess


class _TensorDef:
    """Data flow item, for definitions of data flow in a method"""

    def __init__(self, tag, tensor_index):
        self.tag = tag
        self.tensor_index = tensor_index

    def as_pair(self):
        return (self.tag, self.tensor_index)


def _create_tensor_def_outputs(tag, outputs_cnt):
    """Create data flow item for output"""
    result = [_TensorDef(tag, i) for i in range(outputs_cnt)]
    if len(result) == 1:
        return result[0]
    return tuple(result)


def _wrap_fun_to_pipeline(fun, input_count):
    """wrap preprocess and postprocess to pipeline"""
    argspec_len = len(inspect.signature(fun).parameters)
    if argspec_len != input_count:
        raise RuntimeError(f"function {fun.__name__} input args count {argspec_len} not match "
                           f"registered in method count {input_count}")

    @wraps(fun)
    def call_func(instances):
        for instance in instances:
            inputs = []
            for i in range(input_count):
                inputs.append(instance[i])
            yield fun(*inputs)

    return call_func


def call_preprocess_pipeline(preprocess_fun, *args):
    r"""For method registration, define the preprocessing pipeline function and its' parameters.

    A single request can include multiple instances, and multiple queued requests will also have multiple instances.
    If you need to process multiple instances through multi thread or other parallel processing capability
    in `preprocess` or `postprocess`, such as using MindData concurrency ability to process multiple input
    images in `preprocess`, MindSpore Serving provides 'call_preprocess_pipeline' and 'call_pstprocess_pipeline'
    to register such preprocessing and postprocessing. For more detail,
    please refer to `Resnet50 model configuration example <https://gitee.com/mindspore/serving/blob/r1.2/example/resnet/resnet50/servable_config.py>`_.

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
    global method_def_context_
    if method_def_context_.preprocess_name:
        raise RuntimeError(f"Check failed in method '{method_def_context_.method_name}', "
                           f"call_preprocess or call_preprocess_pipeline should not be invoked more than once")
    if method_def_context_.servable_name:
        raise RuntimeError(f"Check failed in method '{method_def_context_.method_name}', "
                           f"call_servable should be invoked after call_preprocess_pipeline")
    if method_def_context_.postprocess_name:
        raise RuntimeError(f"Check failed in method '{method_def_context_.method_name}', call_postprocess "
                           f"or call_postprocess_pipeline should be invoked after call_preprocess_pipeline")

    if _call_preprocess_pipeline_name not in method_def_ast_meta_:
        raise RuntimeError(f"Invalid call of '{_call_preprocess_pipeline_name}'")
    inputs_count, outputs_count = method_def_ast_meta_[_call_preprocess_pipeline_name]

    preprocess_name = preprocess_fun
    if inspect.isfunction(preprocess_fun):
        register_preprocess(preprocess_fun, inputs_count=inputs_count, outputs_count=outputs_count)
        preprocess_name = get_servable_dir() + "." + get_func_name(preprocess_fun)
    else:
        if not isinstance(preprocess_name, str):
            raise RuntimeError(
                f"Check failed in method '{method_def_context_.method_name}', "
                f"call_preprocess first must be function or str, now is {type(preprocess_name)}")
        check_preprocess(preprocess_name, inputs_count=inputs_count, outputs_count=outputs_count)

    method_def_context_.preprocess_name = preprocess_name
    method_def_context_.preprocess_inputs = [item.as_pair() for item in args]

    return _create_tensor_def_outputs(method_tag_preprocess, outputs_count)


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
    global method_def_context_
    if method_def_context_.preprocess_name:
        raise RuntimeError(f"Check failed in method '{method_def_context_.method_name}', "
                           f"call_preprocess or call_preprocess_pipeline should not be invoked more than once")
    if method_def_context_.servable_name:
        raise RuntimeError(f"Check failed in method '{method_def_context_.method_name}', "
                           f"call_servable should be invoked after call_preprocess")
    if method_def_context_.postprocess_name:
        raise RuntimeError(f"Check failed in method '{method_def_context_.method_name}', "
                           f"call_postprocess or call_postprocess_pipeline should be invoked after call_preprocess")

    if _call_preprocess_name not in method_def_ast_meta_:
        raise RuntimeError(f"Invalid call of '{_call_preprocess_name}'")
    inputs_count, outputs_count = method_def_ast_meta_[_call_preprocess_name]

    preprocess_name = preprocess_fun
    if inspect.isfunction(preprocess_fun):
        register_preprocess(_wrap_fun_to_pipeline(preprocess_fun, inputs_count),
                            inputs_count=inputs_count, outputs_count=outputs_count)
        preprocess_name = get_servable_dir() + "." + get_func_name(preprocess_fun)
    else:
        if not isinstance(preprocess_name, str):
            raise RuntimeError(
                f"Check failed in method '{method_def_context_.method_name}', "
                f"call_preprocess first must be function or str, now is {type(preprocess_name)}")
        check_preprocess(preprocess_name, inputs_count=inputs_count, outputs_count=outputs_count)

    method_def_context_.preprocess_name = preprocess_name
    method_def_context_.preprocess_inputs = [item.as_pair() for item in args]

    return _create_tensor_def_outputs(method_tag_preprocess, outputs_count)


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
    global method_def_context_
    if method_def_context_.servable_name:
        raise RuntimeError(f"Check failed in method '{method_def_context_.method_name}', "
                           f"call_servable should not be invoked more than once")
    if method_def_context_.postprocess_name:
        raise RuntimeError(f"Check failed in method '{method_def_context_.method_name}', "
                           f"call_postprocess or call_postprocess_pipeline should be invoked after call_servable")

    servable_name = get_servable_dir()
    inputs_count, outputs_count = method_def_ast_meta_[_call_servable_name]
    ServableStorage_.register_servable_input_output_info(servable_name, inputs_count, outputs_count)
    if inputs_count != len(args):
        raise RuntimeError(f"Check failed in method '{method_def_context_.method_name}', given servable input "
                           f"size {len(args)} not match '{servable_name}' ast parse size {inputs_count}")

    method_def_context_.servable_name = servable_name
    method_def_context_.servable_inputs = [item.as_pair() for item in args]

    return _create_tensor_def_outputs(method_tag_predict, outputs_count)


def call_postprocess_pipeline(postprocess_fun, *args):
    r"""For method registration, define the postprocessing pipeline function and its' parameters.

    A single request can include multiple instances, and multiple queued requests will also have multiple instances.
    If you need to process multiple instances through multi thread or other parallel processing capability
    in `preprocess` or `postprocess`, such as using MindData concurrency ability to process multiple input
    images in `preprocess`, MindSpore Serving provides 'call_preprocess_pipeline' and 'call_pstprocess_pipeline'
    to register such preprocessing and postprocessing. For more detail,
    please refer to `Resnet50 model configuration example <https://gitee.com/mindspore/serving/blob/r1.2/example/resnet/resnet50/servable_config.py>`_.

    Args:
        postprocess_fun (function): Python pipeline function for postprocess.
        args: Preprocess inputs. The length of 'args' should equal to the input parameters number
            of implemented python function.

    Raises:
        RuntimeError: The type or value of the parameters is invalid, or other error happened.
    """
    global method_def_context_
    if method_def_context_.postprocess_name:
        raise RuntimeError(f"Check failed in method '{method_def_context_.method_name}', "
                           f"call_postprocess or call_postprocess_pipeline should not be invoked more than once")

    if _call_postprocess_pipeline_name not in method_def_ast_meta_:
        raise RuntimeError(f"Invalid call of '{_call_postprocess_pipeline_name}'")
    inputs_count, outputs_count = method_def_ast_meta_[_call_postprocess_pipeline_name]

    postprocess_name = postprocess_fun
    if inspect.isfunction(postprocess_fun):
        register_postprocess(postprocess_fun, inputs_count=inputs_count, outputs_count=outputs_count)
        postprocess_name = get_servable_dir() + "." + get_func_name(postprocess_fun)
    else:
        if not isinstance(postprocess_name, str):
            raise RuntimeError(
                f"Check failed in method '{method_def_context_.method_name}', "
                f"call_postprocess first must be function or str, now is {type(postprocess_name)}")
        check_postprocess(postprocess_name, inputs_count=inputs_count, outputs_count=outputs_count)

    method_def_context_.postprocess_name = postprocess_name
    method_def_context_.postprocess_inputs = [item.as_pair() for item in args]

    return _create_tensor_def_outputs(method_tag_postprocess, outputs_count)


def call_postprocess(postprocess_fun, *args):
    r"""For method registration, define the postprocessing function and its' parameters.

    Args:
        postprocess_fun (function): Python function for postprocess.
        args: Preprocess inputs. The length of 'args' should equal to the input parameters number
            of implemented python function.

    Raises:
        RuntimeError: The type or value of the parameters is invalid, or other error happened.
    """
    global method_def_context_
    if method_def_context_.postprocess_name:
        raise RuntimeError(f"Check failed in method '{method_def_context_.method_name}', "
                           f"call_postprocess or call_postprocess_pipeline should not be invoked more than once")

    if _call_postprocess_name not in method_def_ast_meta_:
        raise RuntimeError(f"Invalid call of '{_call_postprocess_name}'")
    inputs_count, outputs_count = method_def_ast_meta_[_call_postprocess_name]

    postprocess_name = postprocess_fun
    if inspect.isfunction(postprocess_fun):
        register_postprocess(_wrap_fun_to_pipeline(postprocess_fun, inputs_count),
                             inputs_count=inputs_count, outputs_count=outputs_count)
        postprocess_name = get_servable_dir() + "." + get_func_name(postprocess_fun)
    else:
        if not isinstance(postprocess_name, str):
            raise RuntimeError(
                f"Check failed in method '{method_def_context_.method_name}', "
                f"call_postprocess first must be function or str, now is {type(postprocess_name)}")
        check_postprocess(postprocess_name, inputs_count=inputs_count, outputs_count=outputs_count)

    method_def_context_.postprocess_name = postprocess_name
    method_def_context_.postprocess_inputs = [item.as_pair() for item in args]

    return _create_tensor_def_outputs(method_tag_postprocess, outputs_count)


_call_preprocess_name = call_preprocess.__name__
_call_servable_name = call_servable.__name__
_call_postprocess_name = call_postprocess.__name__
_call_preprocess_pipeline_name = call_preprocess_pipeline.__name__
_call_postprocess_pipeline_name = call_postprocess_pipeline.__name__


def _get_method_def_func_meta(method_def_func):
    """Parse register_method func, and get the input and output count of preprocess, servable and postprocess"""
    source = inspect.getsource(method_def_func)
    call_list = ast.parse(source).body[0].body
    func_meta = EasyDict()

    for call_item in call_list:
        if not isinstance(call_item, ast.Assign):
            continue
        target = call_item.targets[0]
        if isinstance(target, ast.Name):
            outputs_count = 1
        elif isinstance(target, ast.Tuple):
            outputs_count = len(target.elts)
        else:
            continue

        call = call_item.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Attribute):
            func_name = func.attr
        elif isinstance(func, ast.Name):
            func_name = func.id
        else:
            continue

        inputs_count = len(call.args)
        if func_name in (_call_preprocess_name, _call_preprocess_pipeline_name,
                         _call_postprocess_name, _call_postprocess_pipeline_name):
            inputs_count -= 1
        elif func_name == _call_servable_name:
            pass
        else:
            continue

        if inputs_count <= 0:
            raise RuntimeError(f"Invalid '{func_name}' invoke args")

        logger.info(f"call type '{func_name}', inputs count {inputs_count}, outputs count {outputs_count}")
        func_meta[func_name] = [inputs_count, outputs_count]

    if _call_servable_name not in func_meta:
        raise RuntimeError(f"Not find the invoke of '{_call_servable_name}'")
    return func_meta


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
    output_names = check_type.check_and_as_str_tuple_list('output_names', output_names)

    def register(func):
        name = get_func_name(func)
        sig = inspect.signature(func)
        input_names = []
        for k, v in sig.parameters.items():
            if v.kind == inspect.Parameter.VAR_POSITIONAL:
                raise RuntimeError(f"'{name}' input {k} cannot be VAR_POSITIONAL !")
            if v.kind == inspect.Parameter.VAR_KEYWORD:
                raise RuntimeError(f"'{name}' input {k} cannot be VAR_KEYWORD !")
            if v.kind == inspect.Parameter.KEYWORD_ONLY:
                raise RuntimeError(f"'{name}' input {k} cannot be KEYWORD_ONLY !")
            input_names.append(k)

        input_tensors = []
        for i in range(len(input_names)):
            input_tensors.append(_TensorDef(method_tag_input, i))

        global method_def_context_
        method_def_context_ = MethodSignature_()
        method_def_context_.method_name = name
        method_def_context_.inputs = input_names
        method_def_context_.outputs = output_names

        global method_def_ast_meta_
        method_def_ast_meta_ = _get_method_def_func_meta(func)

        output_tensors = func(*tuple(input_tensors))
        if isinstance(output_tensors, _TensorDef):
            output_tensors = (output_tensors,)
        if len(output_tensors) != len(output_names):
            raise RuntimeError(
                f"Method return output size {len(output_tensors)} not match registered {len(output_names)}")

        method_def_context_.returns = [item.as_pair() for item in output_tensors]
        logger.info(f"Register method: method_name {method_def_context_.method_name} "
                    f", servable_name {method_def_context_.servable_name}, inputs: {input_names}, outputs: "
                    f"{output_names}")

        ServableStorage_.register_method(method_def_context_)
        return func

    return register
