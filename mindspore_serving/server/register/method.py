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

from mindspore_serving._mindspore_serving import ServableRegister_
from mindspore_serving._mindspore_serving import MethodSignature_
from mindspore_serving import log as logger
from mindspore_serving.server.common import check_type, deprecated
from .utils import get_func_name, get_servable_dir
from .stage_function import register_stage_function, check_stage_function
from .model import g_declared_models, Model

method_def_context_ = MethodSignature_()
cur_stage_index_ = 0
has_called_preprocess_ = False
has_called_servable_ = False
has_called_postprocess_ = False

method_def_ast_meta_ = []


class _TensorDef:
    """Data flow item, for definitions of data flow in a method"""

    def __init__(self, tag, tensor_index):
        self.tag = tag
        self.tensor_index = tensor_index

    def as_pair(self):
        return self.tag, self.tensor_index


def _create_tensor_def_outputs(tag, outputs_cnt):
    """Create data flow item for output"""
    result = [_TensorDef(tag, i) for i in range(outputs_cnt)]
    if len(result) == 1:
        return result[0]
    return tuple(result)


def _wrap_fun_to_batch(fun, input_count):
    """wrap preprocess and postprocess to pipeline"""
    argspec_len = len(inspect.signature(fun).parameters)
    if argspec_len != input_count:
        raise RuntimeError(f"function {fun.__name__} input args count {argspec_len} not match the count {input_count} "
                           f"registered in method")

    @wraps(fun)
    def call_func(instances):
        for instance in instances:
            inputs = []
            for i in range(input_count):
                inputs.append(instance[i])
            yield fun(*inputs)

    return call_func


def _get_stage_outputs_count(call_name):
    global method_def_ast_meta_
    method_name = method_def_context_.method_name
    if call_name not in method_def_ast_meta_:
        raise RuntimeError(
            f"Failed to parse method '{method_name}', complex statements such as conditions and loops are not supported"
            f" in register_method when the interface '{call_name}' is used, use 'add_stage' to replace '{call_name}'")
    _, outputs_count = method_def_ast_meta_[call_name]
    return outputs_count


@deprecated("1.5.0", "mindspore_serving.server.register.add_stage")
def call_preprocess(preprocess_fun, *args):
    r"""For method registration, define the preprocessing function and its' parameters.

    .. warning::
        'call_preprocess' is deprecated from version 1.5.0 and will be removed in a future version, use
        :class:`mindspore_serving.server.register.add_stage` instead.

    Note:
        The length of 'args' should be equal to the inputs number of preprocess_fun.

    Args:
        preprocess_fun (function): Python function for preprocess.
        args: Preprocess inputs. The length of 'args' should equal to the input parameters number
            of implemented python function.

    Raises:
        RuntimeError: The type or value of the parameters are invalid, or other error happened.

    Examples:
        >>> from mindspore_serving.server import register
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
    global has_called_preprocess_, has_called_servable_, has_called_postprocess_
    if has_called_preprocess_:
        raise RuntimeError(f"Check failed in method '{method_def_context_.method_name}', "
                           f"call_preprocess or call_preprocess_pipeline should not be invoked more than once")
    if has_called_servable_:
        raise RuntimeError(f"Check failed in method '{method_def_context_.method_name}', "
                           f"call_servable should be invoked after call_preprocess")
    if has_called_postprocess_:
        raise RuntimeError(f"Check failed in method '{method_def_context_.method_name}', "
                           f"call_postprocess or call_postprocess_pipeline should be invoked after call_preprocess")
    has_called_preprocess_ = True
    outputs_count = _get_stage_outputs_count('call_preprocess')
    return add_stage(preprocess_fun, *args, outputs_count=outputs_count, tag="Preprocess")


@deprecated("1.5.0", "mindspore_serving.server.register.add_stage")
def call_preprocess_pipeline(preprocess_fun, *args):
    r"""For method registration, define the preprocessing pipeline function and its' parameters.

    .. warning::
        'call_preprocess_pipeline' is deprecated from version 1.5.0 and will be removed in a future version, use
        :class:`mindspore_serving.server.register.add_stage` instead.

    A single request can include multiple instances, so multiple queued requests will also have multiple instances.
    If you need to process multiple instances through multi thread or other parallel processing capability
    in `preprocess` or `postprocess`, such as using MindData concurrency ability to process multiple input
    images in `preprocess`, MindSpore Serving provides 'call_preprocess_pipeline' and 'call_postprocess_pipeline'
    to register such preprocessing and postprocessing. For more detail,
    please refer to `Resnet50 model configuration example
    <https://gitee.com/mindspore/serving/blob/r2.0/example/resnet/resnet50/servable_config.py>`_.

    Args:
        preprocess_fun (function): Python pipeline function for preprocess.
        args: Preprocess inputs. The length of 'args' should equal to the input parameters number
            of implemented python function.

    Raises:
        RuntimeError: The type or value of the parameters are invalid, or other error happened.

    Examples:
        >>> from mindspore_serving.server import register
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
    global has_called_preprocess_, has_called_servable_, has_called_postprocess_
    if has_called_preprocess_:
        raise RuntimeError(f"Check failed in method '{method_def_context_.method_name}', "
                           f"call_preprocess or call_preprocess_pipeline should not be invoked more than once")
    if has_called_servable_:
        raise RuntimeError(f"Check failed in method '{method_def_context_.method_name}', "
                           f"call_servable should be invoked after call_preprocess_pipeline")
    if has_called_postprocess_:
        raise RuntimeError(f"Check failed in method '{method_def_context_.method_name}', call_postprocess "
                           f"or call_postprocess_pipeline should be invoked after call_preprocess_pipeline")
    has_called_preprocess_ = True
    outputs_count = _get_stage_outputs_count('call_preprocess_pipeline')
    return add_stage(preprocess_fun, *args, outputs_count=outputs_count, batch_size=0, tag="Preprocess")


@deprecated("1.5.0", "mindspore_serving.server.register.add_stage")
def call_postprocess(postprocess_fun, *args):
    r"""For method registration, define the postprocessing function and its' parameters.

    .. warning::
        'call_postprocess' is deprecated from version 1.5.0 and will be removed in a future version, use
        :class:`mindspore_serving.server.register.add_stage` instead.

    Note:
        The length of 'args' should be equal to the inputs number of postprocess_fun.

    Args:
        postprocess_fun (function): Python function for postprocess.
        args: Preprocess inputs. The length of 'args' should equal to the input parameters number
            of implemented python function.

    Raises:
        RuntimeError: The type or value of the parameters are invalid, or other error happened.
    """
    global method_def_context_
    global has_called_postprocess_
    if has_called_postprocess_:
        raise RuntimeError(f"Check failed in method '{method_def_context_.method_name}', "
                           f"call_postprocess or call_postprocess_pipeline should not be invoked more than once")
    has_called_postprocess_ = True
    outputs_count = _get_stage_outputs_count('call_postprocess')
    return add_stage(postprocess_fun, *args, outputs_count=outputs_count, tag="Postprocess")


@deprecated("1.5.0", "mindspore_serving.server.register.add_stage")
def call_postprocess_pipeline(postprocess_fun, *args):
    r"""For method registration, define the postprocessing pipeline function and its' parameters.

    .. warning::
        'call_postprocess_pipeline' is deprecated from version 1.5.0 and will be removed in a future version, use
        :class:`mindspore_serving.server.register.add_stage` instead.

    A single request can include multiple instances, so multiple queued requests will also have multiple instances.
    If you need to process multiple instances through multi thread or other parallel processing capability
    in `preprocess` or `postprocess`, such as using MindData concurrency ability to process multiple input
    images in `preprocess`, MindSpore Serving provides 'call_preprocess_pipeline' and 'call_postprocess_pipeline'
    to register such preprocessing and postprocessing. For more detail,
    please refer to `Resnet50 model configuration example
    <https://gitee.com/mindspore/serving/blob/r2.0/example/resnet/resnet50/servable_config.py>`_.

    Args:
        postprocess_fun (function): Python pipeline function for postprocess.
        args: Preprocess inputs. The length of 'args' should equal to the input parameters number
            of implemented python function.

    Raises:
        RuntimeError: The type or value of the parameters are invalid, or other error happened.
    """
    global method_def_context_
    global has_called_postprocess_
    if has_called_postprocess_:
        raise RuntimeError(f"Check failed in method '{method_def_context_.method_name}', "
                           f"call_postprocess or call_postprocess_pipeline should not be invoked more than once")
    has_called_postprocess_ = True
    outputs_count = _get_stage_outputs_count('call_postprocess_pipeline')
    return add_stage(postprocess_fun, *args, outputs_count=outputs_count, batch_size=0, tag="Postprocess")


@deprecated("1.5.0", "mindspore_serving.server.register.add_stage")
def call_servable(*args):
    r"""For method registration, define the inputs data of model inference.

    .. warning::
        'call_servable' is deprecated from version 1.5.0 and will be removed in a future version, use
        :class:`mindspore_serving.server.register.add_stage` instead.

    Note:
        The length of 'args' should be equal to the inputs number of model.

    Args:
        args: Model's inputs, the length of 'args' should be equal to the inputs number of model.

    Raises:
        RuntimeError: The type or value of the parameters are invalid, or other error happened.

    Examples:
        >>> from mindspore_serving.server import register
        >>> register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
        >>>
        >>> @register.register_method(output_names=["y"]) # register add_common method in add
        >>> def add_common(x1, x2):
        ...     y = register.call_servable(x1, x2)
        ...     return y
    """
    global method_def_context_
    global has_called_servable_, has_called_postprocess_
    method_name = method_def_context_.method_name
    if has_called_servable_:
        raise RuntimeError(f"Check failed in method '{method_name}', "
                           f"call_servable should not be invoked more than once")
    if has_called_postprocess_:
        raise RuntimeError(f"Check failed in method '{method_name}', "
                           f"call_postprocess or call_postprocess_pipeline should be invoked after call_servable")
    has_called_servable_ = True

    if not g_declared_models:
        raise RuntimeError(f"There is no model declared, you can use declare_model to declare models.")
    outputs_count = _get_stage_outputs_count("call_servable")
    if len(g_declared_models) == 1:
        model = g_declared_models[0]
    else:
        raise RuntimeError(
            f"There are more than one servable declared when the interface 'call_servable' is used, use 'add_stage'"
            f" to replace 'call_servable'")
    return add_stage(model, *args, outputs_count=outputs_count)


def add_stage(stage, *args, outputs_count, batch_size=None, tag=None):
    r"""In the `servable_config.py` file of one servable, we use `register_method` to wrap a Python function to define
    a `method` of the servable, and `add_stage` is used to define a stage of this `method`, which can be a Python
    function or a model.

    Note:
        The length of 'args' should be equal to the inputs number of function or model.

    Args:
        stage (Union(function, Model)): User-defined python function or `Model` object return by declare_model.
        outputs_count (int): Outputs count of the user-defined python function or model.
        batch_size (int, optional): This parameter is valid only when stage is a function and the function
            can process multi instances at a time. default None.

            - None, The input of the function will be the inputs of one instance.
            - 0, The input of the function will be tuple object of instances, and the maximum number
              of the instances is determined by the server based on the batch size of models.
            - int value >= 1, The input of the function will be tuple object of instances, and the maximum number
              of the instances is the value specified by 'batch_size'.

        args: Stage inputs placeholders, which come from the inputs of the function wrapped by register_method or the
            outputs of add_stage. The length of 'args' should equal to the input number of the function or model.
        tag (str, optional): Customized flag of the stage, such as "Preprocess", default None.

    Raises:
        RuntimeError: The type or value of the parameters are invalid, or other error happened.

    Examples:
        >>> import numpy as np
        >>> from mindspore_serving.server import register
        >>> add_model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")
        >>>
        >>> def preprocess(x1, x2):
        ...     return x1.astype(np.float32), x2.astype(np.float32)
        >>>
        >>> @register.register_method(output_names=["y"]) # register add_common method in add
        >>> def add_common(x1, x2):
        ...     x1, x2 = register.add_stage(preprocess, x1, x2, outputs_count=2) # call preprocess in stage 1
        ...     y = register.add_stage(add_model, x1, x2, outputs_count=1) # call add model in stage 2
        ...     return y
    """
    global method_def_context_
    global cur_stage_index_
    method_name = method_def_context_.method_name
    if tag is not None:
        check_type.check_str("tag", tag)
    else:
        tag = ""
    for item in args:
        if not isinstance(item, _TensorDef):
            raise RuntimeError(f"Each value of parameter *args is a placeholder for data and must come from the method"
                               f" inputs or the outputs of add_stage")
    func_inputs = [item.as_pair() for item in args]

    inputs_count = len(args)
    if isinstance(stage, Model):
        if stage not in g_declared_models:
            raise RuntimeError(
                f"Check failed in method '{method_name}', the parameter 'stage' of add_stage must be function "
                f"or Model returned by declare_model, and ensure that interface 'declare_model' can take effect "
                f"when importing servable_config.py by the serving server")
        model = stage
        model_key = model.model_key
        ServableRegister_.register_model_input_output_info(model_key, inputs_count, outputs_count, 0)
        method_def_context_.add_stage_model(model_key, func_inputs, 0, tag)
    elif inspect.isfunction(stage):
        if batch_size is None:
            register_stage_function(method_name, _wrap_fun_to_batch(stage, inputs_count),
                                    inputs_count=inputs_count, outputs_count=outputs_count, use_with_size=False)
            batch_size = 0
        else:
            check_type.check_int("batch_size", batch_size, 0)
            register_stage_function(method_name, stage, inputs_count=inputs_count, outputs_count=outputs_count,
                                    use_with_size=True)
        func_name = get_servable_dir() + "." + get_func_name(stage)
        method_def_context_.add_stage_function(func_name, func_inputs, batch_size, tag)
    else:
        if not isinstance(stage, str):
            raise RuntimeError(
                f"Check failed in method '{method_name}', the parameter 'stage' of add_stage must be function "
                f"or Model returned by declare_model, now is {type(stage)}")
        func_name = stage
        check_stage_function(method_name, func_name, inputs_count=inputs_count, outputs_count=outputs_count)
        method_def_context_.add_stage_function(func_name, func_inputs, 0, tag)

    cur_stage_index_ += 1  # call_xxx stage index start begin 1
    return _create_tensor_def_outputs(cur_stage_index_, outputs_count)


_call_servable_name = call_servable.__name__
_call_stage_names = [call_preprocess.__name__, call_postprocess.__name__]
_call_stage_batch_names = [call_preprocess_pipeline.__name__, call_postprocess_pipeline.__name__]


def _ast_node_info(method_def_func, ast_node):
    """Ast node code info"""
    func_name = method_def_func.__name__
    func_codes = inspect.getsource(method_def_func)
    func_codes_lines = func_codes.split("\n")
    _, start_lineno = inspect.findsource(method_def_func)

    codes = ""
    if hasattr(ast_node, "end_lineno"):
        end_lineno = ast_node.end_lineno
    else:
        end_lineno = ast_node.lineno
    for line in range(ast_node.lineno, end_lineno + 1):
        codes += func_codes_lines[line - 1] + "\n"
    lineno = ast_node.lineno + start_lineno
    end_lineno = end_lineno + start_lineno
    if lineno != end_lineno:
        line_info = f"{lineno}~{end_lineno}"
    else:
        line_info = f"{lineno}"
    return f"line {line_info} in {func_name}, code: \n" + codes


def _get_method_def_stage_meta(method_def_func):
    """Parse register_method func, and get the input and output count of preprocess, servable and postprocess"""
    source = inspect.getsource(method_def_func)
    method_name = method_def_func.__name__
    call_list = ast.parse(source).body[0].body
    func_meta = {}
    code_infos = []
    code_other = None

    def update_other_code(code):
        nonlocal code_other
        if not code_other:
            code_other = code

    for call_item in call_list:
        if isinstance(call_item, ast.Return):
            continue
        if isinstance(call_item, ast.Expr):
            continue
        if not isinstance(call_item, ast.Assign):
            update_other_code(call_item)
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
            update_other_code(call_item)
            continue

        inputs_count = len(call.args)
        if func_name in _call_stage_names or func_name in _call_stage_batch_names:
            inputs_count -= 1
        elif func_name == _call_servable_name:
            pass
        else:
            update_other_code(call_item)
            continue
        if inputs_count <= 0:
            raise RuntimeError(f"Invalid '{func_name}' invoke args")

        logger.info(f"stage {len(func_meta) + 1} call type '{func_name}', inputs count {inputs_count}, "
                    f"outputs count {outputs_count}")
        func_meta[func_name] = [inputs_count, outputs_count]
        code_infos.append([call_item, func_name])
    if code_infos and code_other:
        call_names = [item[1] for item in code_infos]
        call_names = ";".join(call_names)
        raise RuntimeError(
            f"Failed to parse method '{method_name}', complex statements such as conditions and loops are not supported"
            f" in register_method when the interface '{call_names}' is used, use 'add_stage' to replace '{call_names}',"
            f" code {type(code_other)}: {_ast_node_info(method_def_func, code_other)}")

    if code_infos and _call_servable_name not in func_meta:
        raise RuntimeError(f"Not find the invoke of '{_call_servable_name}'")
    return func_meta


def register_method(output_names):
    """Define a method of the servable when importing servable_config.py of one servable. One servable can include one
    or more methods, and eache method provides different services base on models. A client needs to specify the
    servable name and method name when accessing one service. MindSpore Serving supports a service consisting of
    multiple python functions and multiple models.

    Note:
        This interface should take effect when importing servable_config.py by the serving server. Therefore, it's
        recommended that this interface be used globally in servable_config.py.

    This interface will define the signatures and pipeline of the method.

    The signatures include the method name, input and outputs names of the method. When accessing a service, the client
    needs to specify the servable name, the method name, and provide one or more inference instances. Each instance
    specifies the input data by the input names and obtains the output data by the outputs names.

    The pipeline consists of one or more stages, each stage can be a python function or a model. This is, a pipline can
    include one or more python functions and one or more models. In addition, the interface also defines the data flow
    of these stages.

    Args:
        output_names (Union[str, tuple[str], list[str]]): The output names of method. The input names is
            the args names of the registered function.

    Raises:
        RuntimeError: The type or value of the parameters are invalid, or other error happened.

    Examples:
        >>> from mindspore_serving.server import register
        >>> add_model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR")
        >>> sub_model = register.declare_model(model_file="tensor_sub.mindir", model_format="MindIR")
        >>>
        >>> @register.register_method(output_names=["y"]) # register predict method in servable
        >>> def predict(x1, x2, x3): # x1+x2-x3
        ...     y = register.add_stage(add_model, x1, x2, outputs_count=1)
        ...     y = register.add_stage(sub_model, y, x3, outputs_count=1)
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
            input_tensors.append(_TensorDef(0, i))

        servable_name = get_servable_dir()
        global method_def_context_
        method_def_context_ = MethodSignature_()
        method_def_context_.servable_name = servable_name
        method_def_context_.method_name = name
        method_def_context_.inputs = input_names
        method_def_context_.outputs = output_names

        global method_def_ast_meta_
        method_def_ast_meta_ = _get_method_def_stage_meta(func)
        global cur_stage_index_
        cur_stage_index_ = 0

        global has_called_preprocess_, has_called_servable_, has_called_postprocess_
        has_called_preprocess_ = False
        has_called_servable_ = False
        has_called_postprocess_ = False

        output_tensors = func(*tuple(input_tensors))
        if method_def_ast_meta_ and cur_stage_index_ != len(method_def_ast_meta_):
            raise RuntimeError(f"Failed to parse method {name}, the number of stages obtained through the AST "
                               f"{len(method_def_ast_meta_)} is inconsistent with the running result {cur_stage_index_}"
                               f". Condition and loop statements are not supported in methods currently.")

        if isinstance(output_tensors, _TensorDef):
            output_tensors = (output_tensors,)

        for item in output_tensors:
            if not isinstance(item, _TensorDef):
                raise RuntimeError(f"Each value returned is a placeholder for data and must come from the method"
                                   f" inputs or the outputs of add_stage")

        if len(output_tensors) != len(output_names):
            raise RuntimeError(
                f"Method return output size {len(output_tensors)} not match registered {len(output_names)}")

        return_inputs = [item.as_pair() for item in output_tensors]
        method_def_context_.set_return(return_inputs)
        logger.info(f"Register method: method_name {method_def_context_.method_name}, "
                    f"inputs: {input_names}, outputs: {output_names}")

        ServableRegister_.register_method(method_def_context_)
        return func

    return register
