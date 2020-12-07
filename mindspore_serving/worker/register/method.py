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
from easydict import EasyDict

from mindspore_serving._mindspore_serving import ServableStorage_, MethodSignature_, PredictPhaseTag_
from mindspore_serving.worker.common import get_func_name, get_servable_dir
from mindspore_serving.worker import check_type
from .preprocess import register_preprocess, check_preprocess
from .postprocess import register_postprocess, check_postprocess

method_def_context_ = MethodSignature_()
method_def_ast_meta_ = EasyDict()

method_tag_input = PredictPhaseTag_.kPredictPhaseTag_Input
method_tag_preprocess = PredictPhaseTag_.kPredictPhaseTag_Preproces
method_tag_predict = PredictPhaseTag_.kPredictPhaseTag_Predict
method_tag_postprocess = PredictPhaseTag_.kPredictPhaseTag_Postprocess


class _ServableStorage:
    """Declare servable info"""
    def __init__(self):
        self.methods = {}
        self.servable_metas = {}
        self.storage = ServableStorage_.get_instance()

    def declare_servable(self, servable_meta):
        """Declare servable info excluding method, input and output count"""
        self.storage.declare_servable(servable_meta)
        self.servable_metas[servable_meta.servable_name] = servable_meta

    def declare_servable_input_output(self, servable_name, inputs_count, outputs_count):
        """Declare input and output count of servable"""
        self.storage.register_servable_input_output_info(servable_name, inputs_count, outputs_count)
        servable_meta = self.servable_metas[servable_name]
        servable_meta.inputs_count = inputs_count
        servable_meta.outputs_count = outputs_count

    def register_method(self, method_signature):
        """Declare method of servable"""
        self.storage.register_method(method_signature)
        self.methods[method_signature.method_name] = method_signature

    def get_method(self, method_name):
        method = self.methods.get(method_name, None)
        if method is None:
            raise RuntimeError(f"Method {method_name} not found")
        return method

    def get_servable_meta(self, servable_name):
        servable = self.servable_metas.get(servable_name, None)
        if servable is None:
            raise RuntimeError(f"Servable {servable_name} not found")
        return servable


_servable_storage = _ServableStorage()


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


def call_preprocess(preprocess_fun, *args):
    """For method registration, define the inputs data of preprocess
    preprocess_fun can be :
        Python function of preprocess.
        The name of the preprocess implemented by C++ registered by REGISTER_PREPROCESS.
        The input parameters number of implemented python and C++ function should equal to length of 'args'
    """
    if _call_preprocess_name not in method_def_ast_meta_:
        raise RuntimeError(f"Invalid call of ${_call_preprocess_name}")
    inputs_count, outputs_count = method_def_ast_meta_[_call_preprocess_name]

    preprocess_name = preprocess_fun
    if inspect.isfunction(preprocess_fun):
        register_preprocess(preprocess_fun, inputs_count=inputs_count, outputs_count=outputs_count)
        preprocess_name = get_servable_dir() + "." + get_func_name(preprocess_fun)
    else:
        if not isinstance(preprocess_name, str):
            raise RuntimeError(
                f"Check failed, call_preprocess first must be function or str, now is {type(preprocess_name)}")
        check_preprocess(preprocess_name, inputs_count=inputs_count, outputs_count=outputs_count)

    global method_def_context_
    method_def_context_.preprocess_name = preprocess_name
    method_def_context_.preprocess_inputs = [item.as_pair() for item in args]

    return _create_tensor_def_outputs(method_tag_preprocess, outputs_count)


def call_servable(*args):
    """For method registration, define the inputs data of model inference
    The length of 'args' should be equal to model inputs number declared by declare_servable
    """
    servable_name = get_servable_dir()
    inputs_count, outputs_count = method_def_ast_meta_[_call_servable_name]
    _servable_storage.declare_servable_input_output(servable_name, inputs_count, outputs_count)
    if inputs_count != len(args):
        raise RuntimeError(f"Given servable input size {len(args)} not match "
                           f"{servable_name} ast parse size ${inputs_count}")

    global method_def_context_
    method_def_context_.servable_name = servable_name
    method_def_context_.servable_inputs = [item.as_pair() for item in args]

    return _create_tensor_def_outputs(method_tag_predict, outputs_count)


def call_postprocess(postprocess_fun, *args):
    """For method registration, define the inputs data of postprocess
    postprocess_name can be :
        Python function of postprocess.
        The name of the postprocess implemented by C++ registered by REGISTER_POSTPROCESS.
        The input parameters number of implemented python and C++ function should equal to length of 'args'
    """
    if _call_postprocess_name not in method_def_ast_meta_:
        raise RuntimeError(f"Invalid call of ${_call_postprocess_name}")
    inputs_count, outputs_count = method_def_ast_meta_[_call_postprocess_name]

    postprocess_name = postprocess_fun
    if inspect.isfunction(postprocess_fun):
        register_postprocess(postprocess_fun, inputs_count=inputs_count, outputs_count=outputs_count)
        postprocess_name = get_servable_dir() + "." + get_func_name(postprocess_fun)
    else:
        if not isinstance(postprocess_name, str):
            raise RuntimeError(
                f"Check failed, call_postprocess first must be function or str, now is {type(postprocess_name)}")
        check_postprocess(postprocess_name, inputs_count=inputs_count, outputs_count=outputs_count)

    global method_def_context_
    method_def_context_.postprocess_name = postprocess_name
    method_def_context_.postprocess_inputs = [item.as_pair() for item in args]

    return _create_tensor_def_outputs(method_tag_postprocess, outputs_count)


_call_preprocess_name = call_preprocess.__name__
_call_servable_name = call_servable.__name__
_call_postprocess_name = call_postprocess.__name__


def _get_method_def_func_meta(method_def_func):
    """Parse register_method func, and get the input and output count of preproces, servable and postprocess"""
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
        if func_name in (_call_preprocess_name, _call_postprocess_name):
            inputs_count -= 1
        elif func_name == _call_servable_name:
            pass
        else:
            continue

        if inputs_count <= 0:
            raise RuntimeError(f"Invalid {func.id} invoke args")

        print(f"call type {func_name}, inputs count {inputs_count}, outputs count {outputs_count}")
        func_meta[func_name] = [inputs_count, outputs_count]

    if _call_servable_name not in func_meta:
        raise RuntimeError(f"Not find the invoke of {_call_servable_name}")
    return func_meta


def register_method(output_names):
    """register method for servable.
    Define the data flow of preprocess, model inference and postprocess in the method.
    Preprocess and postprocess are optional.
    Example:
        @register_method(output_names="y")
        def method_name(x1, x2):
            x1, x2 = call_preprocess(preprocess_fun, x1, x2)
            y = call_servable(y)
            y = call_postprocess(postprocess_fun, y)
            return y
    """
    output_names = check_type.check_and_as_str_tuple_list('output_names', output_names)

    def register(func):
        name = get_func_name(func)
        sig = inspect.signature(func)
        input_names = []
        for k, v in sig.parameters.items():
            if v.kind == inspect.Parameter.VAR_POSITIONAL:
                raise RuntimeError(name + " input %s cannot be VAR_POSITIONAL !" % k)
            if v.kind == inspect.Parameter.VAR_KEYWORD:
                raise RuntimeError(name + " input %s cannot be VAR_KEYWORD !" % k)
            input_names.append(k)

        input_tensors = []
        for i in range(len(input_names)):
            input_tensors.append(_TensorDef(method_tag_input, i))

        global method_def_context_
        method_def_context_ = MethodSignature_()
        global method_def_ast_meta_
        method_def_ast_meta_ = _get_method_def_func_meta(func)

        output_tensors = func(*tuple(input_tensors))
        if isinstance(output_tensors, _TensorDef):
            output_tensors = (output_tensors,)
        if len(output_tensors) != len(output_names):
            raise RuntimeError(
                f"Method return output size {len(output_tensors)} not match registed {len(output_names)}")

        method_def_context_.method_name = name
        method_def_context_.inputs = input_names
        method_def_context_.outputs = output_names
        method_def_context_.returns = [item.as_pair() for item in output_tensors]
        print("------------Register method: method_name", method_def_context_.method_name,
              ", servable_name", method_def_context_.servable_name, ", inputs", input_names, ", outputs", output_names)

        global _servable_storage
        _servable_storage.register_method(method_def_context_)
        return func

    return register
