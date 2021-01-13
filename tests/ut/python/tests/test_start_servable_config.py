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
"""test Serving: test servable_config"""

from mindspore_serving import worker
from common import ServingTestBase, serving_test

# test servable_config.py
servable_config_import = r"""
import numpy as np
from mindspore_serving.worker import register
"""

servable_config_declare_servable = r"""
register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
"""

servable_config_preprocess_cast = r"""
def add_trans_datatype(x1, x2):
    return x1.astype(np.float32), x2.astype(np.float32)
"""

servable_config_method_add_common = r"""
@register.register_method(output_names=["y"])
def add_common(x1, x2):  # only support float32 inputs
    y = register.call_servable(x1, x2)
    return y
"""

servable_config_method_add_cast = r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.call_preprocess(add_trans_datatype, x1, x2)  # cast input to float32
    y = register.call_servable(x1, x2)
    return y
"""


@serving_test
def test_register_method_common_success():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += servable_config_method_add_common
    servable_content += servable_config_method_add_cast

    base.init_servable_with_servable_config(1, servable_content)
    worker.start_servable_in_master(base.servable_dir, base.servable_name)


@serving_test
def test_register_method_no_declare_servable_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    # servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += servable_config_method_add_common
    servable_content += servable_config_method_add_cast

    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "RegisterInputOutputInfo failed, cannot find servable" in str(e)


@serving_test
def test_register_method_no_method_registered_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    # servable_content += servable_config_method_add_common
    # servable_content += servable_config_method_add_cast

    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "There is no method registered for servable" in str(e)


@serving_test
def test_register_method_reference_invalid_preprocess_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    # servable_content += servable_config_preprocess_cast
    servable_content += servable_config_method_add_common
    servable_content += servable_config_method_add_cast

    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except NameError as e:
        assert "name 'add_trans_datatype' is not defined" in str(e)


@serving_test
def test_register_method_preprocess_inputs_count_not_match_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def add_trans_datatype(x1, x2, x3):
    return x1.astype(np.float32), x2.astype(np.float32), x3.astype(np.float32)
    """
    servable_content += servable_config_method_add_common
    servable_content += servable_config_method_add_cast

    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "function add_trans_datatype input args count 3 not match registered in method count 2" in str(e)


@serving_test
def test_register_method_preprocess_inputs_count_not_match2_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += r"""
def add_trans_datatype(x1):
    return x1.astype(np.float32)
    """
    servable_content += servable_config_method_add_common
    servable_content += servable_config_method_add_cast

    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "function add_trans_datatype input args count 1 not match registered in method count 2" in str(e)


@serving_test
def test_register_method_postprocess_inputs_count_not_match_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
def add_trans_datatype_back(x1, x2):
    return x1.astype(np.float32), x2.astype(np.float32)

@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.call_preprocess(add_trans_datatype, x1, x2)  # cast input to float32
    y = register.call_servable(x1, x2)    
    y1, y2 = register.call_postprocess(add_trans_datatype_back, y)  # cast output to int32
    return y1
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "function add_trans_datatype_back input args count 2 not match registered in method count 1" in str(e)


# preprocess order error
@serving_test
def test_register_method_preprocess_after_predict_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.call_servable(x1, x2)    
    x1, x2 = register.call_preprocess(add_trans_datatype, x1, x2)
    return x1
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "call_servable should be invoked after call_preprocess" in str(e)


@serving_test
def test_register_method_preprocess_after_postprocess_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.call_postprocess(add_trans_datatype, x1, x2)
    x1, x2 = register.call_preprocess(add_trans_datatype, x1, x2)
    y = register.call_servable(x1, x2)    
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "call_postprocess or call_postprocess_pipeline should be invoked after call_preprocess" in str(e)


@serving_test
def test_register_method_preprocess_after_postprocess_pipeline_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.call_postprocess_pipeline(add_trans_datatype, x1, x2)
    x1, x2 = register.call_preprocess(add_trans_datatype, x1, x2)
    y = register.call_servable(x1, x2)    
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "call_postprocess or call_postprocess_pipeline should be invoked after call_preprocess" in str(e)


# preprocess_pipeline order error
@serving_test
def test_register_method_preprocess_pipeline_after_predict_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.call_servable(x1, x2)    
    x1, x2 = register.call_preprocess_pipeline(add_trans_datatype, x1, x2)
    return x1
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "call_servable should be invoked after call_preprocess_pipeline" in str(e)


@serving_test
def test_register_method_preprocess_pipeline_after_postprocess_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.call_postprocess(add_trans_datatype, x1, x2)
    x1, x2 = register.call_preprocess_pipeline(add_trans_datatype, x1, x2)
    y = register.call_servable(x1, x2)    
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "call_postprocess or call_postprocess_pipeline should be invoked after call_preprocess_pipeline" \
               in str(e)


@serving_test
def test_register_method_preprocess_pipeline_after_postprocess_pipeline_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.call_postprocess_pipeline(add_trans_datatype, x1, x2)
    x1, x2 = register.call_preprocess_pipeline(add_trans_datatype, x1, x2)
    y = register.call_servable(x1, x2)    
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "call_postprocess or call_postprocess_pipeline should be invoked after call_preprocess_pipeline" \
               in str(e)


# repeat preprocess
@serving_test
def test_register_method_preprocess_twice_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.call_preprocess(add_trans_datatype, x1, x2)
    x1, x2 = register.call_preprocess(add_trans_datatype, x1, x2)
    y = register.call_servable(x1, x2)    
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "call_preprocess or call_preprocess_pipeline should not be invoked more than once" in str(e)


@serving_test
def test_register_method_preprocess_twice2_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.call_preprocess(add_trans_datatype, x1, x2)
    x1, x2 = register.call_preprocess_pipeline(add_trans_datatype, x1, x2)
    y = register.call_servable(x1, x2)    
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "call_preprocess or call_preprocess_pipeline should not be invoked more than once" in str(e)


@serving_test
def test_register_method_preprocess_pipeline_twice_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.call_preprocess_pipeline(add_trans_datatype, x1, x2)
    x1, x2 = register.call_preprocess_pipeline(add_trans_datatype, x1, x2)
    y = register.call_servable(x1, x2)    
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "call_preprocess or call_preprocess_pipeline should not be invoked more than once" in str(e)


# repeat postprocess
@serving_test
def test_register_method_postprocess_twice_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
def postprocess(y):
    return y.astype(np.int32)
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.call_servable(x1, x2)  
    y = register.call_postprocess(postprocess, y)
    y = register.call_postprocess(postprocess, y)  
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "call_postprocess or call_postprocess_pipeline should not be invoked more than once" in str(e)


@serving_test
def test_register_method_postprocess_twice2_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
def postprocess(y):
    return y.astype(np.int32)
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.call_servable(x1, x2)  
    y = register.call_postprocess_pipeline(postprocess, y)
    y = register.call_postprocess(postprocess, y)  
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "call_postprocess or call_postprocess_pipeline should not be invoked more than once" in str(e)


@serving_test
def test_register_method_postprocess_pipeline_twice_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
def postprocess(y):
    return y.astype(np.int32)
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.call_servable(x1, x2)  
    y = register.call_postprocess_pipeline(postprocess, y)
    y = register.call_postprocess_pipeline(postprocess, y)  
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "call_postprocess or call_postprocess_pipeline should not be invoked more than once" in str(e)


# call servable repeat
@serving_test
def test_register_method_call_servable_twice_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.call_preprocess(add_trans_datatype, x1, x2)
    y = register.call_servable(x1, x2)
    y = register.call_servable(x1, x2)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "call_servable should not be invoked more than once" in str(e)


@serving_test
def test_register_method_call_servable_after_postprocess_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.call_postprocess(add_trans_datatype, x1, x2)
    y = register.call_servable(x1, x2)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "call_postprocess or call_postprocess_pipeline should be invoked after call_servable" in str(e)


@serving_test
def test_register_method_call_servable_after_postprocess_pipeline_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.call_postprocess_pipeline(add_trans_datatype, x1, x2)
    y = register.call_servable(x1, x2)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "call_postprocess or call_postprocess_pipeline should be invoked after call_servable" in str(e)


@serving_test
def test_register_method_call_preprocess_pipeline_input_count_not_same_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.call_preprocess_pipeline(add_trans_datatype, x1, x2)
    y = register.call_servable(x1, x2)
    return y

@register.register_method(output_names=["y"])
def add_cast2(x1, x2):
    x1, x2 = register.call_preprocess_pipeline(add_trans_datatype, x1, x2, x2)
    y = register.call_servable(x1, x2)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert f"Preprocess '{base.servable_name}.add_trans_datatype' inputs count 3 " \
               f"not match last registered count 2" in str(e)


@serving_test
def test_register_method_call_preprocess_pipeline_output_count_not_same_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2, x3 = register.call_preprocess_pipeline(add_trans_datatype, x1, x2)
    y = register.call_servable(x1, x2)
    return y

@register.register_method(output_names=["y"])
def add_cast2(x1, x2):
    x1, x2 = register.call_preprocess_pipeline(add_trans_datatype, x1, x2,)
    y = register.call_servable(x1, x2)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert f"Preprocess '{base.servable_name}.add_trans_datatype' outputs count 2 " \
               f"not match last registered count 3" in str(e)


@serving_test
def test_register_method_call_postprocess_pipeline_input_count_not_same_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.call_servable(x1, x2)
    x1, x2 = register.call_postprocess_pipeline(add_trans_datatype, x1, x2)
    return y

@register.register_method(output_names=["y"])
def add_cast2(x1, x2):
    y = register.call_servable(x1, x2)    
    x1, x2 = register.call_postprocess_pipeline(add_trans_datatype, x1, x2, y)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert f"Postprocess '{base.servable_name}.add_trans_datatype' inputs count 3 " \
               f"not match last registered count 2" in str(e)


@serving_test
def test_register_method_call_postprocess_pipeline_output_count_not_same_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.call_servable(x1, x2)
    x1, x2 = register.call_postprocess_pipeline(add_trans_datatype, x1, x2)
    return y

@register.register_method(output_names=["y"])
def add_cast2(x1, x2):
    y = register.call_servable(x1, x2)    
    x1, x2, x3 = register.call_postprocess_pipeline(add_trans_datatype, x1, x2)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert f"Postprocess '{base.servable_name}.add_trans_datatype' outputs count 3 " \
               f"not match last registered count 2" in str(e)


@serving_test
def test_register_method_call_servable_input_count_not_same_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.call_servable(x1, x2)
    return y

@register.register_method(output_names=["y"])
def add_cast2(x1, x2):
    y = register.call_servable(x1, x2, x2)    
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "RegisterInputOutputInfo failed, inputs count 3 not match old count 2" in str(e)


@serving_test
def test_register_method_call_servable_output_count_not_same_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.call_servable(x1, x2)
    return y

@register.register_method(output_names=["y"])
def add_cast2(x1, x2):
    y, y2 = register.call_servable(x1, x2)    
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "RegisterInputOutputInfo failed, outputs count 2 not match old count 1" in str(e)


@serving_test
def test_register_method_call_servable_input_count_not_match_model_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.call_servable(x1, x2, x2)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "The inputs count 3 registered in method not equal to the count 2 defined in servable" in str(e)


@serving_test
def test_register_method_call_servable_output_count_not_match_model_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y, y2 = register.call_servable(x1, x2)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "The outputs count 2 registered in method not equal to the count 1 defined in servable" in str(e)


@serving_test
def test_register_method_method_output_count_not_match_output_names_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.call_servable(x1, x2)
    return y, x2
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "Method return output size 2 not match registered 1" in str(e)


@serving_test
def test_register_method_method_registered_repeat_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.call_servable(x1, x2)
    return y

@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.call_servable(x1, x2)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "Method add_cast has been registered more than once." in str(e)


@serving_test
def test_register_method_input_arg_invalid_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, **x2):
    y = register.call_servable(x1, x2)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "'add_cast' input x2 cannot be VAR_KEYWORD !" in str(e)


@serving_test
def test_register_method_input_arg_invalid2_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, *x2):
    y = register.call_servable(x1, x2)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except RuntimeError as e:
        assert "'add_cast' input x2 cannot be VAR_POSITIONAL !" in str(e)


@serving_test
def test_register_method_call_preprocess_invalid_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.call_preprocess(add_trans_datatype, x1, np.ones([2,2]))
    y = register.call_servable(x1, x2)
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except AttributeError as e:
        assert "'numpy.ndarray' object has no attribute 'as_pair'" in str(e)


@serving_test
def test_register_method_call_servable_invalid_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.call_servable(x1, np.ones([2,2]))
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except AttributeError as e:
        assert "'numpy.ndarray' object has no attribute 'as_pair'" in str(e)


@serving_test
def test_register_method_call_postprocess_invalid_input_failed():
    base = ServingTestBase()
    servable_content = servable_config_import
    servable_content += servable_config_declare_servable
    servable_content += servable_config_preprocess_cast
    servable_content += r"""
def postprocess(y, data):
    return y.astype(np.int32)
    
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    y = register.call_servable(x1, x2)
    y = register.call_postprocess(postprocess, y, np.ones([2,2]))
    return y
"""
    base.init_servable_with_servable_config(1, servable_content)
    try:
        worker.start_servable_in_master(base.servable_dir, base.servable_name)
        assert False
    except AttributeError as e:
        assert "'numpy.ndarray' object has no attribute 'as_pair'" in str(e)
