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

from common import ServingTestBase, serving_test
from mindspore_serving import server

# test servable_config.py
servable_config_import = r"""
import numpy as np
from mindspore_serving.server import register
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
    server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))


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
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert "There is no model declared, you can use declare_model to declare models" in str(e)


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
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert "name 'add_trans_datatype' is not defined" in str(e)


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
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
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
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
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
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
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
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
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
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
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
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
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
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
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
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
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
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
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
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
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
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
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
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
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
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
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
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
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
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert "call_postprocess or call_postprocess_pipeline should be invoked after call_servable" in str(e)


@serving_test
def test_register_method_without_call_servable_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
register.declare_servable(servable_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def add_trans_datatype(x1, x2):
    return x1.astype(np.float32), x2.astype(np.float32)

def add_func(x1, x2):
    return x1+x2   

@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    x1, x2 = register.call_preprocess(add_trans_datatype, x1, x2)  # cast input to float32
    y = register.call_postprocess(add_func, x1, x2)
    return y
    """
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content)
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert "Not find the invoke of 'call_servable'" in str(e)


@serving_test
def test_register_method_invalid_call_servable():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

@register.register_method(output_names="y")
def predict(x1, x2):
    if True:
        y = register.call_servable(model, x1, x2)
        return y
    return x1
    """
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content)
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert "conditions and loops are not supported in register_method when the interface 'call_servable' is used," \
               " use 'add_stage' to replace 'call_servable'" in str(e)


@serving_test
def test_register_method_invalid_call_servable2():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)
model2 = register.declare_model(model_file="tensor_add2.mindir", model_format="MindIR", with_batch_dim=False)

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.call_servable(x1, x2)
    return y
    """
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content)
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert "There are more than one servable declared when the interface 'call_servable' is used, use 'add_stage'" \
               " to replace 'call_servable'" in str(e)


@serving_test
def test_register_method_invalid_call_preprocess():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def preprocess(x1, x2):
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    if True:
        y = register.call_preprocess(preprocess, x1, x2)
        return y
    return x1
    """
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content)
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert "conditions and loops are not supported in register_method when the interface 'call_preprocess'" \
               " is used, use 'add_stage' to replace 'call_preprocess'" in str(e)


@serving_test
def test_register_method_invalid_call_preprocess_pipeline():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def preprocess(x1, x2):
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    if True:
        y = register.call_preprocess_pipeline(preprocess, x1, x2)
        return y
    return x1
    """
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content)
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert "conditions and loops are not supported in register_method when the interface" \
               " 'call_preprocess_pipeline' is used, use 'add_stage' to replace 'call_preprocess_pipeline'" in str(e)


@serving_test
def test_register_method_invalid_call_postprocess():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def preprocess(x1, x2):
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    if True:
        y = register.call_postprocess(preprocess, x1, x2)
        return y
    return x1
    """
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content)
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert "conditions and loops are not supported in register_method when the interface 'call_postprocess'" \
               " is used, use 'add_stage' to replace 'call_postprocess'" in str(e)


@serving_test
def test_register_method_invalid_call_postprocess_pipeline():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def preprocess(x1, x2):
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    if True:
        y = register.call_postprocess_pipeline(preprocess, x1, x2)
        return y
    return x1
    """
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content)
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert "conditions and loops are not supported in register_method when the interface " \
               "'call_postprocess_pipeline' is used, use 'add_stage' to replace 'call_postprocess_pipeline'" in str(e)


@serving_test
def test_register_method_invalid_call_preprocess_with_condition():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def preprocess(x1, x2):
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    y = register.call_preprocess(preprocess, x1, x2)
    if True:
        y = register.call_postprocess(preprocess, x1, x2)
        return y
    return y
    """
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content)
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert "complex statements such as conditions and loops are not supported in register_method when the " \
               "interface 'call_preprocess' is used, use 'add_stage' to replace 'call_preprocess'" in str(e)


@serving_test
def test_register_method_invalid_call_preprocess_with_condition2():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def preprocess(x1, x2):
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    if True:
        y = register.call_postprocess(preprocess, x1, x2)
        return y
    y = register.call_preprocess(preprocess, x1, x2)
    return y
    """
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content)
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert "complex statements such as conditions and loops are not supported in register_method when the " \
               "interface 'call_preprocess' is used, use 'add_stage' to replace 'call_preprocess'" in str(e)


@serving_test
def test_register_method_mix_call_xxx_add_stage_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def preprocess(x1, x2):
    return y

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4):
    y = register.call_preprocess(preprocess, x1, x2)
    y = register.call_servable(y, x3)
    y = register.call_postprocess(preprocess, y, x4)
    y = register.add_stage(preprocess, y, x2, outputs_count=1)
    return y
    """
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content)
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert "complex statements such as conditions and loops are not supported in register_method when the" in str(e)


@serving_test
def test_register_method_mix_call_xxx_add_stage2_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def preprocess(x1, x2):
    return y

@register.register_method(output_names="y")
def predict(x1, x2, x3, x4):
    y = register.add_stage(preprocess, x1, x2, outputs_count=1)
    y = register.call_preprocess(preprocess, y, x2)
    y = register.call_servable(y, x3)
    y = register.call_postprocess(preprocess, y, x4)
    return y
    """
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content)
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert "complex statements such as conditions and loops are not supported in register_method when the" in str(e)


@serving_test
def test_register_method_mix_call_xxx_add_stage3_failed():
    servable_content = r"""
import numpy as np
from mindspore_serving.server import register
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

def preprocess(x1, x2):
    return y

@register.register_method(output_names="y")
def predict(x1, x2):
    if True:
        y = register.call_postprocess(preprocess, x1, x2)
        return y
    y = register.add_stage(preprocess, x1, x2, outputs_count=1)
    return y
    """
    base = ServingTestBase()
    base.init_servable_with_servable_config(1, servable_content)
    try:
        server.start_servables(server.ServableStartConfig(base.servable_dir, base.servable_name, device_ids=0))
        assert False
    except RuntimeError as e:
        assert "complex statements such as conditions and loops are not supported in register_method when the" in str(e)
