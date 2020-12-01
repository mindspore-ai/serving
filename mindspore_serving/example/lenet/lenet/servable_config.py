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

import numpy as np
from mindspore_serving.worker import register
import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms as P
import mindspore.dataset.vision.py_transforms as PV


# define preprocess pipeline, the function arg is multi instances, every instance is tuple of inputs
# this example has one input and one output
def lenet_preprocess(instances):
    def generator_func():
        for instance in instances:
            image = instance[0]
            yield (image,)

    mnist_ds = ds.GeneratorDataset(generator_func, ["image"], shuffle=False)
    resize_height, resize_width = 32, 32
    nml_mean = [0.1307]
    nml_std = [0.3081]
    mnist_ds = mnist_ds.map(operations=P.Compose([
        PV.Decode(),
        PV.Grayscale(1),
        PV.Resize(size=(resize_height, resize_width)),
        PV.ToTensor(),
        PV.Normalize(nml_mean, nml_std)
    ]), input_columns="image", num_parallel_workers=1)
    for data in mnist_ds.create_dict_iterator():
        image_result = data["image"]
        yield (image_result,)


# define postprocess pipeline, the function arg is multi instances, every instance is tuple of inputs
# this example has one input and one output
def lenet_postprocess(instances):
    for instance in instances:
        result = instance[0]
        yield (np.argmax(result),)


register.declare_servable(servable_file="checkpoint_lenet_2-9_1875.om", model_format="OM")


# register predict method in lenet
@register.register_method(output_names=["result"])
def predict(image):
    x = register.call_preprocess(lenet_preprocess, image)
    x = register.call_servable(x)
    x = register.call_postprocess(lenet_postprocess, x)
    return x
