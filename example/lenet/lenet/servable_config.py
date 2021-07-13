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
"""Lenet config python file"""
from io import BytesIO
import numpy as np
from PIL import Image

from mindspore_serving.server import register


def preprocess_eager(image):
    """
    Define preprocess, input is image numpy, return preprocess result.
    Return type can be numpy, str, bytes, int, float, or bool.
    Use MindData Eager, this image processing can also use other image processing library, likes numpy, PIL or cv2 etc.
    """
    image = Image.open(BytesIO(image.tobytes())).convert('L').resize((32, 32), Image.ANTIALIAS)
    image = np.array(image, np.float32)
    image = image / 255.0
    return image


def postprocess_top1(score):
    """
    Define postprocess. This example has one input and one output.
    The input is the numpy tensor of the score, and the output is the label str of top one.
    """
    max_idx = np.argmax(score)
    return max_idx


lenet_model = register.declare_model(model_file="lenet.mindir", model_format="MindIR")


@register.register_method(output_names=["label"])
def classify_top1(image):
    """Define method `classify_top1` for servable `resnet50`.
     The input is `image` and the output is `lable`."""
    x = register.add_stage(preprocess_eager, image, outputs_count=1)
    x = register.add_stage(lenet_model, x, outputs_count=1)
    x = register.add_stage(postprocess_top1, x, outputs_count=1)
    return x
