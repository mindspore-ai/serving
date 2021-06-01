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
"""Resnet50 ImageNet config python file"""
import os
import ast
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as TC
import mindspore.dataset.vision.c_transforms as VC

from mindspore_serving.server import register

cur_dir = os.path.abspath(os.path.dirname(__file__))
print("current dir:", cur_dir)
with open(os.path.join(cur_dir, "imagenet1000_clsidx_to_labels.txt"), "r") as fp:
    idx_2_label = ast.literal_eval(fp.read())
idx_2_label[1000] = "empty"


def preprocess_eager(image):
    """
    Define preprocess, input is image numpy, return preprocess result.
    Return type can be numpy, str, bytes, int, float, or bool.
    Use MindData Eager, this image processing can also use other image processing library, likes numpy, PIL or cv2 etc.
    """
    image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    decode = VC.Decode()
    resize = VC.Resize([image_size, image_size])
    normalize = VC.Normalize(mean=mean, std=std)
    hwc2chw = VC.HWC2CHW()

    image = decode(image)
    image = resize(image)
    image = normalize(image)
    image = hwc2chw(image)
    return image


def preprocess_pipeline(instances):
    """
    Define preprocess pipeline, the function arg is multi instances, every instance is tuple of inputs.
    This example has one input and one output.
    Use MindData Pipeline.
    """

    def generator_func():
        for instance in instances:
            image = instance[0]
            yield (image,)

    resnet_ds = ds.GeneratorDataset(generator_func, ["image"], shuffle=False)
    image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    resnet_ds = resnet_ds.map(operations=VC.Decode(), input_columns="image", num_parallel_workers=8)

    trans = [
        VC.Resize([image_size, image_size]),
        VC.Normalize(mean=mean, std=std),
        VC.HWC2CHW()
    ]
    resnet_ds = resnet_ds.map(operations=TC.Compose(trans), input_columns="image", num_parallel_workers=2)

    for data in resnet_ds.create_dict_iterator():
        image_result = data["image"]
        yield (image_result,)


def postprocess_top1(score):
    """
    Define postprocess. This example has one input and one output.
    The input is the numpy tensor of the score, and the output is the label str of top one.
    """
    max_idx = np.argmax(score)
    return idx_2_label[max_idx]


def postprocess_top5(score):
    """
    Define postprocess. This example has one input and two outputs.
    The input is the numpy tensor of the score. The first output is the str joined by labels of top five,
    and the second output is the score tensor of the top five.
    """
    idx = np.argsort(score)[::-1][:5]  # top 5
    ret_label = [idx_2_label[i] for i in idx]
    ret_score = score[idx]
    return ";".join(ret_label), ret_score


register.declare_servable(servable_file="resnet50_1b_imagenet.mindir", model_format="MindIR")


@register.register_method(output_names=["label"])
def classify_top1(image):
    """Define method `classify_top1` for servable `resnet50`.
     The input is `image` and the output is `lable`."""
    x = register.call_preprocess_pipeline(preprocess_pipeline, image)
    x = register.call_servable(x)
    x = register.call_postprocess(postprocess_top1, x)
    return x


@register.register_method(output_names=["label"])
def classify_top1_v1(image):
    """Define method `classify_top1_v1` for servable `resnet50`.
     The input is `image` and the output is `label`. """
    x = register.call_preprocess(preprocess_eager, image)
    x = register.call_servable(x)
    x = register.call_postprocess(postprocess_top1, x)
    return x


@register.register_method(output_names=["label", "score"])
def classify_top5(image):
    """Define method `classify_top5` for servable `resnet50`.
     The input is `image` and the output is `label` and `score`. """
    x = register.call_preprocess_pipeline(preprocess_pipeline, image)
    x = register.call_servable(x)
    label, score = register.call_postprocess(postprocess_top5, x)
    return label, score
