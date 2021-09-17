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
"""Resnet50 cifar10 config python file"""
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as TC
import mindspore.dataset.vision.c_transforms as VC

from mindspore_serving.server import register

# cifar 10
idx_2_label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def preprocess_eager(image):
    """
    Define preprocess, input is image numpy, return preprocess result.
    Return type can be numpy, str, bytes, int, float, or bool.
    Use MindData Eager, this image processing can also use other image processing library, likes numpy, PIL or cv2 etc.
    """
    image_size = 224
    mean = [0.4914 * 255, 0.4822 * 255, 0.4465 * 255]
    std = [0.2023 * 255, 0.1994 * 255, 0.2010 * 255]

    decode = VC.Decode()
    resize = VC.Resize([image_size, image_size])
    normalize = VC.Normalize(mean=mean, std=std)
    hwc2chw = VC.HWC2CHW()

    image = decode(image)
    image = resize(image)
    image = normalize(image)
    image = hwc2chw(image)
    return image


def preprocess_batch(instances):
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
    mean = [0.4914 * 255, 0.4822 * 255, 0.4465 * 255]
    std = [0.2023 * 255, 0.1994 * 255, 0.2010 * 255]
    resnet_ds = resnet_ds.map(operations=VC.Decode(), input_columns="image", num_parallel_workers=8)

    trans = [
        VC.Resize([image_size, image_size]),
        VC.Normalize(mean=mean, std=std),
        VC.HWC2CHW()
    ]
    resnet_ds = resnet_ds.map(operations=TC.Compose(trans), input_columns="image", num_parallel_workers=2)

    for data in resnet_ds.create_dict_iterator(num_epochs=1):
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


resnet_model = register.declare_model(model_file="resnet50_1b_cifar10.mindir", model_format="MindIR")


def call_resnet_model(image):
    """call model with only one instance a time"""
    image = preprocess_eager(image)
    score = resnet_model.call(image)  # for only one instance
    return postprocess_top1(score)


def call_resnet_model_batch(instances):
    """call model with multiply instances a time"""
    input_instances = []
    for instance in instances:
        image = instance[0] # only one input
        image = preprocess_eager(image) # [3,224,224]
        input_instances.append([image])
    output_instances = resnet_model.call(input_instances)  # for multiply instances
    for instance in output_instances:
        output = instance[0]  # only one output for each instance
        output = postprocess_top1(output)
        yield output


@register.register_method(output_names=["label"])
def classify_top1_batch(image):
    """Define method `classify_top1` for servable `resnet50`.
     The input is `image` and the output is `lable`."""
    x = register.add_stage(preprocess_batch, image, outputs_count=1, batch_size=1024)
    x = register.add_stage(resnet_model, x, outputs_count=1)
    x = register.add_stage(postprocess_top1, x, outputs_count=1)
    return x


@register.register_method(output_names=["label"])
def classify_top1(image):  # pipeline: preprocess_eager/postprocess_top1, model
    """Define method `classify_top1` for servable `resnet50`.
     The input is `image` and the output is `label`. """
    x = register.add_stage(preprocess_eager, image, outputs_count=1)
    x = register.add_stage(resnet_model, x, outputs_count=1)
    x = register.add_stage(postprocess_top1, x, outputs_count=1)
    return x


@register.register_method(output_names=["label"])
def classify_top1_v2(image):  # without pipeline, call model with only one instance a time
    """Define method `classify_top1_v2` for servable `resnet50`.
     The input is `image` and the output is `label`. """
    label = register.add_stage(call_resnet_model, image, outputs_count=1)
    return label


@register.register_method(output_names=["label"])
def classify_top1_v3(image):  # without pipeline, call model with maximum 32 instances a time
    """Define method `classify_top1_v2` for servable `resnet50`.
     The input is `image` and the output is `label`. """
    label = register.add_stage(call_resnet_model_batch, image, outputs_count=1, batch_size=32)
    return label


@register.register_method(output_names=["label", "score"])
def classify_top5(image):
    """Define method `classify_top5` for servable `resnet50`.
     The input is `image` and the output is `label` and `score`. """
    x = register.add_stage(preprocess_eager, image, outputs_count=1)
    x = register.add_stage(resnet_model, x, outputs_count=1)
    label, score = register.add_stage(postprocess_top5, x, outputs_count=2)
    return label, score
