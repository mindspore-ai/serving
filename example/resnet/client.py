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
"""Client for resnet50"""
import os
from mindspore_serving.client import Client


def read_images():
    """Read images for directory test_image"""
    images_buffer = []
    for path, _, file_list in os.walk("./test_image/"):
        for file_name in file_list:
            image_file = os.path.join(path, file_name)
            print(image_file)
            with open(image_file, "rb") as fp:
                images_buffer.append(fp.read())
    return images_buffer


def run_classify_top1():
    """Client for servable resnet50 and method classify_top1"""
    client = Client("localhost", 5500, "resnet50", "classify_top1")
    instances = []
    for image in read_images():
        instances.append({"image": image})
    result = client.infer(instances)
    print(result)


def run_classify_top1_v1():
    """Client for servable resnet50 and method classify_top1_v1"""
    client = Client("localhost", 5500, "resnet50", "classify_top1_v1")
    instances = []
    for image in read_images():
        instances.append({"image": image})
    result = client.infer(instances)
    print(result)


def run_classify_top5():
    """Client for servable resnet50 and method classify_top5"""
    client = Client("localhost", 5500, "resnet50", "classify_top5")
    instances = []
    for image in read_images():  # read multi image
        instances.append({"image": image})  # input `image`
    result = client.infer(instances)
    print(result)
    for result_item in result:  # result for every image
        label = result_item["label"]  # result `label`
        score = result_item["score"]  # result `score`
        print("label result", label)
        print("score result", score)


def run_restful_classify_top1():
    """RESTful Client for servable resnet50 and method classify_top1"""
    import base64
    import requests
    import json
    instances = []
    for image in read_images():
        base64_data = base64.b64encode(image).decode()
        instances.append({"image": {"b64": base64_data}})
    instances_map = {"instances": instances}
    post_payload = json.dumps(instances_map)
    ip = "localhost"
    restful_port = 1500
    servable_name = "resnet50"
    method_name = "classify_top1"
    result = requests.post(f"http://{ip}:{restful_port}/model/{servable_name}:{method_name}", data=post_payload)
    print(result.text)


if __name__ == '__main__':
    run_classify_top1()
    run_classify_top1_v1()
    run_classify_top5()
    run_restful_classify_top1()
