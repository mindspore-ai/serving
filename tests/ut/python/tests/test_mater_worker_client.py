import os
from functools import wraps
from shutil import copyfile, rmtree
import numpy as np
from mindspore_serving import master
from mindspore_serving import worker
from mindspore_serving.client import Client


class ServingTestBase:
    def __init__(self):
        self.servable_dir_list = []
        self.servable_name_path_list = []
        self.servable_config_path_list = []
        self.version_number_path_list = []
        self.model_file_name_path_list = []

    def __del__(self):
        rmtree(self.servable_dir, True)
        print("end remove servable directory and model file")

    def init_servable(self, servable_name, version_number, model_file_name, config_file):
        servable_dir = "serving_python_ut_servables"
        self.servable_dir = os.path.join(os.getcwd(), servable_dir)
        rmtree(self.servable_dir, True)

        self.servable_name = servable_name
        self.version_number = version_number
        self.model_file_name = model_file_name
        self.servable_name_path = os.path.join(self.servable_dir, servable_name)
        self.version_number_path = os.path.join(self.servable_name_path, str(version_number))
        self.model_file_name_path = os.path.join(self.version_number_path, model_file_name)
        print("model file name:", self.model_file_name_path)

        try:
            os.mkdir(self.servable_dir)
        except FileExistsError:
            pass
        try:
            os.mkdir(self.servable_name_path)
        except FileExistsError:
            pass
        try:
            os.mkdir(self.version_number_path)
        except FileExistsError:
            pass
        with open(self.model_file_name_path, "w") as fp:
            print("model content", file=fp)

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_abs = os.path.join(os.path.join(cur_dir, "../servable_config/"), config_file)
        dst_file = os.path.join(self.servable_name_path, "servable_config.py")
        copyfile(config_file_abs, dst_file)

        self.servable_dir_list.append(self.servable_dir)
        self.servable_name_path_list.append(self.servable_name_path)
        self.servable_config_path_list.append(dst_file)
        self.version_number_path_list.append(self.version_number_path)
        self.model_file_name_path_list.append(self.model_file_name_path)


def create_multi_instances_fp32(instance_count):
    instances = []
    # instance 1
    y_data_list = []
    for i in range(instance_count):
        x1 = np.asarray([[1.1, 2.2], [3.3, 4.4]]).astype(np.float32) * (i + 1)
        x2 = np.asarray([[5.5, 6.6], [7.7, 8.8]]).astype(np.float32) * (i + 1)
        y_data_list.append(x1+x2)
        instances.append({"x1": x1, "x2": x2})
    return instances, y_data_list


def check_result(result, y_data_list):
    assert len(result) == len(y_data_list)
    for result_item, y_data in zip(result, y_data_list):
        assert (result_item["y"] == y_data).all()


def serving_test(func):
    @wraps(func)
    def wrap_test(*args, **kwargs):
        try:
            func(*args, **kwargs)
        finally:
            worker.stop()
    return wrap_test


@serving_test
def test_master_worker_client_success():
    base = ServingTestBase()
    base.init_servable("add", 1, "tensor_add.mindir", "add_servable_config.py")
    worker.start_servable_in_master(base.servable_dir, base.servable_name, 0)
    master.start_grpc_server("0.0.0.0", 5500)
    # Client
    client = Client("localhost", 5500, "add", "add_common")
    instance_count = 3
    instances, y_data_list = create_multi_instances_fp32(instance_count)
    result = client.infer(instances)

    print(result)
    check_result(result, y_data_list)


@serving_test
def test_master_worker_client_servable_dir_invalid_failed():
    base = ServingTestBase()
    base.init_servable("add", 1, "tensor_add.mindir", "add_servable_config.py")
    try:
        worker.start_servable_in_master(base.servable_dir+"_error", base.servable_name, 0)
        assert False
    except RuntimeError as e:
        assert "Load servable config failed, directory " in str(e)
