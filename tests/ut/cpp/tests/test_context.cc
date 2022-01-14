/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "../common/common_test.h"
#define private public
#include "worker/inference/inference.h"
#include "worker/inference/mindspore_model_wrap.h"
#undef private

using std::string;
using std::vector;
namespace mindspore {
namespace serving {
class TestModelContext : public UT::Common {
 public:
  TestModelContext() = default;
  void Init(std::string file_name) {
    char *dir;
    dir = get_current_dir_name();
    std::string file_path(dir);
    file_path += file_name;
    std::ofstream fp(file_path);
    fp << "model content";
    fp.close();
    model_file = file_path;
    free(dir);
  }
  virtual void SetUp() {
    setenv("SERVING_ENABLE_CPU_DEVICE", "1", 1);
    setenv("SERVING_ENABLE_GPU_DEVICE", "1", 1);
  }
  virtual void TearDown() {
    remove(model_file.c_str());
    setenv("SERVING_ENABLE_CPU_DEVICE", "0", 1);
    setenv("SERVING_ENABLE_GPU_DEVICE", "0", 1);
  }
  std::string model_file;
};

/// Feature: model context
/// Description: ascend910 device with mindspore
/// Expectation: the context has ascend910 and load success
TEST_F(TestModelContext, test_ms_set_ascend910) {
  setenv("SERVING_ENABLE_CPU_DEVICE", "0", 1);
  setenv("SERVING_ENABLE_GPU_DEVICE", "0", 1);

  Init("tensor_add.mindir@ms_ascend");
  ModelContext model_context;
  auto mindspore_wrap = InferenceLoader::Instance().CreateMindSporeInfer();
  auto status = mindspore_wrap->LoadModelFromFile(serving::DeviceType::kDeviceTypeAscend, 0, {model_file},
                                                  serving::kMindIR, false, {}, model_context, {}, {}, {}, false);
  ASSERT_TRUE(status.IsSuccess());
}

/// Feature: model context
/// Description: gpu device with lite
/// Expectation: the context has gpu and load success
TEST_F(TestModelContext, test_lite_set_gpu) {
  Init("tensor_add.mindir@lite_gpu_cpu");
  ModelContext model_context;
  auto mindspore_wrap = InferenceLoader::Instance().CreateMindSporeInfer();
  auto status = mindspore_wrap->LoadModelFromFile(serving::DeviceType::kDeviceTypeGpu, 0, {model_file},
                                                  serving::kMindIR, false, {}, model_context, {}, {}, {}, true);
  ASSERT_TRUE(status.IsSuccess());
}

/// Feature: Model context
/// Description: gpu cpu device with lite
/// Expectation: the context has gpu and cpu and load success
TEST_F(TestModelContext, test_lite_set_gpu_cpu) {
  Init("tensor_add.mindir@lite_gpu_cpu");
  ModelContext model_context;
  DeviceInfo cpu_device_info{{"device_type", "cpu"}};
  model_context.device_list.push_back(cpu_device_info);
  auto mindspore_wrap = InferenceLoader::Instance().CreateMindSporeInfer();
  auto status = mindspore_wrap->LoadModelFromFile(serving::DeviceType::kDeviceTypeGpu, 0, {model_file},
                                                  serving::kMindIR, false, {}, model_context, {}, {}, {}, true);
  ASSERT_TRUE(status.IsSuccess());
}

/// Feature: Model context
/// Description: gpu cpu device with mindspore
/// Expectation: the context only has gpu and load success
TEST_F(TestModelContext, test_ms_set_gpu) {
  Init("tensor_add.mindir@ms_gpu");
  ModelContext model_context;
  DeviceInfo cpu_device_info{{"device_type", "cpu"}};
  model_context.device_list.push_back(cpu_device_info);
  auto mindspore_wrap = InferenceLoader::Instance().CreateMindSporeInfer();
  auto status = mindspore_wrap->LoadModelFromFile(serving::DeviceType::kDeviceTypeGpu, 0, {model_file},
                                                  serving::kMindIR, false, {}, model_context, {}, {}, {}, false);
  ASSERT_TRUE(status.IsSuccess());
}

}  // namespace serving
}  // namespace mindspore
