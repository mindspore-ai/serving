/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "tests/ut/cpp/common/test_servable_common.h"

namespace mindspore {
namespace serving {
class TestStartWorker : public TestMasterWorker {
 public:
  TestStartWorker() = default;
  ~TestStartWorker() = default;
  virtual void SetUp() {}
  virtual void TearDown() { TestMasterWorker::TearDown(); }
};

TEST_F(TestStartWorker, test_worker_start_success) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  RegisterMethod("test_servable", "test_add.mindir", "add_common", {"x1", "x2"}, {"y"}, 2, 1);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_TRUE(status.IsSuccess());
}

TEST_F(TestStartWorker, test_worker_start_error_model_file_name) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add_error.mindir", "mindir", true);
  RegisterMethod("test_servable", "test_add.mindir", "add_common", {"x1", "x2"}, {"y"}, 2, 1);

  // start_servable
  auto status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), "Load model failed, servable directory: ");
}

TEST_F(TestStartWorker, test_worker_start_error_version_number) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  RegisterMethod("test_servable", "test_add.mindir", "add_common", {"x1", "x2"}, {"y"}, 2, 1);

  // start_servable
  int error_version_number = 2;
  auto status = StartServable("test_servable_dir", "test_servable", error_version_number);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(
    status.StatusMessage(),
    "Start servable failed, there is no specified version directory of models, specified version number: 2");
}

TEST_F(TestStartWorker, test_worker_start_multi_version_number) {
  auto servable_dir = std::string(test_info_->test_case_name()) + "_test_servable_dir";
  Init(servable_dir, "test_servable", 1, "test_add.mindir");
  Init(servable_dir, "test_servable", 2, "test_add.mindir");

  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  RegisterMethod("test_servable", "test_add.mindir", "add_common", {"x1", "x2"}, {"y"}, 2, 1);

  // start_servable
  int version_number = 2;
  Status status = StartServable(servable_dir, "test_servable", version_number);
  EXPECT_TRUE(status.IsSuccess());
}

TEST_F(TestStartWorker, test_worker_start_version_number_no_valid) {
  auto servable_dir = std::string(test_info_->test_case_name()) + "_test_servable_dir";

  Init(servable_dir, "test_servable", 0, "test_add.mindir");
  Init(servable_dir, "test_servable", -2, "test_add.mindir");

  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  RegisterMethod("test_servable", "test_add.mindir", "add_common", {"x1", "x2"}, {"y"}, 2, 1);

  // start_servable
  Status status = StartServable(servable_dir, "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(
    status.StatusMessage(),
    "Start servable failed, there is no specified version directory of models, specified version number: 1");
}

TEST_F(TestStartWorker, test_worker_start_error_servable_dir) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  RegisterMethod("test_servable", "test_add.mindir", "add_common", {"x1", "x2"}, {"y"}, 2, 1);

  // start_servable
  std::string error_servable_dir = "test_servable_dir_error";
  Status status = StartServable(error_servable_dir, "test_servable", 0);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(
    status.StatusMessage(),
    "Start servable failed, there is no specified version directory of models, specified version number: 0");
}

TEST_F(TestStartWorker, test_worker_start_error_servable_name) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  RegisterMethod("test_servable", "test_add.mindir", "add_common", {"x1", "x2"}, {"y"}, 2, 1);

  // start_servable
  std::string error_servable_name = "test_servable_error";
  Status status = StartServable("test_servable_dir", error_servable_name, 0);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), "'test_servable_error' has not been registered");
}

TEST_F(TestStartWorker, test_worker_start_error_servable_format) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "om", true);
  RegisterMethod("test_servable", "test_add.mindir", "add_common", {"x1", "x2"}, {"y"}, 2, 1);

  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), "Not support device type Ascend and model type OM. ");
}

TEST_F(TestStartWorker, test_worker_start_no_registered_method) {
  Init("test_servable_dir", "test_servable", 2, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  // no registered method
  // RegisterMethod("test_servable", "test_add.mindir", "add_common", {"x1", "x2"}, {"y"}, 2, 1);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 2);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), "There is no method registered for servable");
}

TEST_F(TestStartWorker, test_worker_start_no_declared_servable) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  // no declared method
  // DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  auto status = RegisterMethod("test_servable", "test_add.mindir", "add_common", {"x1", "x2"}, {"y"}, 2, 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), "RegisterInputOutputInfo failed, cannot find model test_add.mindir");
}

TEST_F(TestStartWorker, test_worker_start_multi_method) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  RegisterMethod("test_servable", "test_add.mindir", "add_common", {"x1", "x2"}, {"y"}, 2, 1);
  RegisterMethod("test_servable", "test_add.mindir", "add_common2", {"x1", "x2"}, {"y"}, 2, 1);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_TRUE(status.IsSuccess());
}

TEST_F(TestStartWorker, test_worker_start_method_servable_input_count_not_match) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  size_t servable_input_count = 1;
  RegisterMethod("test_servable", "test_add.mindir", "add_common", {"x1", "x2"}, {"y"}, servable_input_count, 1);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "The inputs count 1 in register_method not equal to the count 2 defined in model")
}

TEST_F(TestStartWorker, test_worker_start_method_servable_output_count_not_match) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  size_t servable_output_count = 2;
  RegisterMethod("test_servable", "test_add.mindir", "add_common", {"x1", "x2"}, {"y"}, 2, servable_output_count);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "The outputs count 2 in register_method not equal to the count 1 defined in model")
}

// Test data flow in input\preprocess\predict\postprocess
TEST_F(TestStartWorker, test_worker_start_preprocess_not_found) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature;
  method_signature.servable_name = "test_servable";
  method_signature.method_name = "add_common";
  method_signature.inputs = {"x1", "x2"};
  method_signature.outputs = {"y"};
  // preprocess
  try {
    method_signature.AddStageFunction("preprocess_fake_fun", {{0, 0}, {0, 0}});
    // method input 0 and input 1 as servable input
    method_signature.AddStageModel("test_add.mindir", {{1, 0}, {0, 1}}, 0, "");
    // servable output as method output
    method_signature.SetReturn({{2, 0}});
    ServableRegister::Instance().RegisterMethod(method_signature);
  } catch (std::runtime_error &ex) {
    ExpectContainMsg(ex.what(), "Function 'preprocess_fake_fun' is not defined")
  }
}

TEST_F(TestStartWorker, test_worker_start_with_preproces_and_postprocess_success) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature;
  method_signature.servable_name = "test_servable";
  method_signature.method_name = "add_cast";
  method_signature.inputs = {"x1", "x2"};
  method_signature.outputs = {"y"};
  // preprocess, stage 1, input is input data(stage index = 0) 0 and 1
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {0, 0}});
  // model, stage 2, input is stage 1 output data 0 and 1
  method_signature.AddStageModel("test_add.mindir", {{1, 0}, {1, 1}}, 0);
  // postprocess, stage 3, input is stage 2 output data 0 and 1
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{2, 0}});
  // method output, stage 3 output data 0
  method_signature.SetReturn({{3, 0}});
  ServableRegister::Instance().RegisterMethod(method_signature);

  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_TRUE(status.IsSuccess());
}

}  // namespace serving
}  // namespace mindspore
