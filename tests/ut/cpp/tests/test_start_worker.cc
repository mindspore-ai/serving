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
  RegisterMethod("test_servable", "add_common", {"x1", "x2"}, {"y"}, 2, 1);
  // start_servable
  auto notify_master = std::make_shared<LocalNotifyMaster>();
  ServableContext::Instance()->SetDeviceId(0);
  ServableContext::Instance()->SetDeviceTypeStr("Ascend");
  Status status = Worker::GetInstance().StartServable("test_servable_dir", "test_servable", 0, notify_master);
  EXPECT_TRUE(status.IsSuccess());
}

TEST_F(TestStartWorker, test_worker_start_error_model_file_name) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add_error.mindir", "mindir", true);
  RegisterMethod("test_servable", "add_common", {"x1", "x2"}, {"y"}, 2, 1);

  // start_servable
  auto notify_master = std::make_shared<LocalNotifyMaster>();
  ServableContext::Instance()->SetDeviceId(0);
  ServableContext::Instance()->SetDeviceTypeStr("Ascend");
  Status status = Worker::GetInstance().StartServable("test_servable_dir", "test_servable", 0, notify_master);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), "Load model failed, servable directory: ");
}

TEST_F(TestStartWorker, test_worker_start_error_version_number) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  RegisterMethod("test_servable", "add_common", {"x1", "x2"}, {"y"}, 2, 1);

  // start_servable
  auto notify_master = std::make_shared<LocalNotifyMaster>();
  ServableContext::Instance()->SetDeviceId(0);
  ServableContext::Instance()->SetDeviceTypeStr("Ascend");
  int error_version_number = 2;
  Status status =
    Worker::GetInstance().StartServable("test_servable_dir", "test_servable", error_version_number, notify_master);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "Start servable failed, there is no servable of"
                   " the specified version number, specified version number: ");
}

TEST_F(TestStartWorker, test_worker_start_multi_version_number) {
  auto servable_dir = std::string(test_info_->test_case_name()) + "_test_servable_dir";
  Init(servable_dir, "test_servable", 1, "test_add.mindir");
  Init(servable_dir, "test_servable", 2, "test_add.mindir");

  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  RegisterMethod("test_servable", "add_common", {"x1", "x2"}, {"y"}, 2, 1);

  // start_servable
  auto notify_master = std::make_shared<LocalNotifyMaster>();
  ServableContext::Instance()->SetDeviceId(0);
  ServableContext::Instance()->SetDeviceTypeStr("Ascend");
  int version_number = 0;
  Status status = Worker::GetInstance().StartServable(servable_dir, "test_servable", version_number, notify_master);
  EXPECT_TRUE(status.IsSuccess());
}

TEST_F(TestStartWorker, test_worker_start_version_number_no_valid) {
  auto servable_dir = std::string(test_info_->test_case_name()) + "_test_servable_dir";

  Init(servable_dir, "test_servable", 0, "test_add.mindir");
  Init(servable_dir, "test_servable", -2, "test_add.mindir");

  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  RegisterMethod("test_servable", "add_common", {"x1", "x2"}, {"y"}, 2, 1);

  // start_servable
  auto notify_master = std::make_shared<LocalNotifyMaster>();
  ServableContext::Instance()->SetDeviceId(0);
  ServableContext::Instance()->SetDeviceTypeStr("Ascend");
  Status status = Worker::GetInstance().StartServable(servable_dir, "test_servable", 0, notify_master);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "Start servable failed, there is no servable of"
                   " the specified version number, specified version number: ");
}

TEST_F(TestStartWorker, test_worker_start_error_servable_dir) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  RegisterMethod("test_servable", "add_common", {"x1", "x2"}, {"y"}, 2, 1);

  // start_servable
  auto notify_master = std::make_shared<LocalNotifyMaster>();
  ServableContext::Instance()->SetDeviceId(0);
  ServableContext::Instance()->SetDeviceTypeStr("Ascend");
  std::string error_servable_dir = "test_servable_dir_error";
  Status status = Worker::GetInstance().StartServable(error_servable_dir, "test_servable", 0, notify_master);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "Start servable failed, there is no servable of"
                   " the specified version number, specified version number: ");
}

TEST_F(TestStartWorker, test_worker_start_error_servable_name) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  RegisterMethod("test_servable", "add_common", {"x1", "x2"}, {"y"}, 2, 1);

  // start_servable
  auto notify_master = std::make_shared<LocalNotifyMaster>();
  ServableContext::Instance()->SetDeviceId(0);
  ServableContext::Instance()->SetDeviceTypeStr("Ascend");
  std::string error_servable_name = "test_servable_error";
  Status status = Worker::GetInstance().StartServable("test_servable_dir", error_servable_name, 0, notify_master);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), "'test_servable_error' has not been registered");
}

TEST_F(TestStartWorker, test_worker_start_error_servable_format) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "om", true);
  RegisterMethod("test_servable", "add_common", {"x1", "x2"}, {"y"}, 2, 1);

  // start_servable
  auto notify_master = std::make_shared<LocalNotifyMaster>();
  ServableContext::Instance()->SetDeviceId(0);
  ServableContext::Instance()->SetDeviceTypeStr("Ascend");
  Status status = Worker::GetInstance().StartServable("test_servable_dir", "test_servable", 0, notify_master);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), "Cannot find session registered for device type Ascend and model type OM");
}

TEST_F(TestStartWorker, test_worker_start_no_registered_method) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  // no registered method
  // RegisterMethod("test_servable", "add_common", {"x1", "x2"}, {"y"}, 2, 1);
  // start_servable
  auto notify_master = std::make_shared<LocalNotifyMaster>();
  ServableContext::Instance()->SetDeviceId(0);
  ServableContext::Instance()->SetDeviceTypeStr("Ascend");
  Status status = Worker::GetInstance().StartServable("test_servable_dir", "test_servable", 0, notify_master);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), "There is no method registered for servable");
}

TEST_F(TestStartWorker, test_worker_start_no_declared_servable) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  // no declared method
  // DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  auto status = RegisterMethod("test_servable", "add_common", {"x1", "x2"}, {"y"}, 2, 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), "RegisterInputOutputInfo failed, cannot find servable");
}

TEST_F(TestStartWorker, test_worker_start_multi_method) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  RegisterMethod("test_servable", "add_common", {"x1", "x2"}, {"y"}, 2, 1);
  RegisterMethod("test_servable", "add_common2", {"x1", "x2"}, {"y"}, 2, 1);
  // start_servable
  auto notify_master = std::make_shared<LocalNotifyMaster>();
  ServableContext::Instance()->SetDeviceId(0);
  ServableContext::Instance()->SetDeviceTypeStr("Ascend");
  Status status = Worker::GetInstance().StartServable("test_servable_dir", "test_servable", 0, notify_master);
  EXPECT_TRUE(status.IsSuccess());
}

TEST_F(TestStartWorker, test_worker_start_method_servable_input_count_not_match) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  size_t servable_input_count = 1;
  RegisterMethod("test_servable", "add_common", {"x1", "x2"}, {"y"}, servable_input_count, 1);
  // start_servable
  auto notify_master = std::make_shared<LocalNotifyMaster>();
  ServableContext::Instance()->SetDeviceId(0);
  ServableContext::Instance()->SetDeviceTypeStr("Ascend");
  Status status = Worker::GetInstance().StartServable("test_servable_dir", "test_servable", 0, notify_master);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "The inputs count 1 registered in method not equal to "
                   "the count 2 defined in servable")
}

TEST_F(TestStartWorker, test_worker_start_method_servable_output_count_not_match) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  size_t servable_output_count = 2;
  RegisterMethod("test_servable", "add_common", {"x1", "x2"}, {"y"}, 2, servable_output_count);
  // start_servable
  auto notify_master = std::make_shared<LocalNotifyMaster>();
  ServableContext::Instance()->SetDeviceId(0);
  ServableContext::Instance()->SetDeviceTypeStr("Ascend");
  Status status = Worker::GetInstance().StartServable("test_servable_dir", "test_servable", 0, notify_master);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "The outputs count 2 registered in method not equal to "
                   "the count 1 defined in servable")
}

// Test data flow in input\preprocess\predict\postprocess
TEST_F(TestStartWorker, test_worker_start_preprocess_not_found) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature;
  method_signature.servable_name = "test_servable";
  method_signature.method_name = "add_common";
  method_signature.inputs = {"x1", "x2"};
  method_signature.outputs = {"y"};
  // preprocess
  method_signature.preprocess_name = "preprocess_fake_fun";
  method_signature.preprocess_inputs = {{kPredictPhaseTag_Input, 0}, {kPredictPhaseTag_Input, 0}};
  // method input 0 and input 1 as servable input
  method_signature.servable_inputs = {{kPredictPhaseTag_Preproces, 0}, {kPredictPhaseTag_Input, 1}};
  // servable output as method output
  method_signature.returns = {{kPredictPhaseTag_Predict, 0}};
  ServableStorage::Instance().RegisterMethod(method_signature);

  // start_servable
  auto notify_master = std::make_shared<LocalNotifyMaster>();
  ServableContext::Instance()->SetDeviceId(0);
  ServableContext::Instance()->SetDeviceTypeStr("Ascend");
  Status status = Worker::GetInstance().StartServable("test_servable_dir", "test_servable", 0, notify_master);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), " preprocess preprocess_fake_fun not defined")
}

TEST_F(TestStartWorker, test_worker_start_postprocess_not_found) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature;
  method_signature.servable_name = "test_servable";
  method_signature.method_name = "add_common";
  method_signature.inputs = {"x1", "x2"};
  method_signature.outputs = {"y"};
  // preprocess
  method_signature.postprocess_name = "postprocess_fake_fun";
  method_signature.postprocess_inputs = {{kPredictPhaseTag_Input, 0}, {kPredictPhaseTag_Input, 0}};
  // method input 0 and input 1 as servable input
  method_signature.servable_inputs = {{kPredictPhaseTag_Input, 0}, {kPredictPhaseTag_Input, 1}};
  // servable output as method output
  method_signature.returns = {{kPredictPhaseTag_Predict, 0}};
  ServableStorage::Instance().RegisterMethod(method_signature);

  // start_servable
  auto notify_master = std::make_shared<LocalNotifyMaster>();
  ServableContext::Instance()->SetDeviceId(0);
  ServableContext::Instance()->SetDeviceTypeStr("Ascend");
  Status status = Worker::GetInstance().StartServable("test_servable_dir", "test_servable", 0, notify_master);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), " postprocess postprocess_fake_fun not defined")
}

TEST_F(TestStartWorker, test_worker_start_with_preproces_and_postprocess_success) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature;
  method_signature.servable_name = "test_servable";
  method_signature.method_name = "add_cast";
  method_signature.inputs = {"x1", "x2"};
  method_signature.outputs = {"y"};
  // preprocess
  method_signature.preprocess_name = "stub_preprocess_cast_int32_to_fp32_cpp";
  method_signature.preprocess_inputs = {{kPredictPhaseTag_Input, 0}, {kPredictPhaseTag_Input, 1}};
  // method input 0 and input 1 as servable input
  method_signature.servable_inputs = {{kPredictPhaseTag_Preproces, 0}, {kPredictPhaseTag_Preproces, 1}};
  // postprocess
  method_signature.postprocess_name = "stub_postprocess_cast_fp32_to_int32_cpp";
  method_signature.postprocess_inputs = {{kPredictPhaseTag_Predict, 0}};
  // servable output as method output
  method_signature.returns = {{kPredictPhaseTag_Postprocess, 0}};
  ServableStorage::Instance().RegisterMethod(method_signature);

  // start_servable
  auto notify_master = std::make_shared<LocalNotifyMaster>();
  ServableContext::Instance()->SetDeviceId(0);
  ServableContext::Instance()->SetDeviceTypeStr("Ascend");
  Status status = Worker::GetInstance().StartServable("test_servable_dir", "test_servable", 0, notify_master);
  EXPECT_TRUE(status.IsSuccess());
}

}  // namespace serving
}  // namespace mindspore
