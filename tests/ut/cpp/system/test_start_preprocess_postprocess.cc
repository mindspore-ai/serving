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
#include "system/test_servable_common.h"

namespace mindspore {
namespace serving {
class TestPreprocessPostprocess : public TestMasterWorkerClient {
 public:
  TestPreprocessPostprocess() = default;
  ~TestPreprocessPostprocess() = default;
  virtual void SetUp() {}
  virtual void TearDown() { TestMasterWorkerClient::TearDown(); }
  MethodSignature InitDefaultMethod() {
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
    return method_signature;
  }
};

TEST_F(TestPreprocessPostprocess, test_master_worker_with_preproces_and_postprocess_success) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  // declare_servable
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  // register method
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  ServableStorage::Instance().RegisterMethod(method_signature);

  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_TRUE(status.IsSuccess());

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  auto request_servable_spec = request.mutable_servable_spec();
  request_servable_spec->set_name(servable_name_);
  request_servable_spec->set_method_name("add_cast");
  request_servable_spec->set_version_number(0);

  std::vector<std::vector<int32_t>> y_data_list;
  for (size_t k = 0; k < instances_count; k++) {
    std::vector<int32_t> x1_data = {1, 2, 3, 4};
    std::vector<int32_t> x2_data = {2, 3, 4, 5};
    std::vector<int32_t> y_data;
    for (size_t i = 0; i < x1_data.size(); i++) {
      x1_data[i] *= (k + 1);
      x2_data[i] *= (k + 1);
      y_data.push_back(x1_data[i] + x2_data[i]);
    }
    y_data_list.push_back(y_data);

    auto instance = request.add_instances();
    auto &input_map = (*instance->mutable_items());
    // input x1
    InitTensor(&input_map["x1"], {2, 2}, proto::MS_INT32, x1_data.data(), x1_data.size() * sizeof(int32_t));
    // input x2
    InitTensor(&input_map["x2"], {2, 2}, proto::MS_INT32, x2_data.data(), x2_data.size() * sizeof(int32_t));
  }

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.instances_size(), instances_count);
  ASSERT_EQ(reply.error_msg_size(), 0);
  for (size_t k = 0; k < instances_count; k++) {
    auto &output_instance = reply.instances(k);
    ASSERT_EQ(output_instance.items_size(), 1);
    auto &output_items = output_instance.items();
    ASSERT_EQ(output_items.begin()->first, "y");
    auto &output_tensor = output_items.begin()->second;

    CheckTensor(output_tensor, {2, 2}, proto::MS_INT32, y_data_list[k].data(), y_data_list[k].size() * sizeof(int32_t));
  }
}

TEST_F(TestPreprocessPostprocess, test_master_worker_with_only_preproces_success) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  // declare_servable
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  // register method
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);
  MethodSignature method_signature = InitDefaultMethod();
  method_signature.postprocess_name.clear();
  method_signature.postprocess_inputs.clear();
  method_signature.returns = {{kPredictPhaseTag_Predict, 0}};
  ServableStorage::Instance().RegisterMethod(method_signature);

  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_TRUE(status.IsSuccess());

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  // invalid version_number
  auto request_servable_spec = request.mutable_servable_spec();
  request_servable_spec->set_name(servable_name_);
  request_servable_spec->set_method_name("add_cast");
  request_servable_spec->set_version_number(0);

  std::vector<std::vector<float>> y_data_list;
  for (size_t k = 0; k < instances_count; k++) {
    std::vector<int32_t> x1_data = {1, 2, 3, 4};
    std::vector<int32_t> x2_data = {2, 3, 4, 5};
    std::vector<float> y_data;
    for (size_t i = 0; i < x1_data.size(); i++) {
      x1_data[i] *= (k + 1);
      x2_data[i] *= (k + 1);
      y_data.push_back(x1_data[i] + x2_data[i]);
    }
    y_data_list.push_back(y_data);

    auto instance = request.add_instances();
    auto &input_map = (*instance->mutable_items());
    // input x1
    InitTensor(&input_map["x1"], {2, 2}, proto::MS_INT32, x1_data.data(), x1_data.size() * sizeof(int32_t));
    // input x2
    InitTensor(&input_map["x2"], {2, 2}, proto::MS_INT32, x2_data.data(), x2_data.size() * sizeof(int32_t));
  }

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.instances_size(), instances_count);
  ASSERT_EQ(reply.error_msg_size(), 0);
  for (size_t k = 0; k < instances_count; k++) {
    auto &output_instance = reply.instances(k);
    ASSERT_EQ(output_instance.items_size(), 1);
    auto &output_items = output_instance.items();
    ASSERT_EQ(output_items.begin()->first, "y");
    auto &output_tensor = output_items.begin()->second;

    CheckTensor(output_tensor, {2, 2}, proto::MS_FLOAT32, y_data_list[k].data(), y_data_list[k].size() * sizeof(float));
  }
}

TEST_F(TestPreprocessPostprocess, test_master_worker_with_only_postprocess_success) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  // declare_servable
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  // register method
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);
  MethodSignature method_signature = InitDefaultMethod();
  method_signature.preprocess_name.clear();
  method_signature.preprocess_inputs.clear();
  method_signature.servable_inputs = {{kPredictPhaseTag_Input, 0}, {kPredictPhaseTag_Input, 1}};
  ServableStorage::Instance().RegisterMethod(method_signature);

  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_TRUE(status.IsSuccess());

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  // invalid version_number
  auto request_servable_spec = request.mutable_servable_spec();
  request_servable_spec->set_name(servable_name_);
  request_servable_spec->set_method_name("add_cast");
  request_servable_spec->set_version_number(0);

  std::vector<std::vector<int32_t>> y_data_list;
  for (size_t k = 0; k < instances_count; k++) {
    std::vector<float> x1_data = {1.1, 2.1, 3.1, 4.1};
    std::vector<float> x2_data = {2.1, 3.2, 4.3, 5.4};
    std::vector<int32_t> y_data;
    for (size_t i = 0; i < x1_data.size(); i++) {
      x1_data[i] *= (k + 1);
      x2_data[i] *= (k + 1);
      y_data.push_back(static_cast<int32_t>(x1_data[i] + x2_data[i]));
    }
    y_data_list.push_back(y_data);

    auto instance = request.add_instances();
    auto &input_map = (*instance->mutable_items());
    // input x1
    InitTensor(&input_map["x1"], {2, 2}, proto::MS_FLOAT32, x1_data.data(), x1_data.size() * sizeof(int32_t));
    // input x2
    InitTensor(&input_map["x2"], {2, 2}, proto::MS_FLOAT32, x2_data.data(), x2_data.size() * sizeof(int32_t));
  }

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.instances_size(), instances_count);
  ASSERT_EQ(reply.error_msg_size(), 0);
  for (size_t k = 0; k < instances_count; k++) {
    auto &output_instance = reply.instances(k);
    ASSERT_EQ(output_instance.items_size(), 1);
    auto &output_items = output_instance.items();
    ASSERT_EQ(output_items.begin()->first, "y");
    auto &output_tensor = output_items.begin()->second;

    CheckTensor(output_tensor, {2, 2}, proto::MS_INT32, y_data_list[k].data(), y_data_list[k].size() * sizeof(int32_t));
  }
}

// Test data flow in input\preprocess\predict\postprocess
TEST_F(TestPreprocessPostprocess, test_worker_start_preprocess_not_found) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  method_signature.preprocess_name = "preprocess_fake_fun";
  ServableStorage::Instance().RegisterMethod(method_signature);

  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), " preprocess preprocess_fake_fun not defined")
}

TEST_F(TestPreprocessPostprocess, test_worker_start_postprocess_not_found) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  method_signature.postprocess_name = "postprocess_fake_fun";
  ServableStorage::Instance().RegisterMethod(method_signature);

  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), " postprocess postprocess_fake_fun not defined")
}

TEST_F(TestPreprocessPostprocess, test_preproces_process_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  ServableStorage::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_TRUE(status.IsSuccess());

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  auto request_servable_spec = request.mutable_servable_spec();
  request_servable_spec->set_name(servable_name_);
  request_servable_spec->set_method_name("add_cast");
  request_servable_spec->set_version_number(0);

  std::vector<std::vector<int32_t>> y_data_list;
  for (size_t k = 0; k < instances_count; k++) {
    std::vector<int32_t> x1_data = {1, 2, 3, 4};
    std::vector<int32_t> x2_data = {2, 3, 4, 5};
    std::vector<int32_t> y_data;
    for (size_t i = 0; i < x1_data.size(); i++) {
      x1_data[i] *= (k + 1);
      x2_data[i] *= (k + 1);
      y_data.push_back(x1_data[i] + x2_data[i]);
    }
    y_data_list.push_back(y_data);

    auto instance = request.add_instances();
    auto &input_map = (*instance->mutable_items());
    // input x1, required int32 input
    InitTensor(&input_map["x1"], {2, 2}, proto::MS_FLOAT32, x1_data.data(), x1_data.size() * sizeof(int32_t));
    // input x2, required int32 input
    InitTensor(&input_map["x2"], {2, 2}, proto::MS_FLOAT32, x2_data.data(), x2_data.size() * sizeof(int32_t));
  }

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.error_msg_size(), instances_count);
  ExpectContainMsg(reply.error_msg(0).error_msg(), "Preprocess failed: Input data type invalid");
}

TEST_F(TestPreprocessPostprocess, test_postproces_process_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
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
  method_signature.postprocess_inputs = {{kPredictPhaseTag_Input, 0}};  // use method input as postprocess input
  // servable output as method output
  method_signature.returns = {{kPredictPhaseTag_Predict, 0}};

  ServableStorage::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_TRUE(status.IsSuccess());

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  auto request_servable_spec = request.mutable_servable_spec();
  request_servable_spec->set_name(servable_name_);
  request_servable_spec->set_method_name("add_cast");
  request_servable_spec->set_version_number(0);

  std::vector<std::vector<int32_t>> y_data_list;
  for (size_t k = 0; k < instances_count; k++) {
    std::vector<int32_t> x1_data = {1, 2, 3, 4};
    std::vector<int32_t> x2_data = {2, 3, 4, 5};
    std::vector<int32_t> y_data;
    for (size_t i = 0; i < x1_data.size(); i++) {
      x1_data[i] *= (k + 1);
      x2_data[i] *= (k + 1);
      y_data.push_back(x1_data[i] + x2_data[i]);
    }
    y_data_list.push_back(y_data);

    auto instance = request.add_instances();
    auto &input_map = (*instance->mutable_items());
    // input x1,
    InitTensor(&input_map["x1"], {2, 2}, proto::MS_INT32, x1_data.data(), x1_data.size() * sizeof(int32_t));
    // input x2
    InitTensor(&input_map["x2"], {2, 2}, proto::MS_INT32, x2_data.data(), x2_data.size() * sizeof(int32_t));
  }

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.error_msg_size(), instances_count);
  ExpectContainMsg(reply.error_msg(0).error_msg(), "Postprocess failed: Input data type invalid");
}

TEST_F(TestPreprocessPostprocess, test_preproces_input_invalid1_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  method_signature.preprocess_inputs = {{kPredictPhaseTag_Preproces, 0}, {kPredictPhaseTag_Input, 1}};
  ServableStorage::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_FALSE(status.IsSuccess());

  Status expect_status = INFER_STATUS(FAILED) << "method add_cast"
                                              << ", the data of preprocess " << 0 << "th input cannot not come from '"
                                              << kPredictPhaseTag_Preproces << "'";
  ExpectContainMsg(status.StatusMessage(), expect_status.StatusMessage());
}

TEST_F(TestPreprocessPostprocess, test_preproces_input_invalid2_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  method_signature.preprocess_inputs = {{kPredictPhaseTag_Input, 0}, {kPredictPhaseTag_Predict, 1}};
  ServableStorage::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_FALSE(status.IsSuccess());
  Status expect_status = INFER_STATUS(FAILED) << "method add_cast"
                                              << ", the data of preprocess " << 1 << "th input cannot not come from '"
                                              << kPredictPhaseTag_Predict << "'";
  ExpectContainMsg(status.StatusMessage(), expect_status.StatusMessage());
}

TEST_F(TestPreprocessPostprocess, test_preproces_input_invalid3_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  method_signature.preprocess_inputs = {{kPredictPhaseTag_Input, 0}, {kPredictPhaseTag_Postprocess, 1}};
  ServableStorage::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_FALSE(status.IsSuccess());
  Status expect_status = INFER_STATUS(FAILED) << "method add_cast"
                                              << ", the data of preprocess " << 1 << "th input cannot not come from '"
                                              << kPredictPhaseTag_Postprocess << "'";
  ExpectContainMsg(status.StatusMessage(), expect_status.StatusMessage());
}

TEST_F(TestPreprocessPostprocess, test_preproces_input_invalid4_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  method_signature.preprocess_inputs = {{kPredictPhaseTag_Input, 0}, {kPredictPhaseTag_Input, 2}};
  ServableStorage::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "the preprocess 1th input uses method 2th input,"
                   " that is greater than the method inputs size");
}

TEST_F(TestPreprocessPostprocess, test_predict_input_invalid1_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  method_signature.servable_inputs = {{kPredictPhaseTag_Predict, 0}, {kPredictPhaseTag_Preproces, 1}};
  ServableStorage::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_FALSE(status.IsSuccess());
  Status expect_status = INFER_STATUS(FAILED) << "method add_cast"
                                              << ", the data of servable " << 0 << "th input cannot not come from '"
                                              << kPredictPhaseTag_Predict << "'";
  ExpectContainMsg(status.StatusMessage(), expect_status.StatusMessage());
}

TEST_F(TestPreprocessPostprocess, test_predict_input_invalid2_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  method_signature.servable_inputs = {{kPredictPhaseTag_Preproces, 0}, {kPredictPhaseTag_Postprocess, 1}};
  ServableStorage::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_FALSE(status.IsSuccess());
  Status expect_status = INFER_STATUS(FAILED) << "method add_cast"
                                              << ", the data of servable " << 1 << "th input cannot not come from '"
                                              << kPredictPhaseTag_Postprocess << "'";
  ExpectContainMsg(status.StatusMessage(), expect_status.StatusMessage());
}

TEST_F(TestPreprocessPostprocess, test_predict_input_invalid3_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  method_signature.servable_inputs = {{kPredictPhaseTag_Preproces, 0}, {kPredictPhaseTag_Preproces, 2}};
  ServableStorage::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "the servable 1th input uses preprocess 2th output, "
                   "that is greater than the preprocess outputs size ");
}

TEST_F(TestPreprocessPostprocess, test_predict_input_invalid4_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  method_signature.servable_inputs = {{kPredictPhaseTag_Input, 2}, {kPredictPhaseTag_Preproces, 1}};
  ServableStorage::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "the servable 0th input uses method 2th input, "
                   "that is greater than the method inputs size ");
}

TEST_F(TestPreprocessPostprocess, test_postprocess_input_invalid1_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  method_signature.postprocess_inputs = {{kPredictPhaseTag_Postprocess, 0}};
  ServableStorage::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_FALSE(status.IsSuccess());
  Status expect_status = INFER_STATUS(FAILED) << "method add_cast"
                                              << ", the data of postprocess " << 0 << "th input cannot not come from '"
                                              << kPredictPhaseTag_Postprocess << "'";
  ExpectContainMsg(status.StatusMessage(), expect_status.StatusMessage());
}

TEST_F(TestPreprocessPostprocess, test_postprocess_input_invalid2_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  method_signature.postprocess_inputs = {{kPredictPhaseTag_Input, 2}};
  ServableStorage::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "the postprocess 0th input uses method 2th input, "
                   "that is greater than the method inputs size");
}

TEST_F(TestPreprocessPostprocess, test_postprocess_input_invalid3_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  method_signature.postprocess_inputs = {{kPredictPhaseTag_Preproces, 2}};
  ServableStorage::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "the postprocess 0th input uses preprocess 2th output, "
                   "that is greater than the preprocess outputs size ");
}

TEST_F(TestPreprocessPostprocess, test_postprocess_input_invalid4_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  method_signature.postprocess_inputs = {{kPredictPhaseTag_Predict, 1}};
  ServableStorage::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "the postprocess 0th input uses servable 1th output, "
                   "that is greater than the servable outputs size");
}

TEST_F(TestPreprocessPostprocess, test_return_invalid1_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  method_signature.returns = {{kPredictPhaseTag_Input, 2}};
  ServableStorage::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "the method 0th output uses method "
                   "2th input, that is greater than the method inputs size");
}

TEST_F(TestPreprocessPostprocess, test_return_invalid2_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  method_signature.returns = {{kPredictPhaseTag_Preproces, 2}};
  ServableStorage::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "the method 0th output uses preprocess "
                   "2th output, that is greater than the preprocess outputs size");
}

TEST_F(TestPreprocessPostprocess, test_return_invalid3_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  method_signature.returns = {{kPredictPhaseTag_Predict, 1}};
  ServableStorage::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "the method 0th output uses servable "
                   "1th output, that is greater than the servable outputs size");
}

TEST_F(TestPreprocessPostprocess, test_return_invalid4_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableStorage::Instance().RegisterInputOutputInfo("test_servable", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  method_signature.returns = {{kPredictPhaseTag_Postprocess, 1}};
  ServableStorage::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 0);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "the method 0th output uses postprocess "
                   "1th output, that is greater than the postprocess outputs size");
}

}  // namespace serving
}  // namespace mindspore
