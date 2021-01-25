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
  // input int32 --> preprocess int32-float32 --> servable float32-float32 --> postprocess int32-int32, shape [2,2]
  auto y_data_list =
    InitMultiInstancesRequest<int32_t, int32_t>(&request, servable_name_, "add_cast", 0, instances_count);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  CheckMultiInstanceResult(reply, y_data_list, instances_count);
}

TEST_F(TestPreprocessPostprocess, test_master_worker_with_preproces_and_postprocess_batching_success) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  // declare_servable
  // with_batch_dim = true
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
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
  // input int32 --> preprocess int32-float32 --> servable float32-float32 --> postprocess int32-int32, shape [2]
  auto y_data_list =
    InitMultiInstancesShape2Request<int32_t, int32_t>(&request, servable_name_, "add_cast", 0, instances_count);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  CheckMultiInstanceResult(reply, y_data_list, instances_count);
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
  // input int32 --> preprocess int32-float32 --> servable float32-float32, shape [2,2]
  auto y_data_list =
    InitMultiInstancesRequest<int32_t, float>(&request, servable_name_, "add_cast", 0, instances_count);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  CheckMultiInstanceResult(reply, y_data_list, instances_count);
}

TEST_F(TestPreprocessPostprocess, test_master_worker_with_only_preproces_batching_success) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  // declare_servable
  // with_batch_dim=true
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
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
  // input int32 --> preprocess int32-float32 --> servable float32-float32, shape [2]
  auto y_data_list =
    InitMultiInstancesShape2Request<int32_t, float>(&request, servable_name_, "add_cast", 0, instances_count);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  CheckMultiInstanceResult(reply, y_data_list, instances_count);
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
  // input float32 --> servable float32-float32 --> postprocess float32-int32, shape [2,2]
  auto y_data_list =
    InitMultiInstancesRequest<float, int32_t>(&request, servable_name_, "add_cast", 0, instances_count);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  CheckMultiInstanceResult(reply, y_data_list, instances_count);
}

TEST_F(TestPreprocessPostprocess, test_master_worker_with_only_postprocess_batching_success) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  // declare_servable
  // with_batch_dim=true
  DeclareServable("test_servable", "test_add.mindir", "mindir", true);
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
  // input float32 --> servable float32-float32 --> postprocess float32-int32, shape [2]
  auto y_data_list =
    InitMultiInstancesShape2Request<float, int32_t>(&request, servable_name_, "add_cast", 0, instances_count);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  CheckMultiInstanceResult(reply, y_data_list, instances_count);
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
  // input float32, invalid for preprocess, which required int32
  auto y_data_list = InitMultiInstancesRequest<float, float>(&request, servable_name_, "add_cast", 0, instances_count);

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
  // input int32, invalid for postprocess
  auto y_data_list =
    InitMultiInstancesRequest<int32_t, int32_t>(&request, servable_name_, "add_cast", 0, instances_count);

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
