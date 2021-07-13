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
    MethodSignature method_signature = InitMethodSig();
    // preprocess
    method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {0, 1}});

    // method input 0 and input 1 as servable input
    method_signature.AddStageModel(model_file_, {{1, 0}, {1, 1}});
    // postprocess
    method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{2, 0}});
    // servable output as method output
    method_signature.SetReturn({{3, 0}});
    return method_signature;
  }
  MethodSignature InitMethodSig() {
    MethodSignature method_signature;
    method_signature.servable_name = "test_servable";
    method_signature.method_name = "add_cast";
    method_signature.inputs = {"x1", "x2"};
    method_signature.outputs = {"y"};
    return method_signature;
  }
  const std::string model_file_ = "test_add.mindir";
};

TEST_F(TestPreprocessPostprocess, test_master_worker_with_preproces_and_postprocess_success) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  // declare_servable
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  // register method
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  ServableRegister::Instance().RegisterMethod(method_signature);

  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
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
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  ServableRegister::Instance().RegisterMethod(method_signature);

  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
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
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);
  MethodSignature method_signature = InitMethodSig();
  // preprocess
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {0, 1}});

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{1, 0}, {1, 1}});
  // servable output as method output
  method_signature.SetReturn({{2, 0}});

  ServableRegister::Instance().RegisterMethod(method_signature);

  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
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
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);
  MethodSignature method_signature = InitMethodSig();
  // preprocess
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {0, 1}});

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{1, 0}, {1, 1}});
  method_signature.SetReturn({{2, 0}});
  ServableRegister::Instance().RegisterMethod(method_signature);

  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_TRUE(status.IsSuccess());

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  // input int32 --> preprocess int32-float32 --> servable float32-float32, shape [2]
  auto y_data_list =
    InitMultiInstancesShape2Request<int32_t, float>(&request, servable_name_, "add_cast", 0, instances_count);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  ASSERT_TRUE(grpc_status.ok());
  // checkout output
  CheckMultiInstanceResult(reply, y_data_list, instances_count);
}

TEST_F(TestPreprocessPostprocess, test_master_worker_with_only_postprocess_success) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  // declare_servable
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  // register method
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);
  MethodSignature method_signature = InitMethodSig();

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{0, 0}, {0, 1}});
  // postprocess
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{1, 0}});
  // servable output as method output
  method_signature.SetReturn({{2, 0}});
  ServableRegister::Instance().RegisterMethod(method_signature);

  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
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
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);
  MethodSignature method_signature = InitMethodSig();

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{0, 0}, {0, 1}});
  // postprocess
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{1, 0}});
  // servable output as method output
  method_signature.SetReturn({{2, 0}});
  ServableRegister::Instance().RegisterMethod(method_signature);

  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
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
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  try {
    MethodSignature method_signature = InitMethodSig();
    // preprocess
    method_signature.AddStageFunction("preprocess_fake_fun", {{0, 0}, {0, 1}});

    // method input 0 and input 1 as servable input
    method_signature.AddStageModel(model_file_, {{1, 0}, {1, 1}});
    // postprocess
    method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{2, 0}});
    // servable output as method output
    method_signature.SetReturn({{3, 0}});
    FAIL();
  } catch (std::runtime_error &ex) {
    ExpectContainMsg(ex.what(), "Function 'preprocess_fake_fun' is not defined")
  }
}

TEST_F(TestPreprocessPostprocess, test_worker_start_postprocess_not_found) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  try {
    MethodSignature method_signature = InitMethodSig();
    // preprocess
    method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {0, 1}});

    // method input 0 and input 1 as servable input
    method_signature.AddStageModel(model_file_, {{1, 0}, {1, 1}});
    // postprocess
    method_signature.AddStageFunction("postprocess_fake_fun", {{2, 0}});
    // servable output as method output
    method_signature.SetReturn({{3, 0}});
    FAIL();
  } catch (std::runtime_error &ex) {
    ExpectContainMsg(ex.what(), "Function 'postprocess_fake_fun' is not defined")
  }
}

TEST_F(TestPreprocessPostprocess, test_preproces_process_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitDefaultMethod();
  ServableRegister::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
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
  ExpectContainMsg(reply.error_msg(0).error_msg(), "Call failed: Input data type invalid");
}

TEST_F(TestPreprocessPostprocess, test_postproces_process_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitMethodSig();
  // preprocess
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {0, 1}});

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{1, 0}, {1, 1}});
  // postprocess
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp",
                                    {{0, 0}});  // use method input as postprocess input
  // servable output as method output
  method_signature.SetReturn({{2, 0}});

  ServableRegister::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
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
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitMethodSig();
  // preprocess
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{1, 0}, {0, 1}});

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{1, 0}, {1, 1}});
  // postprocess
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{2, 0}});
  // servable output as method output
  method_signature.SetReturn({{3, 0}});
  ServableRegister::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());

  ExpectContainMsg(status.StatusMessage(), "The 0th input data of stage 1 cannot not come from stage 1");
}

TEST_F(TestPreprocessPostprocess, test_preproces_input_invalid2_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitMethodSig();
  // preprocess
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {2, 1}});

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{1, 0}, {1, 1}});
  // postprocess
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{2, 0}});
  // servable output as method output
  method_signature.SetReturn({{3, 0}});
  ServableRegister::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), "The 1th input data of stage 1 cannot not come from stage 2");
}

TEST_F(TestPreprocessPostprocess, test_preproces_input_invalid3_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitMethodSig();
  // preprocess
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {3, 1}});

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{1, 0}, {1, 1}});
  // postprocess
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{2, 0}});
  // servable output as method output
  method_signature.SetReturn({{3, 0}});
  ServableRegister::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), "The 1th input data of stage 1 cannot not come from stage 3");
}

TEST_F(TestPreprocessPostprocess, test_preproces_input_invalid4_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitMethodSig();
  // preprocess
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {0, 2}});

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{1, 0}, {1, 1}});
  // postprocess
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{2, 0}});
  // servable output as method output
  method_signature.SetReturn({{3, 0}});
  ServableRegister::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "The stage 1 1th input uses method 2th input, that is greater than the method inputs size 2");
}

TEST_F(TestPreprocessPostprocess, test_predict_input_invalid1_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitMethodSig();
  // preprocess
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {0, 1}});

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{2, 0}, {1, 1}});
  // postprocess
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{2, 0}});
  // servable output as method output
  method_signature.SetReturn({{3, 0}});
  ServableRegister::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), "The 0th input data of stage 2 cannot not come from stage 2");
}

TEST_F(TestPreprocessPostprocess, test_predict_input_invalid2_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitMethodSig();
  // preprocess
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {0, 1}});

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{1, 0}, {3, 1}});
  // postprocess
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{2, 0}});
  // servable output as method output
  method_signature.SetReturn({{3, 0}});
  ServableRegister::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), "The 1th input data of stage 2 cannot not come from stage 3");
}

TEST_F(TestPreprocessPostprocess, test_predict_input_invalid3_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitMethodSig();
  // preprocess
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {0, 1}});

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{1, 0}, {1, 2}});
  // postprocess
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{2, 0}});
  // servable output as method output
  method_signature.SetReturn({{3, 0}});
  ServableRegister::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "The stage(begin with 1) 2 1th input uses c++ function stub_preprocess_cast_int32_to_fp32_cpp "
                   "2th output, that is greater than the function output size 2");
}

TEST_F(TestPreprocessPostprocess, test_predict_input_invalid4_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitMethodSig();
  // preprocess
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {0, 1}});

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{0, 2}, {1, 1}});
  // postprocess
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{2, 0}});
  // servable output as method output
  method_signature.SetReturn({{3, 0}});
  ServableRegister::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "The stage 2 0th input uses method 2th input, that is greater than the method inputs size 2");
}

TEST_F(TestPreprocessPostprocess, test_postprocess_input_invalid1_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitMethodSig();
  // preprocess
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {0, 1}});

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{1, 0}, {1, 1}});
  // postprocess
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{3, 0}});
  // servable output as method output
  method_signature.SetReturn({{3, 0}});
  ServableRegister::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), "The 0th input data of stage 3 cannot not come from stage 3");
}

TEST_F(TestPreprocessPostprocess, test_postprocess_input_invalid2_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitMethodSig();
  // preprocess
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {0, 1}});

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{1, 0}, {1, 1}});
  // postprocess
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{0, 2}});
  // servable output as method output
  method_signature.SetReturn({{3, 0}});
  ServableRegister::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "The stage 3 0th input uses method 2th input, that is greater than the method inputs size 2");
}

TEST_F(TestPreprocessPostprocess, test_postprocess_input_invalid3_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitMethodSig();
  // preprocess
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {0, 1}});

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{1, 0}, {1, 1}});
  // postprocess
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{1, 2}});
  // servable output as method output
  method_signature.SetReturn({{3, 0}});
  ServableRegister::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "The stage(begin with 1) 3 0th input uses c++ function stub_preprocess_cast_int32_to_fp32_cpp"
                   " 2th output, that is greater than the function output size 2");
}

TEST_F(TestPreprocessPostprocess, test_postprocess_input_invalid4_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitMethodSig();
  // preprocess
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {0, 1}});

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{1, 0}, {1, 1}});
  // postprocess
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{2, 1}});
  // servable output as method output
  method_signature.SetReturn({{3, 0}});
  ServableRegister::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), "The stage(begin with 1) 3 0th input uses model "
                   "test_add.mindir subgraph 0 1th output, that is greater than the model output size 1");
}

TEST_F(TestPreprocessPostprocess, test_return_invalid1_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitMethodSig();
  // preprocess
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {0, 1}});

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{1, 0}, {1, 1}});
  // postprocess
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{2, 0}});
  // servable output as method output
  method_signature.SetReturn({{0, 2}});
  ServableRegister::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "The stage 4 0th input uses method 2th input, "
                   "that is greater than the method inputs size 2");
}

TEST_F(TestPreprocessPostprocess, test_return_invalid2_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitMethodSig();
  // preprocess
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {0, 1}});

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{1, 0}, {1, 1}});
  // postprocess
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{2, 0}});
  // servable output as method output
  method_signature.SetReturn({{1, 2}});
  ServableRegister::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "The stage(begin with 1) 4 0th input uses c++ function stub_preprocess_cast_int32_to_fp32_cpp"
                   " 2th output, that is greater than the function output size 2");
}

TEST_F(TestPreprocessPostprocess, test_return_invalid3_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitMethodSig();
  // preprocess
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {0, 1}});

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel(model_file_, {{1, 0}, {1, 1}});
  // postprocess
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{2, 0}});
  // servable output as method output
  method_signature.SetReturn({{2, 1}});
  ServableRegister::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(), "The stage(begin with 1) 4 0th input uses model "
                   "test_add.mindir subgraph 0 1th output, that is greater than the model output size 1");
}

TEST_F(TestPreprocessPostprocess, test_return_invalid4_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  DeclareServable("test_servable", "test_add.mindir", "mindir", false);
  ServableRegister::Instance().RegisterInputOutputInfo("test_add.mindir", 2, 1);

  MethodSignature method_signature = InitMethodSig();
  // preprocess
  method_signature.AddStageFunction("stub_preprocess_cast_int32_to_fp32_cpp", {{0, 0}, {0, 1}});

  // method input 0 and input 1 as servable input
  method_signature.AddStageModel("test_add.mindir", {{1, 0}, {1, 1}});
  // postprocess
  method_signature.AddStageFunction("stub_postprocess_cast_fp32_to_int32_cpp", {{2, 0}});
  // servable output as method output
  method_signature.SetReturn({{3, 1}});
  ServableRegister::Instance().RegisterMethod(method_signature);
  // start_servable
  Status status = StartServable("test_servable_dir", "test_servable", 1);
  EXPECT_FALSE(status.IsSuccess());
  ExpectContainMsg(status.StatusMessage(),
                   "The stage(begin with 1) 4 0th input uses c++ function stub_postprocess_cast_fp32_to_int32_cpp"
                                " 1th output, that is greater than the function output size 1");
}

}  // namespace serving
}  // namespace mindspore
