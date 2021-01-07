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

#define private public
#undef private

using std::string;
using std::vector;
namespace mindspore {
namespace serving {

TEST_F(TestMasterWorkerClient, test_master_worker_success) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  RegisterAddServable();

  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  auto y_data = InitOneInstanceRequest(&request, servable_name_, "add_common", 0);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.instances_size(), 1);
  ASSERT_EQ(reply.error_msg_size(), 0);
  auto &output_instance = reply.instances(0);
  ASSERT_EQ(output_instance.items_size(), 1);
  auto &output_items = output_instance.items();
  ASSERT_EQ(output_items.begin()->first, "y");
  auto &output_tensor = output_items.begin()->second;

  CheckTensor(output_tensor, {2, 2}, proto::MS_FLOAT32, y_data.data(), y_data.size() * sizeof(float));
}

TEST_F(TestMasterWorkerClient, test_master_worker_success_version_number_1_request_version_1) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  RegisterAddServable();

  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  auto y_data = InitOneInstanceRequest(&request, servable_name_, "add_common", 1);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.instances_size(), 1);
  ASSERT_EQ(reply.error_msg_size(), 0);
  auto &output_instance = reply.instances(0);
  ASSERT_EQ(output_instance.items_size(), 1);
  auto &output_items = output_instance.items();
  ASSERT_EQ(output_items.begin()->first, "y");
  auto &output_tensor = output_items.begin()->second;

  CheckTensor(output_tensor, {2, 2}, proto::MS_FLOAT32, y_data.data(), y_data.size() * sizeof(float));
}

TEST_F(TestMasterWorkerClient, test_master_worker_success_version_number_2_request_version_2) {
  Init("test_servable_dir", "test_servable", 2, "test_add.mindir");
  RegisterAddServable();

  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  auto y_data = InitOneInstanceRequest(&request, servable_name_, "add_common", 2);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  CheckInstanceResult(reply, y_data);
}

TEST_F(TestMasterWorkerClient, test_master_worker_success_version_number_2_request_lastest) {
  Init("test_servable_dir", "test_servable", 2, "test_add.mindir");
  RegisterAddServable();

  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  auto y_data = InitOneInstanceRequest(&request, servable_name_, "add_common", 0);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  CheckInstanceResult(reply, y_data);
}

TEST_F(TestMasterWorkerClient, test_master_worker_success_multi_version_number_1_2_request_lastest) {
  auto servable_dir = std::string(test_info_->test_case_name()) + "_test_servable_dir";
  Init(servable_dir, "test_servable", 1, "test_add.mindir");
  Init(servable_dir, "test_servable", 2, "test_add.mindir");

  RegisterAddServable();

  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  auto y_data = InitOneInstanceRequest(&request, servable_name_, "add_common", 0);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  CheckInstanceResult(reply, y_data);
}

TEST_F(TestMasterWorkerClient, test_master_worker_success_version_number_1_2_request_2) {
  auto servable_dir = std::string(test_info_->test_case_name()) + "_test_servable_dir";
  Init(servable_dir, "test_servable", 1, "test_add.mindir");
  Init(servable_dir, "test_servable", 2, "test_add.mindir");

  RegisterAddServable();

  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  auto y_data = InitOneInstanceRequest(&request, servable_name_, "add_common", 2);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  CheckInstanceResult(reply, y_data);
}

TEST_F(TestMasterWorkerClient, test_master_worker_success_version_number_1_2_request_1_failed) {
  auto servable_dir = std::string(test_info_->test_case_name()) + "_test_servable_dir";
  Init(servable_dir, "test_servable", 1, "test_add.mindir");
  Init(servable_dir, "test_servable", 2, "test_add.mindir");

  RegisterAddServable();

  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  auto y_data = InitOneInstanceRequest(&request, servable_name_, "add_common", 1);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.error_msg_size(), 1);
  ExpectContainMsg(reply.error_msg(0).error_msg(), "servable is not available");
}

TEST_F(TestMasterWorkerClient, test_master_worker_three_instance_success) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  RegisterAddServable();

  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  // input float32 --> servable float32-float32, shape [2, 2]
  auto y_data_list = InitMultiInstancesRequest(&request, servable_name_, "add_common", 0, instances_count);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  CheckMultiInstanceResult(reply, y_data_list, instances_count);
}

TEST_F(TestMasterWorkerClient, test_master_worker_input_size_not_match_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  RegisterAddServable();

  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  auto request_servable_spec = request.mutable_servable_spec();
  request_servable_spec->set_name(servable_name_);
  request_servable_spec->set_method_name("add_common");
  request_servable_spec->set_version_number(0);

  size_t instances_count = 3;
  std::vector<std::vector<float>> y_data_list;
  for (size_t k = 0; k < instances_count; k++) {
    std::vector<float> x1_data = {1.1, 2.2};
    std::vector<float> x2_data = {1.2, 2.3};
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
    InitTensor(&input_map["x1"], {2}, proto::MS_FLOAT32, x1_data.data(), x1_data.size() * sizeof(float));
    // input x2
    InitTensor(&input_map["x2"], {2}, proto::MS_FLOAT32, x2_data.data(), x2_data.size() * sizeof(float));
  }

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.error_msg_size(), instances_count);
}

TEST_F(TestMasterWorkerClient, test_master_worker_with_batch_dim_true_success) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  RegisterAddServable(true);  // with_batch_dim = true

  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  // input float32 --> servable float32-float32, shape [2]
  auto y_data_list = InitMultiInstancesShape2Request(&request, servable_name_, "add_common", 0, instances_count);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  CheckMultiInstanceResult(reply, y_data_list, instances_count);
}

TEST_F(TestMasterWorkerClient, test_master_worker_with_batch_dim_true_input_size_not_match_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  RegisterAddServable(true);  // with_batch_dim = true
  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  // shape [2,2] not match required shape [2] as with_batch_dim = true
  auto y_data = InitMultiInstancesRequest(&request, servable_name_, "add_common", 0, instances_count);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.error_msg_size(), instances_count);
}

TEST_F(TestMasterWorkerClient, test_master_worker_error_servable_name) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  RegisterAddServable();
  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  // invalid servable name
  auto y_data = InitMultiInstancesRequest(&request, servable_name_ + "_error", "add_common", 0, instances_count);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.error_msg_size(), 1);
  ExpectContainMsg(reply.error_msg(0).error_msg(), "servable is not available");
}

TEST_F(TestMasterWorkerClient, test_master_worker_error_method_name) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  RegisterAddServable();
  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  // invalid method name
  auto y_data = InitMultiInstancesRequest(&request, servable_name_, "add_common_error", 0, instances_count);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.error_msg_size(), 1);
  ExpectContainMsg(reply.error_msg(0).error_msg(), "method is not available");
}

TEST_F(TestMasterWorkerClient, test_master_worker_error_version_number) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  RegisterAddServable();
  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  // invalid version_number
  auto y_data = InitMultiInstancesRequest(&request, servable_name_, "add_common", 2, instances_count);

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.error_msg_size(), 1);
  ExpectContainMsg(reply.error_msg(0).error_msg(), "servable is not available");
}

TEST_F(TestMasterWorkerClient, test_master_worker_invalid_input_name) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  RegisterAddServable();
  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  // invalid version_number
  auto request_servable_spec = request.mutable_servable_spec();
  request_servable_spec->set_name(servable_name_);
  request_servable_spec->set_method_name("add_common");
  request_servable_spec->set_version_number(0);

  std::vector<std::vector<float>> y_data_list;
  for (size_t k = 0; k < instances_count; k++) {
    std::vector<float> x1_data = {1.1, 2.2, 3.3, 4.4};
    std::vector<float> x2_data = {1.2, 2.3, 3.4, 4.5};
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
    InitTensor(&input_map["x1"], {2, 2}, proto::MS_FLOAT32, x1_data.data(), x1_data.size() * sizeof(float));
    // input x3, expected is x2
    InitTensor(&input_map["x3"], {2, 2}, proto::MS_FLOAT32, x2_data.data(), x2_data.size() * sizeof(float));
  }

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.error_msg_size(), 1);
  ExpectContainMsg(reply.error_msg(0).error_msg(), "Cannot find input x2 in instance input");
}

TEST_F(TestMasterWorkerClient, test_master_worker_three_instance_one_input_invalid_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  RegisterAddServable();

  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  // input float32 --> servable float32-float32, shape [2, 2]
  auto y_data_list = InitMultiInstancesRequest(&request, servable_name_, "add_common", 0, instances_count);
  auto items = request.mutable_instances(1)->mutable_items();
  auto it = items->find("x2");
  ASSERT_TRUE(it != items->end());
  items->erase(it);  // erase x2 input

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.error_msg_size(), 1);
  ExpectContainMsg(reply.error_msg(0).error_msg(), "Cannot find input x2 in instance input");
}

TEST_F(TestMasterWorkerClient, test_master_worker_extra_input_success) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  RegisterAddServable();
  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  // invalid version_number
  auto request_servable_spec = request.mutable_servable_spec();
  request_servable_spec->set_name(servable_name_);
  request_servable_spec->set_method_name("add_common");
  request_servable_spec->set_version_number(0);

  std::vector<std::vector<float>> y_data_list;
  for (size_t k = 0; k < instances_count; k++) {
    std::vector<float> x1_data = {1.1, 2.2, 3.3, 4.4};
    std::vector<float> x2_data = {1.2, 2.3, 3.4, 4.5};
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
    InitTensor(&input_map["x1"], {2, 2}, proto::MS_FLOAT32, x1_data.data(), x1_data.size() * sizeof(float));
    // input x2
    InitTensor(&input_map["x2"], {2, 2}, proto::MS_FLOAT32, x2_data.data(), x2_data.size() * sizeof(float));
    // extra input x3
    InitTensor(&input_map["x3"], {2, 2}, proto::MS_FLOAT32, x2_data.data(), x2_data.size() * sizeof(float));
  }

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  CheckMultiInstanceResult(reply, y_data_list, instances_count);
}

TEST_F(TestMasterWorkerClient, test_master_worker_invalid_input_datatype_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  RegisterAddServable();
  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  // invalid version_number
  auto request_servable_spec = request.mutable_servable_spec();
  request_servable_spec->set_name(servable_name_);
  request_servable_spec->set_method_name("add_common");
  request_servable_spec->set_version_number(0);

  std::vector<std::vector<float>> y_data_list;
  for (size_t k = 0; k < instances_count; k++) {
    std::vector<float> x1_data = {1.1, 2.2, 3.3, 4.4};
    std::vector<float> x2_data = {1.2, 2.3, 3.4, 4.5};
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
    InitTensor(&input_map["x1"], {2, 2}, proto::MS_FLOAT32, x1_data.data(), x1_data.size() * sizeof(float));
    // input x2, invalid data type
    InitTensor(&input_map["x2"], {2, 2}, proto::MS_INT32, x2_data.data(), x2_data.size() * sizeof(float));
  }

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.error_msg_size(), 3);
  ExpectContainMsg(reply.error_msg(0).error_msg(), "Given model input 1 data type");
}

TEST_F(TestMasterWorkerClient, test_master_worker_with_batch_dim_true_invalid_input_datatype_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  RegisterAddServable(true);  // with_batch_dim=true
  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  // invalid version_number
  auto request_servable_spec = request.mutable_servable_spec();
  request_servable_spec->set_name(servable_name_);
  request_servable_spec->set_method_name("add_common");
  request_servable_spec->set_version_number(0);

  std::vector<std::vector<float>> y_data_list;
  for (size_t k = 0; k < instances_count; k++) {
    std::vector<float> x1_data = {1.1, 2.2};
    std::vector<float> x2_data = {1.2, 2.3};
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
    InitTensor(&input_map["x1"], {2}, proto::MS_FLOAT32, x1_data.data(), x1_data.size() * sizeof(float));
    // input x2, invalid data type
    InitTensor(&input_map["x2"], {2}, proto::MS_INT32, x2_data.data(), x2_data.size() * sizeof(float));
  }

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.error_msg_size(), 3);
  ExpectContainMsg(reply.error_msg(0).error_msg(), "Given model input 1 data type");
}

TEST_F(TestMasterWorkerClient, test_master_worker_invalid_input_datasize_not_match_shape_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  RegisterAddServable();
  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  // invalid version_number
  auto request_servable_spec = request.mutable_servable_spec();
  request_servable_spec->set_name(servable_name_);
  request_servable_spec->set_method_name("add_common");
  request_servable_spec->set_version_number(0);

  std::vector<std::vector<float>> y_data_list;
  for (size_t k = 0; k < instances_count; k++) {
    std::vector<float> x1_data = {1.1, 2.2, 3.3, 4.4};
    std::vector<float> x2_data = {1.2, 2.3, 3.4, 4.5};
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
    InitTensor(&input_map["x1"], {2, 2}, proto::MS_FLOAT32, x1_data.data(), x1_data.size() * sizeof(float));
    // input x2, invalid data size
    InitTensor(&input_map["x2"], {2, 2}, proto::MS_FLOAT32, x2_data.data(), (x2_data.size() - 1) * sizeof(float));
  }

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.error_msg_size(), 1);  // proto parse check failed
  ExpectContainMsg(reply.error_msg(0).error_msg(), "Tensor check failed: input data size");
}

TEST_F(TestMasterWorkerClient, test_master_worker_invalid_input_datasize_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  RegisterAddServable();
  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  // invalid version_number
  auto request_servable_spec = request.mutable_servable_spec();
  request_servable_spec->set_name(servable_name_);
  request_servable_spec->set_method_name("add_common");
  request_servable_spec->set_version_number(0);

  std::vector<std::vector<float>> y_data_list;
  for (size_t k = 0; k < instances_count; k++) {
    std::vector<float> x1_data = {1.1, 2.2, 3.3, 4.4};
    std::vector<float> x2_data = {1.2, 2.3, 3.4, 4.5};
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
    InitTensor(&input_map["x1"], {2, 2}, proto::MS_FLOAT32, x1_data.data(), x1_data.size() * sizeof(float));
    // input x2, invalid data size
    InitTensor(&input_map["x2"], {2, 1}, proto::MS_FLOAT32, x2_data.data(), (x2_data.size() - 2) * sizeof(float));
  }

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.error_msg_size(), 3);
  ExpectContainMsg(reply.error_msg(0).error_msg(), "Given model input 1 size 8");
}

TEST_F(TestMasterWorkerClient, test_master_worker_with_batch_dim_true_invalid_input_datasize_failed) {
  Init("test_servable_dir", "test_servable", 1, "test_add.mindir");
  RegisterAddServable(true);  // with_batch_dim=true
  // start_servable
  StartAddServable();

  // run servable
  proto::PredictRequest request;
  size_t instances_count = 3;
  // invalid version_number
  auto request_servable_spec = request.mutable_servable_spec();
  request_servable_spec->set_name(servable_name_);
  request_servable_spec->set_method_name("add_common");
  request_servable_spec->set_version_number(0);

  std::vector<std::vector<float>> y_data_list;
  for (size_t k = 0; k < instances_count; k++) {
    std::vector<float> x1_data = {1.1, 2.2};
    std::vector<float> x2_data = {1.2, 2.3};
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
    InitTensor(&input_map["x1"], {2}, proto::MS_FLOAT32, x1_data.data(), x1_data.size() * sizeof(float));
    // input x2, invalid data size
    InitTensor(&input_map["x2"], {1}, proto::MS_FLOAT32, x2_data.data(), (x2_data.size() - 1) * sizeof(float));
  }

  proto::PredictReply reply;
  auto grpc_status = Dispatch(request, &reply);
  EXPECT_TRUE(grpc_status.ok());
  // checkout output
  ASSERT_EQ(reply.error_msg_size(), 3);
  ExpectContainMsg(reply.error_msg(0).error_msg(), "Given model input 1 size 4");
}

}  // namespace serving
}  // namespace mindspore
