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
#include <thread>
#include <chrono>
#include <vector>
#include "gtest/gtest.h"
#include "common/status.h"
#include "proto/ms_agent.pb.h"
#include "tests/ut/cpp/common/common_test.h"
#include "common/grpc_client.h"
#include "worker/distributed_worker/notify_agent/base_notify_agent.h"
#define private public
#include "common/exit_handle.h"
#include "worker/distributed_worker/distributed_servable.h"
#undef private

namespace mindspore {
namespace serving {

struct AgentInferResult {
  int64_t prediction_time = 0;  // milliseconds
  Status status = SUCCESS;
  int64_t error_code = 0;
  std::string error_msg = "";
};

class FakeNotifyAgent : public BaseNotifyAgent {
 public:
  explicit FakeNotifyAgent(int64_t prediction_time = 0, Status status = SUCCESS, int64_t error_code = 0,
                           std::string error_msg = "")
      : prediction_time_(prediction_time), status_(status), error_code_(error_code), error_msg_(error_msg) {}
  ~FakeNotifyAgent() = default;
  Status Exit() override { return SUCCESS; }
  Status DispatchAsync(const proto::DistributedPredictRequest &request, proto::DistributedPredictReply *reply,
                       AsyncPredictCallback callback) override {
    auto error_msg = reply->mutable_error_msg();
    error_msg->set_error_code(error_code_);
    if (!error_msg_.empty()) {
      error_msg->set_error_msg(error_msg_);
    }

    auto predict = [=]() {
      std::chrono::milliseconds dura(prediction_time_);
      std::this_thread::sleep_for(dura);
      callback(status_);
    };
    std::thread t1(predict);
    t1.detach();
    return SUCCESS;
  }

 private:
  int64_t prediction_time_;  // milliseconds
  Status status_;
  int64_t error_code_;
  std::string error_msg_;
};

class TestDistributedInference : public UT::Common {
 public:
  TestDistributedInference() = default;
  ~TestDistributedInference() = default;

  void InitDistributedServable(std::shared_ptr<DistributedServable> servable, size_t rank_size, size_t stage_size,
                               bool is_running, bool is_loaded) {
    ExitSignalHandle::Instance().is_running_ = is_running;
    servable->model_loaded_ = is_loaded;
    servable->config_.distributed_meta.rank_size = rank_size;
    servable->config_.distributed_meta.stage_size = stage_size;
  }

  void InitAgentSpecMap(std::shared_ptr<DistributedServable> servable,
                        const std::vector<AgentInferResult> &result_list) {
    for (size_t rank_id = 0; rank_id < result_list.size(); ++rank_id) {
      const auto &result = result_list[rank_id];
      DistributedAgentContext agent_context;
      agent_context.notify_agent_ =
        std::make_shared<FakeNotifyAgent>(result.prediction_time, result.status, result.error_code, result.error_msg);
      servable->agent_spec_map_.insert({rank_id, agent_context});
    }
  }
};

TEST_F(TestDistributedInference, test_agent_8_stage_1) {
  auto servable = std::make_shared<DistributedServable>();
  InitDistributedServable(servable, 8, 1, true, true);

  std::vector<AgentInferResult> result_list(8);
  InitAgentSpecMap(servable, result_list);

  std::vector<TensorBasePtr> input, output;
  auto status = servable->Predict(input, &output);

  ASSERT_EQ(status.StatusCode(), SUCCESS);
}

TEST_F(TestDistributedInference, test_agent_4) {
  auto servable = std::make_shared<DistributedServable>();
  InitDistributedServable(servable, 4, 1, true, true);

  std::vector<AgentInferResult> result_list(4);
  InitAgentSpecMap(servable, result_list);

  std::vector<TensorBasePtr> input, output;
  auto status = servable->Predict(input, &output);

  ASSERT_EQ(status.StatusCode(), SUCCESS);
}

TEST_F(TestDistributedInference, test_agent_32_stage_1) {
  auto servable = std::make_shared<DistributedServable>();
  InitDistributedServable(servable, 32, 1, true, true);

  std::vector<AgentInferResult> result_list(32);
  InitAgentSpecMap(servable, result_list);

  std::vector<TensorBasePtr> input, output;
  auto status = servable->Predict(input, &output);

  ASSERT_EQ(status.StatusCode(), SUCCESS);
}

TEST_F(TestDistributedInference, test_agent_32_stage_2) {
  auto servable = std::make_shared<DistributedServable>();
  InitDistributedServable(servable, 32, 2, true, true);

  std::vector<AgentInferResult> result_list(32);
  InitAgentSpecMap(servable, result_list);

  std::vector<TensorBasePtr> input, output;
  auto status = servable->Predict(input, &output);

  ASSERT_EQ(status.StatusCode(), SUCCESS);
}

TEST_F(TestDistributedInference, test_agent_32_stage_4) {
  auto servable = std::make_shared<DistributedServable>();
  InitDistributedServable(servable, 32, 4, true, true);

  std::vector<AgentInferResult> result_list(32);
  InitAgentSpecMap(servable, result_list);

  std::vector<TensorBasePtr> input, output;
  auto status = servable->Predict(input, &output);

  ASSERT_EQ(status.StatusCode(), SUCCESS);
}

TEST_F(TestDistributedInference, test_agent_64_stage_8) {
  auto servable = std::make_shared<DistributedServable>();
  InitDistributedServable(servable, 64, 8, true, true);

  std::vector<AgentInferResult> result_list(64);
  InitAgentSpecMap(servable, result_list);

  std::vector<TensorBasePtr> input, output;
  auto status = servable->Predict(input, &output);

  ASSERT_EQ(status.StatusCode(), SUCCESS);
}

TEST_F(TestDistributedInference, test_output_nullptr) {
  auto servable = std::make_shared<DistributedServable>();
  InitDistributedServable(servable, 32, 4, true, true);

  std::vector<AgentInferResult> result_list(32);
  InitAgentSpecMap(servable, result_list);

  Status status;
  std::vector<TensorBasePtr> input, output;
  ASSERT_ANY_THROW({ status = servable->Predict(input, nullptr); });
  ASSERT_EQ(status.StatusCode(), FAILED);
}

TEST_F(TestDistributedInference, test_agent_infer_more_than_10s) {
  auto servable = std::make_shared<DistributedServable>();
  InitDistributedServable(servable, 32, 4, true, true);

  std::vector<AgentInferResult> result_list(32);
  result_list[20].prediction_time = 11000;
  InitAgentSpecMap(servable, result_list);

  std::vector<TensorBasePtr> input, output;
  auto status = servable->Predict(input, &output);

  ASSERT_EQ(status.StatusCode(), FAILED);
}

TEST_F(TestDistributedInference, test_agent_exit) {
  auto servable = std::make_shared<DistributedServable>();
  InitDistributedServable(servable, 32, 4, false, true);

  std::vector<AgentInferResult> result_list(32);
  InitAgentSpecMap(servable, result_list);

  std::vector<TensorBasePtr> input, output;
  auto status = servable->Predict(input, &output);

  ASSERT_EQ(status.StatusCode(), FAILED);
}

TEST_F(TestDistributedInference, test_rank_size_not_equal_agent_num) {
  auto servable = std::make_shared<DistributedServable>();
  InitDistributedServable(servable, 32, 4, true, true);

  std::vector<AgentInferResult> result_list(12);
  InitAgentSpecMap(servable, result_list);

  Status status;
  std::vector<TensorBasePtr> input, output;
  ASSERT_ANY_THROW({ status = servable->Predict(input, &output); });
  ASSERT_EQ(status.StatusCode(), FAILED);
}

TEST_F(TestDistributedInference, test_agent_reply_with_error_msg) {
  auto servable = std::make_shared<DistributedServable>();
  InitDistributedServable(servable, 32, 4, true, true);

  std::vector<AgentInferResult> result_list(32);
  result_list[10].error_msg = "failed";
  result_list[10].error_code = 1;
  InitAgentSpecMap(servable, result_list);

  std::vector<TensorBasePtr> input, output;
  auto status = servable->Predict(input, &output);
  ASSERT_EQ(status.StatusCode(), FAILED);
}

TEST_F(TestDistributedInference, test_model_not_loaded) {
  auto servable = std::make_shared<DistributedServable>();
  InitDistributedServable(servable, 32, 4, true, false);

  std::vector<AgentInferResult> result_list(32);
  InitAgentSpecMap(servable, result_list);

  Status status;
  std::vector<TensorBasePtr> input, output;
  ASSERT_ANY_THROW({ status = servable->Predict(input, &output); });
  ASSERT_EQ(status.StatusCode(), FAILED);
}

}  // namespace serving
}  // namespace mindspore
