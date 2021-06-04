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
#include "common/common_test.h"
#include "common/tensor_base.h"
#define private public
#include "worker/distributed_worker/distributed_process/distributed_process.h"
#include "worker/distributed_worker/notify_distributed/notify_worker.h"
#undef private

using std::string;
using std::vector;
namespace mindspore {
namespace serving {
class TestAgentConfigAcquire : public UT::Common {
 public:
  TestAgentConfigAcquire() = default;
  virtual void SetUp() {}
  virtual void TearDown() {
    UT::Common::TearDown();
  }
};

TEST_F(TestAgentConfigAcquire, test_agent_config_acquire_success) {
  std::shared_ptr<DistributedServable> servable = std::make_shared<DistributedServable>();
  std::string rank_table_content = "rank table content";
  CommonServableMeta commonServableMeta;
  commonServableMeta.servable_name = "servable_name";
  commonServableMeta.outputs_count = 1;
  commonServableMeta.inputs_count = 1;
  commonServableMeta.with_batch_dim = false;
  commonServableMeta.without_batch_dim_inputs.push_back(8);
  DistributedServableMeta distributedServableMeta;
  distributedServableMeta.stage_size = 8;
  distributedServableMeta.rank_size = 8;
  OneRankConfig oneRankConfig;
  oneRankConfig.ip = "1.1.1.1";
  oneRankConfig.device_id = 0;
  servable->config_.rank_table_content = rank_table_content;
  servable->config_.common_meta = commonServableMeta;
  servable->config_.distributed_meta = distributedServableMeta;
  servable->config_.rank_list.push_back(oneRankConfig);
  servable->config_loaded_ = true;
  const std::string server_address = "any_addr";
  MSDistributedImpl mSDistributedImpl(servable, server_address);
  grpc::ServerContext context;
  const proto::AgentConfigAcquireRequest request;
  proto::AgentConfigAcquireReply reply;
  grpc::Status status = mSDistributedImpl.AgentConfigAcquire(&context, &request, &reply);
  ASSERT_EQ(status.error_code(), 0);

  DistributedServableConfig config;
  GrpcNotifyDistributeWorker::ParseAgentConfigAcquireReply(reply, &config);
  ASSERT_EQ(config.rank_table_content, rank_table_content);
  ASSERT_EQ(config.common_meta.servable_name, "servable_name");
  ASSERT_EQ(config.common_meta.inputs_count, 1);
  ASSERT_EQ(config.common_meta.outputs_count, 1);
  ASSERT_EQ(config.common_meta.with_batch_dim, false);
  ASSERT_EQ(config.common_meta.without_batch_dim_inputs.size(), 1);
  ASSERT_EQ(config.common_meta.without_batch_dim_inputs.at(0), 8);
  ASSERT_EQ(config.distributed_meta.rank_size, 8);
  ASSERT_EQ(config.distributed_meta.stage_size, 8);
  ASSERT_EQ(config.rank_list.size(), 1);
  OneRankConfig tempRankConfig = config.rank_list.at(0);
  ASSERT_EQ(tempRankConfig.device_id, 0);
  ASSERT_EQ(tempRankConfig.ip, "1.1.1.1");
}

TEST_F(TestAgentConfigAcquire, test_agent_config_acquire_not_load_config_failed) {
  std::shared_ptr<DistributedServable> servable = std::make_shared<DistributedServable>();
  servable->config_loaded_ = false;
  const std::string server_address = "any_addr";
  MSDistributedImpl mSDistributedImpl(servable, server_address);
  grpc::ServerContext context;
  const proto::AgentConfigAcquireRequest request;
  proto::AgentConfigAcquireReply reply;
  const grpc::Status status = mSDistributedImpl.AgentConfigAcquire(&context, &request, &reply);
  ASSERT_EQ(status.error_code(), 1);
}

}  // namespace serving
}  // namespace mindspore
