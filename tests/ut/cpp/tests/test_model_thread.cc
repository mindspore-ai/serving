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
#include "common/common_test.h"
#include "master/server.h"
#include "common/tensor_base.h"
#define private public
#include "master/model_thread.h"
#undef private

using std::string;
using std::vector;
namespace mindspore {
namespace serving {
class TestModelThead : public UT::Common {
 public:
  TestModelThead() = default;
};

class MS_API TestNotify : public BaseNotifyWorker {
 public:
  explicit TestNotify(proto::PredictReply *reply) {
    if (reply) {
      reply_ = *reply;
    }
  }
  ~TestNotify() override = default;

  Status Exit() override;

  Status DispatchAsync(const proto::PredictRequest &request, proto::PredictReply *reply,
                       PredictOnFinish on_finish) override;

  proto::PredictReply reply_;
};

Status TestNotify::Exit() { return SUCCESS; }

Status TestNotify::DispatchAsync(const proto::PredictRequest &request, proto::PredictReply *reply,
                                 PredictOnFinish on_finish) {
  *reply = reply_;
  on_finish();
  return SUCCESS;
}

std::shared_ptr<WorkerContext> InitWorkerContext(proto::PredictReply *reply = nullptr) {
  std::shared_ptr<WorkerContext> worker_context = std::make_shared<WorkerContext>();
  std::shared_ptr<BaseNotifyWorker> notify = std::make_shared<TestNotify>(reply);
  WorkerRegSpec spec;
  spec.worker_pid = 1;
  spec.servable_spec.servable_name = "test_servable";
  spec.servable_spec.version_number = 1;
  spec.servable_spec.batch_size = 1;
  spec.servable_spec.methods.push_back(ServableMethodInfo{"add_cast", {}});
  worker_context->OnWorkerRegRequest(spec, notify);
  return worker_context;
}

TEST_F(TestModelThead, AddWorker) {
  ServableMethodInfo method_info;
  method_info.name = "add_cast";
  ModelThread thread("test_servable", "add_cast", 0, 1, method_info);
  uint64_t pid = 1;
  std::shared_ptr<WorkerContext> worker_context = InitWorkerContext();
  Status status = thread.AddWorker(pid, worker_context);
  ASSERT_EQ(status.StatusCode(), SUCCESS);
  status = thread.AddWorker(pid, worker_context);
  ASSERT_EQ(status.StatusCode(), FAILED);
  pid = 2;
  status = thread.AddWorker(pid, worker_context);
  ASSERT_EQ(status.StatusCode(), SUCCESS);
}
TEST_F(TestModelThead, DelWorker) {
  ServableMethodInfo method_info;
  method_info.name = "add_cast";
  ModelThread thread("test_servable", "add_cast", 0, 1, method_info);
  uint64_t pid = 1;
  Status status = thread.DelWorker(pid);
  ASSERT_EQ(status.StatusCode(), FAILED);
  std::shared_ptr<WorkerContext> worker_context = InitWorkerContext();
  status = thread.AddWorker(pid, worker_context);
  ASSERT_EQ(status.StatusCode(), SUCCESS);
  status = thread.DelWorker(pid);
  ASSERT_EQ(status.StatusCode(), SUCCESS);
}
TEST_F(TestModelThead, Dispatch) {
  ServableMethodInfo method_info;
  method_info.name = "add_cast";
  ModelThread thread("test_servable", "add_cast", 0, 1, method_info);
  uint64_t pid = 1;
  std::shared_ptr<WorkerContext> worker_context = InitWorkerContext();
  Status status = thread.AddWorker(pid, worker_context);
  ASSERT_EQ(status.StatusCode(), SUCCESS);
  proto::PredictRequest request;
  request.mutable_servable_spec()->set_name("test_servable");
  request.mutable_servable_spec()->set_version_number(0);
  request.mutable_servable_spec()->set_method_name("add_cast");
  proto::Instance instance;
  auto proto_instance = request.add_instances();
  *proto_instance->mutable_items() = instance.items();
  proto::PredictReply reply;
  PredictOnFinish callback = []() {};
  status = thread.DispatchAsync(request, &reply, callback);
  ASSERT_EQ(status.StatusCode(), SUCCESS);
  status = thread.DelWorker(pid);
  ASSERT_EQ(status.StatusCode(), SUCCESS);
}
TEST_F(TestModelThead, Dispatch1) {
  ServableMethodInfo method_info;
  method_info.name = "add_cast";
  ModelThread thread("test_servable", "add_cast", 0, 1, method_info);
  uint64_t pid = 1;
  std::shared_ptr<WorkerContext> worker_context = InitWorkerContext();
  proto::PredictRequest request;
  request.mutable_servable_spec()->set_name("test_servable");
  request.mutable_servable_spec()->set_version_number(0);
  request.mutable_servable_spec()->set_method_name("add_cast");
  proto::Instance instance;
  auto proto_instance = request.add_instances();
  *proto_instance->mutable_items() = instance.items();
  proto::PredictReply reply;
  PredictOnFinish callback = []() {};
  Status status = thread.DispatchAsync(request, &reply, callback);
  ASSERT_NE(status.StatusCode(), SUCCESS);
  status = thread.AddWorker(pid, worker_context);
  ASSERT_EQ(status.StatusCode(), SUCCESS);
  status = thread.DelWorker(pid);
  ASSERT_EQ(status.StatusCode(), SUCCESS);
}

TEST_F(TestModelThead, Commit) {
  ServableMethodInfo method_info;
  method_info.name = "add_cast";
  ModelThread thread("test_servable", "add_cast", 0, 1, method_info);
  uint64_t pid = 1;

  proto::Instance instance;
  proto::PredictReply reply;
  auto proto_instance1 = reply.add_instances();
  *proto_instance1->mutable_items() = instance.items();
  proto::ErrorMsg msg;
  auto proto_instance2 = reply.add_error_msg();
  *proto_instance2 = msg;

  std::shared_ptr<WorkerContext> worker_context = InitWorkerContext(&reply);
  Status status = thread.AddWorker(pid, worker_context);
  ASSERT_EQ(status.StatusCode(), SUCCESS);
  proto::PredictRequest request;
  auto proto_instance = request.add_instances();
  *proto_instance->mutable_items() = instance.items();
  request.mutable_servable_spec()->set_name("test_servable");
  request.mutable_servable_spec()->set_version_number(0);
  request.mutable_servable_spec()->set_method_name("add_cast");

  bool flag = false;
  PredictOnFinish callback = [&flag]() { flag = true; };
  status = thread.DispatchAsync(request, &reply, callback);
  ASSERT_EQ(status.StatusCode(), SUCCESS);
  ASSERT_EQ(status.StatusCode(), SUCCESS);
  ASSERT_EQ(flag, true);
  status = thread.DelWorker(pid);
  ASSERT_EQ(status.StatusCode(), SUCCESS);
}
}  // namespace serving
}  // namespace mindspore
