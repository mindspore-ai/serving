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

#ifndef MINDSPORE_SERVING_MASTER_SERVABLE_ENDPOINT_H
#define MINDSPORE_SERVING_MASTER_SERVABLE_ENDPOINT_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <shared_mutex>
#include <map>
#include "common/serving_common.h"
#include "master/worker_context.h"
#include "master/model_thread.h"

namespace mindspore::serving {

// visit by dispatcher
class ServableEndPoint {
 public:
  explicit ServableEndPoint(const ServableReprInfo &repr);
  ~ServableEndPoint();
  Status DispatchAsync(const proto::PredictRequest &request, proto::PredictReply *reply, PredictOnFinish on_finish);

  Status RegisterWorker(const ServableRegSpec &servable_spec, std::shared_ptr<WorkerContext> worker);
  Status UnregisterWorker(const std::string &worker_address);
  void Clear();

  std::string GetServableName() const { return worker_repr_.servable_name; }
  uint64_t GetVersionNumber() const { return version_number_; }
  std::vector<ServableMethodInfo> GetMethods() const { return methods_; }

 private:
  std::map<std::string, std::shared_ptr<ModelThread>> model_thread_list_;
  ServableReprInfo worker_repr_;
  std::vector<ServableMethodInfo> methods_;
  std::vector<std::shared_ptr<WorkerContext>> worker_contexts_;
  uint32_t version_number_ = 0;
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_MASTER_SERVABLE_ENDPOINT_H
