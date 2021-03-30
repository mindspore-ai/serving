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

#include "master/master_context.h"

namespace mindspore::serving {

std::shared_ptr<MasterContext> MasterContext::Instance() {
  static std::shared_ptr<MasterContext> instance;
  if (instance == nullptr) {
    instance = std::make_shared<MasterContext>();
  }
  return instance;
}

void MasterContext::SetMaxEnqueuedRequests(uint32_t max_enqueued_requests) {
  max_enqueued_requests_ = max_enqueued_requests;
}

uint32_t MasterContext::GetMaxEnqueuedRequests() const { return max_enqueued_requests_; }

}  // namespace mindspore::serving
