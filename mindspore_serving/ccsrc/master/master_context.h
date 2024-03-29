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

#ifndef MINDSPORE_SERVING_MASTER_CONTEXT_H
#define MINDSPORE_SERVING_MASTER_CONTEXT_H

#include <string>
#include <memory>
#include <vector>
#include "common/serving_common.h"

namespace mindspore::serving {
class MS_API MasterContext {
 public:
  static std::shared_ptr<MasterContext> Instance();

  void SetMaxEnqueuedRequests(uint32_t max_enqueued_requests);
  uint32_t GetMaxEnqueuedRequests() const;

 private:
  uint32_t max_enqueued_requests_ = 10000;  // default 10000
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_MASTER_CONTEXT_H
