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
#include "python/worker/servable_py.h"
#include <string>

namespace mindspore::serving {

void PyServableStorage::RegisterMethod(const MethodSignature &method) {
  auto status = ServableStorage::Instance().RegisterMethod(method);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
}
void PyServableStorage::DeclareServable(const ServableMeta &servable) {
  auto status = ServableStorage::Instance().DeclareServable(servable);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
}
void PyServableStorage::DeclareDistributedServable(const ServableMeta &servable) {
  auto status = ServableStorage::Instance().DeclareDistributedServable(servable);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
}
void PyServableStorage::RegisterInputOutputInfo(const std::string &servable_name, size_t inputs_count,
                                                size_t outputs_count) {
  auto status = ServableStorage::Instance().RegisterInputOutputInfo(servable_name, inputs_count, outputs_count);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
}
void PyServableStorage::Clear() { ServableStorage::Instance().Clear(); }
}  // namespace mindspore::serving
