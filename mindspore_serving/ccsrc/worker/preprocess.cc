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

#include "worker/preprocess.h"
#include <utility>

namespace mindspore::serving {

void PreprocessStorage::Register(const std::string &preprocess_name, std::shared_ptr<PreprocessBase> preprocess) {
  if (preprocess_map_.find(preprocess_name) != preprocess_map_.end()) {
    MSI_LOG_WARNING << "preprocess " << preprocess_name << " has been registered";
  }
  preprocess_map_[preprocess_name] = std::move(preprocess);
}

PreprocessStorage &PreprocessStorage::Instance() {
  static PreprocessStorage storage;
  return storage;
}

std::shared_ptr<PreprocessBase> PreprocessStorage::GetPreprocess(const std::string &preprocess_name) const {
  auto it = preprocess_map_.find(preprocess_name);
  if (it != preprocess_map_.end()) {
    return it->second;
  }
  return nullptr;
}

RegPreprocess::RegPreprocess(const std::string &preprocess_name, std::shared_ptr<PreprocessBase> preprocess) {
  MSI_LOG_INFO << "Register C++ preprocess " << preprocess_name;
  PreprocessStorage::Instance().Register(preprocess_name, std::move(preprocess));
}

}  // namespace mindspore::serving
