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

#include "worker/postprocess.h"
#include <utility>

namespace mindspore::serving {

bool PostprocessStorage::Register(const std::string &postprocess_name, std::shared_ptr<PostprocessBase> postprocess) {
  if (postprocess_map_.find(postprocess_name) != postprocess_map_.end()) {
    MSI_LOG_WARNING << "postprocess " << postprocess_name << " has been registered";
    return false;
  }
  postprocess_map_[postprocess_name] = std::move(postprocess);
  return true;
}

void PostprocessStorage::Unregister(const std::string &postprocess_name) {
  auto it = postprocess_map_.find(postprocess_name);
  if (it == postprocess_map_.end()) {
    return;
  }
  postprocess_map_.erase(it);
}

std::shared_ptr<PostprocessBase> PostprocessStorage::GetPostprocess(const std::string &postprocess_name) const {
  auto it = postprocess_map_.find(postprocess_name);
  if (it != postprocess_map_.end()) {
    return it->second;
  }
  return nullptr;
}

PostprocessStorage &PostprocessStorage::Instance() {
  static PostprocessStorage storage;
  return storage;
}

RegPostprocess::RegPostprocess(const std::string &postprocess_name, std::shared_ptr<PostprocessBase> postprocess) {
  postprocess_name_ = postprocess_name;
  MSI_LOG_INFO << "Register C++ postprocess " << postprocess_name;
  register_success_ = PostprocessStorage::Instance().Register(postprocess_name, std::move(postprocess));
}

RegPostprocess::~RegPostprocess() {
  if (register_success_) {
    MSI_LOG_INFO << "Unregister C++ preprocess " << postprocess_name_;
    PostprocessStorage::Instance().Unregister(postprocess_name_);
  }
}

}  // namespace mindspore::serving
