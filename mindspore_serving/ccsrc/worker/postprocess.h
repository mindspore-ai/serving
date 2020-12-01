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

#ifndef MINDSPORE_SERVING_POSTPROCESS_H
#define MINDSPORE_SERVING_POSTPROCESS_H

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>

#include "common/serving_common.h"
#include "common/instance.h"

namespace mindspore::serving {

class PostprocessBase : public std::enable_shared_from_this<PostprocessBase> {
 public:
  PostprocessBase() = default;
  virtual ~PostprocessBase() = default;

  virtual Status Postprocess(const std::string &postprocess_name, const InstanceData &input, InstanceData *output) = 0;
  virtual size_t GetInputsCount(const std::string &postprocess_name) const = 0;
  virtual size_t GetOutputsCount(const std::string &postprocess_name) const = 0;
  virtual bool IsPythonPostprocess() const { return false; }
};

class MS_API PostprocessStorage {
 public:
  void Register(const std::string &postprocess_name, std::shared_ptr<PostprocessBase> postprocess);

  std::shared_ptr<PostprocessBase> GetPostprocess(const std::string &postprocess_name) const;

  static PostprocessStorage &Instance();

 private:
  std::unordered_map<std::string, std::shared_ptr<PostprocessBase>> postprocess_map_;
};

class RegPostprocess {
 public:
  RegPostprocess(const std::string &postprocess_name, std::shared_ptr<PostprocessBase> postprocess);
};

#define REGISTER_POSTPROCESS(cls_name, postprocess_name) \
  static RegPostprocess g_register_postprocess_##cls_name(postprocess_name, std::make_shared<cls_name>());

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_POSTPROCESS_H
