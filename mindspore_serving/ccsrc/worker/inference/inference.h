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

#ifndef MINDSPORE_SERVING_WORKER_INFERENCE_H
#define MINDSPORE_SERVING_WORKER_INFERENCE_H

#include <utility>
#include <unordered_map>
#include <memory>
#include <vector>
#include <string>
#include "common/tensor_base.h"
#include "common/tensor.h"
#include "common/log.h"
#include "common/status.h"
#include "include/api/types.h"

namespace mindspore {
namespace serving {

using api::ModelType;
using api::ModelType::kMindIR;
using api::ModelType::kOM;

struct TensorInfo {
  size_t size;  // -1: unspecified
  DataType data_type;
  std::vector<int64_t> shape;
};

enum DeviceType {
  kDeviceTypeNotSpecified,
  kDeviceTypeAscendMS,
  kDeviceTypeAscendCL,
  kDeviceTypeAscend,
  kDeviceTypeGpu,
  kDeviceTypeCpu,
};

class MS_API InferSession {
 public:
  InferSession() = default;
  virtual ~InferSession() = default;
  virtual Status InitEnv(DeviceType device_type, uint32_t device_id,
                         const std::unordered_map<std::string, std::string> &other_options) = 0;
  virtual Status FinalizeEnv() = 0;

  virtual Status LoadModelFromFile(serving::DeviceType device_type, uint32_t device_id, const std::string &file_name,
                                   ModelType model_type, uint32_t *model_id) = 0;
  virtual Status UnloadModel(uint32_t model_id) = 0;
  // override this method to avoid request/reply data copy
  virtual Status ExecuteModel(uint32_t model_id, const RequestBase &request, ReplyBase *reply) = 0;
  virtual Status ExecuteModel(uint32_t model_id, const std::vector<TensorBasePtr> &request,
                              std::vector<TensorBasePtr> *reply) {
    VectorTensorPtrWrapRequest wrap_request(request);
    VectorTensorPtrWrapReply wrap_reply(reply, []() { return std::make_shared<Tensor>(); });
    return ExecuteModel(model_id, wrap_request, &wrap_reply);
  }

  virtual std::vector<TensorInfo> GetInputInfos(uint32_t model_id) const = 0;
  virtual std::vector<TensorInfo> GetOutputInfos(uint32_t model_id) const = 0;
  virtual ssize_t GetBatchSize(uint32_t model_id) const = 0;
  virtual TensorBasePtr MakeInferenceTensor(DataType data_type, const std::vector<int64_t> &shape) const {
    return nullptr;
  }
  virtual bool CheckModelSupport(DeviceType device_type, ModelType model_type) const { return true; }
};

struct InferSessionRegInfo {
  std::shared_ptr<InferSession> session;
  ModelType model_type;
  int priority;
};

class MS_API InferSessionStorage {
 public:
  void Register(DeviceType device_type, ModelType model_type, const std::shared_ptr<InferSession> &session,
                int priority) {
    auto &list = session_map_[device_type];
    InferSessionRegInfo info{session, model_type, priority};
    list.push_back(info);
  }

  std::shared_ptr<InferSession> Get(DeviceType device_type, ModelType model_type, DeviceType *specified_device_type) {
    MSI_EXCEPTION_IF_NULL(specified_device_type);
    if (device_type == kDeviceTypeNotSpecified) {
      for (auto &item_device : session_map_) {
        std::shared_ptr<InferSession> ret_session = GetSession(item_device.second, item_device.first, model_type);
        if (ret_session) {
          *specified_device_type = item_device.first;
          return ret_session;
        }
      }
      return nullptr;
    } else if (device_type == kDeviceTypeAscend) {
      auto ascend_list = {kDeviceTypeAscendCL, kDeviceTypeAscendMS};
      for (auto ascend_type : ascend_list) {
        auto it = session_map_.find(ascend_type);
        if (it == session_map_.end()) {
          continue;
        }
        auto session_ret = GetSession(it->second, ascend_type, model_type);
        if (session_ret != nullptr) {
          *specified_device_type = ascend_type;
          return session_ret;
        }
      }
      return nullptr;
    }
    auto it = session_map_.find(device_type);
    if (it == session_map_.end()) {
      return nullptr;
    }
    std::shared_ptr<InferSession> session_ret;
    session_ret = GetSession(it->second, device_type, model_type);
    *specified_device_type = device_type;
    return session_ret;
  }

  static InferSessionStorage &Instance() {
    static InferSessionStorage instance;
    return instance;
  }

 private:
  std::unordered_map<DeviceType, std::vector<InferSessionRegInfo>> session_map_;

  std::shared_ptr<InferSession> GetSession(const std::vector<InferSessionRegInfo> &session_list, DeviceType device_type,
                                           ModelType model_type) {
    std::shared_ptr<InferSession> session_ret = nullptr;
    int cur_priority = INT32_MIN;
    for (auto &item : session_list) {
      if (item.model_type != model_type) {
        continue;
      }
      if (session_ret == nullptr || cur_priority < item.priority) {
        if (!item.session->CheckModelSupport(device_type, model_type)) {
          MSI_LOG_INFO << "CheckModelSupport for " << device_type << " " << model_type << " failed, skipped";
          continue;
        }
        cur_priority = item.priority;
        session_ret = item.session;
      }
    }
    return session_ret;
  }
};

class MS_API InferSessionRegister {
 public:
  InferSessionRegister(DeviceType device_type, ModelType model_type, const std::shared_ptr<InferSession> &session,
                       int priority) {
    InferSessionStorage::Instance().Register(device_type, model_type, session, priority);
  }
};

#define REGISTER_INFER_SEESION_UNIQUE(device_type, model_type, cls_name, priority, index)  \
  static mindspore::serving::InferSessionRegister g_register_session_##cls_name##_##index( \
    device_type, model_type, std::make_shared<cls_name>(), priority);

#define REGISTER_INFER_SEESION_HELPER(device_type, model_type, cls_name, priority, index) \
  REGISTER_INFER_SEESION_UNIQUE(device_type, model_type, cls_name, priority, index)

#define REGISTER_INFER_SEESION(device_type, model_type, cls_name, priority) \
  REGISTER_INFER_SEESION_HELPER(device_type, model_type, cls_name, priority, __COUNTER__);

static inline LogStream &operator<<(LogStream &stream, DeviceType device_type) {
  switch (device_type) {
    case kDeviceTypeAscend:
      stream << "kDeviceTypeAscend";
      break;
    case kDeviceTypeAscendMS:
      stream << "kDeviceTypeAscend910";
      break;
    case kDeviceTypeAscendCL:
      stream << "kDeviceTypeAscend310";
      break;
    case kDeviceTypeGpu:
      stream << "kDeviceTypeGpu";
      break;
    case kDeviceTypeCpu:
      stream << "kDeviceTypeCpu";
      break;
    case kDeviceTypeNotSpecified:
      stream << "kDeviceTypeNotSpecified";
      break;
    default:
      stream << "[device type " << static_cast<int>(device_type) << "]";
      break;
  }
  return stream;
}

}  // namespace serving
}  // namespace mindspore
#endif  // MINDSPORE_SERVING_WORKER_INFERENCE_H
