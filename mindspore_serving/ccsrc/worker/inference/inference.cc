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
#include "worker/inference/inference.h"
#include <dlfcn.h>
#include "glog/logging.h"

namespace mindspore::serving {
namespace {
constexpr const char *kMindSporeLibName = "libmindspore.so";
constexpr const char *kMindsporeLiteLibName = "libmindspore-lite.so";
constexpr const char *kServingAscendLibName = "libserving_ascend.so";
}  // namespace

void ModelContext::AppendDeviceInfo(const DeviceInfo &device_info) { (void)device_list.emplace_back(device_info); }

std::string ModelContext::AsString() const {
  std::map<std::string, std::string> output_map;
  if (thread_num > -1) {
    output_map["thread num"] = AsStringHelper::AsString(thread_num);
  }
  if (!thread_affinity_core_list.empty()) {
    output_map["thread affinity core list"] = AsStringHelper::AsString(thread_affinity_core_list);
  }
  if (enable_parallel > -1) {
    output_map["enable parallel"] = AsStringHelper::AsString(enable_parallel);
  }
  if (!device_list.empty()) {
    output_map["device infos"] = AsStringHelper::AsString(device_list);
  }
  return AsStringHelper::AsString(output_map);
}

InferenceLoader::InferenceLoader() {}
InferenceLoader::~InferenceLoader() {
  if (ms_lib_handle_ != nullptr) {
    (void)dlclose(ms_lib_handle_);
  }
  if (ms_cxx_lib_handle_ != nullptr) {
    (void)dlclose(ms_cxx_lib_handle_);
  }
  if (gomp_handler_ != nullptr) {
    (void)dlclose(gomp_handler_);
  }
}

InferenceLoader &InferenceLoader::Instance() {
  static InferenceLoader inference;
  return inference;
}

std::shared_ptr<InferenceBase> InferenceLoader::CreateMindSporeInfer() {
  Status status;
  if (ms_lib_handle_ == nullptr) {
    status = LoadMindSporeModelWrap();
    if (status != SUCCESS) {
      MSI_LOG_EXCEPTION << "Load " << kServingAscendLibName << " failed, error msg: " << status.StatusMessage();
    }
  }
  auto instance = ms_create_handle_();
  if (instance == nullptr) {
    return nullptr;
  } else {
    return std::shared_ptr<InferenceBase>(instance);
  }
}

std::vector<std::string> SplitString(const std::string &s, const std::string &delimiters = ":") {
  auto pos_left = s.find_first_not_of(delimiters, 0);
  auto pos_right = s.find_first_of(delimiters, pos_left);
  std::vector<std::string> tokens;
  while (pos_left != std::string::npos) {
    if (pos_right == std::string::npos) {
      tokens.push_back(s.substr(pos_left));
      break;
    }
    tokens.push_back(s.substr(pos_left, pos_right - pos_left));
    pos_left = s.find_first_not_of(delimiters, pos_right);
    pos_right = s.find_first_of(delimiters, pos_left);
  }
  return tokens;
}

Status InferenceLoader::LoadMindSporeModelWrap() {
  MSI_LOG_INFO << "Start Initialize MindSpore Model Wrap so";
  std::vector<std::string> gomp_list = {"libgomp.so.1"};
  for (auto &item : gomp_list) {
    gomp_handler_ = dlopen(item.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (gomp_handler_ != nullptr) {
      MSI_LOG_INFO << "dlopen libgomp so: " << item << " success";
    }
  }
  if (gomp_handler_ == nullptr) {
    MSI_LOG_WARNING << "dlopen libgomp library failed, try dlopen list: " << gomp_list;
  }

  auto get_dlerror = []() -> std::string {
    auto error = dlerror();
    if (error == nullptr) {
      return std::string();
    }
    return error;
  };

  auto ld_lib_path = common::GetEnv("LD_LIBRARY_PATH");
  MSI_LOG_INFO << "LD_LIBRARY_PATH: " << ld_lib_path;
  ms_cxx_lib_handle_ = dlopen(kMindsporeLiteLibName, RTLD_NOW | RTLD_GLOBAL);
  if (ms_cxx_lib_handle_ == nullptr) {
    std::string load_error = get_dlerror();
    std::string so_no_exist_error =
      std::string(kMindsporeLiteLibName) + ": cannot open shared object file: No such file or directory";
    // libmindspore-lite.so exist but dlopen failed
    if (load_error.find(so_no_exist_error) == std::string::npos) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "dlopen libmindspore-lite.so failed, dlopen error: " << load_error;
    }
    MSI_LOG_WARNING
      << "dlopen libmindspore_lite.so failed, if you want to use MindSpore Lite to do the inference, please append "
         "libmindspore-lite.so's path to LD_LIBRARY_PATH env or put it in the dynamic_library search path"
      << ", dlopen error: " << load_error;
    if (!ld_lib_path.empty()) {
      auto ms_search_path_list = SplitString(ld_lib_path, ":");
      MSI_LOG_INFO << "Search " << kMindSporeLibName << " directory: " << ms_search_path_list;
      for (auto &item : ms_search_path_list) {
        auto lib_path = item + "/" + kMindSporeLibName;
        if (!common::DirOrFileExist(lib_path)) {
          continue;
        }
        ms_cxx_lib_handle_ = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (ms_cxx_lib_handle_ == nullptr) {
          return INFER_STATUS_LOG_ERROR(FAILED) << "dlopen libmindspore.so failed, please check whether the MindSpore "
                                                   "and Ascend/GPU software package versions match"
                                                << ", lib path:" << lib_path << ", dlopen error: " << get_dlerror();
        }
        enable_lite_ = false;
        MSI_LOG_INFO << "Load " << kMindSporeLibName << " in " << item << " successful";
        break;
      }
    }
  } else {
    MSI_LOG_INFO << "Load " << kMindsporeLiteLibName << " successful";
    enable_lite_ = true;
  }

  if (ms_cxx_lib_handle_ == nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Failed to load libmindspore.so and libmindspore-lite.so, please set env LD_LIBRARY_PATH for "
           << "libmindspore-lite.so or pip install MindSpore whl package for libmindspore.so";
  }

  ms_lib_handle_ = dlopen(kServingAscendLibName, RTLD_NOW | RTLD_GLOBAL);
  if (ms_lib_handle_ == nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "dlopen failed, please check whether the MindSpore and Serving versions match, lib name:"
           << kServingAscendLibName << ", dlopen error: " << get_dlerror();
  }
  MSI_LOG_INFO << "Load " << kServingAscendLibName << " successful";
  ms_create_handle_ = (CreateInferHandle)dlsym(ms_lib_handle_, "ServingCreateInfer");
  if (ms_create_handle_ == nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "dlsym ServingCreateInfer failed, lib name:" << kServingAscendLibName
                                          << ", dlopen error: " << get_dlerror();
  }
  return SUCCESS;
}

bool InferenceLoader::GetEnableLite() const { return enable_lite_; }

DeviceType InferenceLoader::GetSupportDeviceType(DeviceType device_type, ModelType model_type) {
  auto mindspore_infer = CreateMindSporeInfer();
  if (mindspore_infer == nullptr) {
    MSI_LOG_ERROR << "Create MindSpore infer failed";
    return kDeviceTypeNotSpecified;
  }
  std::vector<ModelType> check_model_types;
  if (model_type == kUnknownType) {
    check_model_types = {kMindIR, kMindIR_Lite, kOM};
  } else {
    check_model_types = {model_type};
  }
  for (auto &model_type_item : check_model_types) {
    if (device_type == kDeviceTypeNotSpecified) {
      auto device_list = {kDeviceTypeAscend, kDeviceTypeGpu, kDeviceTypeCpu};
      for (auto item : device_list) {
        if (mindspore_infer->CheckModelSupport(item, model_type_item)) {
          return item;
        }
      }
    } else {
      if (mindspore_infer->CheckModelSupport(device_type, model_type_item)) {
        return device_type;
      }
    }
  }
  return kDeviceTypeNotSpecified;
}

bool InferenceLoader::SupportReuseDevice() {
  auto mindspore_infer = CreateMindSporeInfer();
  if (mindspore_infer == nullptr) {
    MSI_LOG_ERROR << "Create MindSpore infer failed";
    return false;
  }
  return mindspore_infer->SupportReuseDevice();
}
}  // namespace mindspore::serving
