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
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "mindspore_serving/client/cpp/client.h"

using namespace mindspore::serving::client;

struct Options {
  std::string ip = "localhost";
  int64_t port = 0;
  std::string img_dir;
};

Options g_options;

std::vector<std::string> GetAllFiles(const std::string &dir_path) {
  DIR *dir = nullptr;
  struct dirent *ptr = nullptr;
  std::vector<std::string> files;

  if ((dir = opendir(dir_path.c_str())) == nullptr) {
    return std::vector<std::string>();
  }

  while ((ptr = readdir(dir)) != nullptr) {
    std::string name = ptr->d_name;
    if (name == "." || name == "..") {
      continue;
    }
    if (ptr->d_type == DT_REG) {
      files.push_back(name);
    }
  }
  closedir(dir);
  std::sort(files.begin(), files.end());
  return files;
}

template <class DT>
std::string VectorAsString(const std::vector<DT> &list) {
  std::stringstream stringstream;
  stringstream << "[";
  for (size_t i = 0; i < list.size(); i++) {
    stringstream << list[i];
    if (i != list.size() - 1) {
      stringstream << ", ";
    }
  }
  stringstream << "]";
  return stringstream.str();
}

void PrintResultTensor(const Tensor &result_tensor) {
  if (!result_tensor.IsValid()) {
    std::cout << "Get result failed" << std::endl;
    return;
  }
  auto data_type = result_tensor.GetDataType();
  switch (data_type) {
    case mindspore::serving::client::DT_INT32: {
      std::vector<int32_t> result_label;
      result_tensor.GetData(&result_label);
      std::cout << "result: " << VectorAsString(result_label) << std::endl;
      break;
    }
    case mindspore::serving::client::DT_INT64: {
      std::vector<int64_t> result_label;
      result_tensor.GetData(&result_label);
      std::cout << "result: " << VectorAsString(result_label) << std::endl;
      break;
    }
    case mindspore::serving::client::DT_STRING: {
      std::string result_label;
      result_tensor.GetStrData(&result_label);
      std::cout << "result: " << result_label << std::endl;
      break;
    }
    default:
      std::cout << "Unreginized data type " << data_type << std::endl;
      break;
  }
}

void GetImageBuffer(std::vector<std::vector<uint8_t>> &img_bytes) {
  std::string dir_path = g_options.img_dir;
  auto files = GetAllFiles(dir_path);
  for (auto &file : files) {
    std::ifstream fp(dir_path + file, std::ios::binary);
    if (!fp.is_open()) {
      continue;
    }
    fp.seekg(0, std::ios_base::end);
    auto file_len = fp.tellg();
    fp.seekg(0, std::ios_base::beg);
    img_bytes.emplace_back(std::vector<uint8_t>(file_len));
    auto &img_buffer = img_bytes.back();
    fp.read(reinterpret_cast<char *>(img_buffer.data()), img_buffer.size());
    std::cout << file << ", " << file_len << std::endl;
  }
}

void RunInstances() {
  std::vector<std::vector<uint8_t>> img_bytes;
  GetImageBuffer(img_bytes);

  Client client(g_options.ip, g_options.port, "lenet", "predict");
  InstancesRequest request;
  InstancesReply reply;
  for (auto &img : img_bytes) {
    auto instance = request.AddInstance();
    auto instance_input = instance.Add("image");
    instance_input.SetBytesData(img);
  }
  auto status = client.SendRequest(request, &reply);
  if (!status.IsSuccess()) {
    std::cout << "Instances Mode: Send request failed, failed detail: " << status.StatusMessage() << std::endl;
    return;
  }
  const auto &result = reply.GetResult();
  for (auto &instance : result) {
    if (!instance.IsValid()) {
      std::cout << "Inputs Mode: Get result failed" << std::endl;
      return;
    }
    int64_t error_code;
    std::string error_msg;
    if (instance.HasErrorMsg(&error_code, &error_msg)) {
      std::cout << "Error: " << error_msg << std::endl;
      continue;
    }
    auto result_tensor = instance.Get("result");
    PrintResultTensor(result_tensor);
  }
}

int main(int argc, char **argv) {
  auto usage = []() {
    std::cout << "Usage: " << std::endl;
    std::cout << "--ip={ip}, optional, default localhost" << std::endl;
    std::cout << "--port={port}, optional, default 5500" << std::endl;
    std::cout << "--img_dir={img_dir}, required" << std::endl;
  };
  if (argc > 1) {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if (arg.find("--ip=") == 0) {
        g_options.ip = arg.substr(std::string("--ip=").length());
      } else if (arg.find("--port=") == 0) {
        g_options.port = std::stoi(arg.substr(std::string("--port=").length()));
      } else if (arg.find("--img_dir=") == 0) {
        g_options.img_dir = arg.substr(std::string("--img_dir=").length());
        auto start = g_options.img_dir.find_first_not_of('\"');
        auto end = g_options.img_dir.find_last_of('\"');
        if (start != 0 || end != std::string::npos) {
          g_options.img_dir =
            g_options.img_dir.substr(start, end != std::string::npos ? end - start : std::string::npos);
        }
      } else {
        usage();
        return -1;
      }
    }
  }
  if (g_options.img_dir.empty()) {
    std::cout << "Expect image dir" << std::endl;
    usage();
    return -1;
  }
  std::cout << "Image dir: " << g_options.img_dir << ", ip: " << g_options.ip << ", port: " << g_options.port
            << std::endl;
  RunInstances();
  return 0;
}
