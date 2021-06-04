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

#include "python/master/master_py.h"
#include "common/exit_handle.h"
#include "master/server.h"

namespace mindspore::serving {

void PyMaster::StartGrpcServer(const std::string &socket_address, const SSLConfig &ssl_config, int max_msg_mb_size) {
  auto status = Server::Instance().StartGrpcServer(socket_address, ssl_config, max_msg_mb_size);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
}

void PyMaster::StartGrpcMasterServer(const std::string &master_address) {
  auto status = Server::Instance().StartGrpcMasterServer(master_address);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
}

void PyMaster::StartRestfulServer(const std::string &socket_address, const SSLConfig &ssl_config, int max_msg_mb_size) {
  auto status = Server::Instance().StartRestfulServer(socket_address, ssl_config, max_msg_mb_size);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
}

void PyMaster::WaitAndClear() {
  {
    py::gil_scoped_release release;
    ExitSignalHandle::Instance().MasterWait();
  }
  Server::Instance().Clear();
  MSI_LOG_INFO << "Python server end wait and clear";
}

void PyMaster::StopAndClear() {
  ExitSignalHandle::Instance().Stop();
  Server::Instance().Clear();
}

}  // namespace mindspore::serving
