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

#ifndef MINDSPORE_SERVER_MASTER_PY_H
#define MINDSPORE_SERVER_MASTER_PY_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <memory>
#include "common/serving_common.h"
#include "common/ssl_config.h"

namespace py = pybind11;

namespace mindspore {
namespace serving {

class MS_API PyMaster {
 public:
  static void StartGrpcServer(const std::string &socket_address, const SSLConfig &ssl_config,
                              int max_msg_mb_size = 100);
  static void StartGrpcMasterServer(const std::string &master_address);
  static void StartRestfulServer(const std::string &socket_address, const SSLConfig &ssl_config,
                                 int max_msg_mb_size = 100);
  static void WaitAndClear();
  static void StopAndClear();
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVER_MASTER_PY_H
