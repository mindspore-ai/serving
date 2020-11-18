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

#include "worker/notfiy_master/local_notify.h"
#include "master/server.h"
namespace mindspore {
namespace serving {

Status LocalNotifyMaster::Register(const std::vector<WorkerSpec> &worker_specs) {
  return Server::Instance().GetDispatcher()->RegisterLocalServable(worker_specs);
}

Status LocalNotifyMaster::Unregister() { return Server::Instance().GetDispatcher()->UnregisterLocalServable(); }

Status LocalNotifyMaster::AddWorker(const WorkerSpec &worker_spec) {
  return Server::Instance().GetDispatcher()->AddLocalServable(worker_spec);
}

Status LocalNotifyMaster::RemoveWorker(const WorkerSpec &worker_spec) {
  return Server::Instance().GetDispatcher()->RemoveLocalServable(worker_spec);
}

}  // namespace serving
}  // namespace mindspore
