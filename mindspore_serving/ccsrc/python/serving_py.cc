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

#include <string>
#include "python/worker/worker_py.h"
#include "python/worker/servable_py.h"
#include "python/tensor_py.h"
#include "common/servable.h"
#include "common/ssl_config.h"
#include "master/server.h"
#include "master/master_context.h"
#include "master/worker_context.h"
#include "worker/context.h"
#include "worker/stage_function.h"
#include "python/master/master_py.h"
#include "python/agent/agent_py.h"
#include "common/exit_handle.h"
#include "worker/distributed_worker/worker_agent.h"

namespace mindspore::serving {

void PyRegServable(pybind11::module *m_ptr) {
  auto &m = *m_ptr;
  // avoid as numpy object memory copy in PyTensor::AsPythonData
  py::class_<TensorBase, TensorBasePtr>(m, "Tensor_");

  py::class_<PyStageFunctionStorage, std::shared_ptr<PyStageFunctionStorage>>(m, "StageFunctionStorage_")
    .def(py::init<>())
    .def_static("get_instance", &PyStageFunctionStorage::Instance)
    .def("register", &PyStageFunctionStorage::Register)
    .def("get_pycpp_function_info", &PyStageFunctionStorage::GetPyCppFunctionInfo);

  py::class_<MethodSignature>(m, "MethodSignature_")
    .def(py::init<>())
    .def_readwrite("servable_name", &MethodSignature::servable_name)
    .def_readwrite("method_name", &MethodSignature::method_name)
    .def_readwrite("inputs", &MethodSignature::inputs)
    .def_readwrite("outputs", &MethodSignature::outputs)
    .def("add_stage_function", &MethodSignature::AddStageFunction)
    .def("add_stage_model", &MethodSignature::AddStageModel)
    .def("set_return", &MethodSignature::SetReturn);

  py::class_<RequestSpec>(m, "RequestSpec_")
    .def(py::init<>())
    .def_readwrite("servable_name", &RequestSpec::servable_name)
    .def_readwrite("version_number", &RequestSpec::version_number)
    .def_readwrite("method_name", &RequestSpec::method_name);

  py::class_<CommonModelMeta>(m, "CommonModelMeta_")
    .def(py::init<>())
    .def_readwrite("servable_name", &CommonModelMeta::servable_name)
    .def_readwrite("model_key", &CommonModelMeta::model_key)
    .def_readwrite("inputs_count", &CommonModelMeta::inputs_count)
    .def_readwrite("outputs_count", &CommonModelMeta::outputs_count)
    .def_readwrite("with_batch_dim", &CommonModelMeta::with_batch_dim)
    .def_readwrite("without_batch_dim_inputs", &CommonModelMeta::without_batch_dim_inputs);

  py::class_<LocalModelMeta>(m, "LocalModelMeta_")
    .def(py::init<>())
    .def_readwrite("model_file", &LocalModelMeta::model_files)
    .def_readwrite("config_file", &LocalModelMeta::config_file)
    .def_readwrite("model_context", &LocalModelMeta::model_context)
    .def("set_model_format", &LocalModelMeta::SetModelFormat);

  py::class_<ModelContext>(m, "ModelContext_")
    .def(py::init<>())
    .def_readwrite("thread_num", &ModelContext::thread_num)
    .def_readwrite("thread_affinity_core_list", &ModelContext::thread_affinity_core_list)
    .def_readwrite("enable_parallel", &ModelContext::enable_parallel)
    .def_readwrite("device_list", &ModelContext::device_list)
    .def("append_device_info", &ModelContext::AppendDeviceInfo);

  py::class_<DistributedModelMeta>(m, "DistributedModelMeta_")
    .def(py::init<>())
    .def_readwrite("rank_size", &DistributedModelMeta::rank_size)
    .def_readwrite("stage_size", &DistributedModelMeta::stage_size);

  py::class_<ModelMeta>(m, "ModelMeta_")
    .def(py::init<>())
    .def_readwrite("common_meta", &ModelMeta::common_meta)
    .def_readwrite("local_meta", &ModelMeta::local_meta)
    .def_readwrite("distributed_meta", &ModelMeta::distributed_meta);

  py::class_<ServableSignature>(m, "ServableSignature_")
    .def(py::init<>())
    .def_readwrite("servable_meta", &ServableSignature::model_metas)
    .def_readwrite("methods", &ServableSignature::methods);

  py::class_<PyServableRegister>(m, "ServableRegister_")
    .def_static("register_model_input_output_info", &PyServableRegister::RegisterInputOutputInfo)
    .def_static("register_method", &PyServableRegister::RegisterMethod)
    .def_static("declare_model", &PyServableRegister::DeclareModel)
    .def_static("declare_distributed_model", &PyServableRegister::DeclareDistributedModel)
    .def_static("run", &PyServableRegister::Run);

  py::class_<OneRankConfig>(m, "OneRankConfig_")
    .def(py::init<>())
    .def_readwrite("device_id", &OneRankConfig::device_id)
    .def_readwrite("ip", &OneRankConfig::ip);

  py::class_<DistributedServableConfig>(m, "DistributedServableConfig_")
    .def(py::init<>())
    .def_readwrite("common_meta", &DistributedServableConfig::common_meta)
    .def_readwrite("distributed_meta", &DistributedServableConfig::distributed_meta)
    .def_readwrite("rank_table_content", &DistributedServableConfig::rank_table_content)
    .def_readwrite("rank_list", &DistributedServableConfig::rank_list);
}

void PyRegMaster(pybind11::module *m_ptr) {
  auto &m = *m_ptr;
  py::class_<PyMaster>(m, "Master_")
    .def_static("start_grpc_server", &PyMaster::StartGrpcServer)
    .def_static("start_grpc_master_server", &PyMaster::StartGrpcMasterServer)
    .def_static("start_restful_server", &PyMaster::StartRestfulServer)
    .def_static("wait_and_clear", &PyMaster::WaitAndClear)
    .def_static("stop_and_clear", &PyMaster::StopAndClear)
    .def_static("only_model_stage", &PyMaster::OnlyModelStage);

  py::class_<WorkerContext, std::shared_ptr<WorkerContext>>(m, "WorkerContext_")
    .def_static("init_worker", &WorkerContext::PyInitWorkerContext)
    .def("has_error_notified", &WorkerContext::HasErrorNotified)
    .def("has_exit_notified", &WorkerContext::HasExitNotified)
    .def("get_notified_error", &WorkerContext::GetNotifiedError)
    .def("ready", &WorkerContext::HasReady)
    .def("print_status", &WorkerContext::PrintStatus)
    .def("is_in_starting", &WorkerContext::IsInStarting)
    .def("update_worker_pid", &WorkerContext::UpdateWorkerPid)
    .def("notify_not_alive", &WorkerContext::PyNotifyNotAlive)
    .def("notify_start_failed", &WorkerContext::PyNotifyStartFailed)
    .def_property_readonly("is_unavailable", &WorkerContext::IsUnavailable)
    .def_property_readonly("normal_handled_count", &WorkerContext::GetNormalHandledCount)
    .def_property_readonly("address", &WorkerContext::GetWorkerAddress);
  py::class_<SSLConfig>(m, "SSLConfig_")
    .def(py::init<>())
    .def_readwrite("certificate", &SSLConfig::certificate)
    .def_readwrite("private_key", &SSLConfig::private_key)
    .def_readwrite("custom_ca", &SSLConfig::custom_ca)
    .def_readwrite("verify_client", &SSLConfig::verify_client)
    .def_readwrite("use_ssl", &SSLConfig::use_ssl);
}

void PyRegWorker(pybind11::module *m_ptr) {
  auto &m = *m_ptr;
  py::class_<TaskItem>(m, "TaskItem_")
    .def(py::init<>())
    .def_readonly("has_stopped", &TaskItem::has_stopped)
    .def_property_readonly("method_name", [](const TaskItem &item) { return item.task_info.group_name; })
    .def_property_readonly("stage_index", [](const TaskItem &item) { return item.task_info.priority; })
    .def_property_readonly("task_name", [](const TaskItem &item) { return item.task_info.task_name; })
    .def_property_readonly("instance_list", [](const TaskItem &item) {
      py::tuple instances(item.instance_list.size());
      for (size_t i = 0; i < item.instance_list.size(); i++) {
        instances[i] = PyTensor::AsNumpyTuple(item.instance_list[i]->data);
      }
      return instances;
    });

  py::class_<PyWorker>(m, "Worker_")
    .def_static("start_servable", &PyWorker::StartServable, py::call_guard<py::gil_scoped_release>())
    .def_static("start_distributed_servable", &PyWorker::StartDistributedServable,
                py::call_guard<py::gil_scoped_release>())
    .def_static("start_extra_servable", &PyWorker::StartExtraServable, py::call_guard<py::gil_scoped_release>())
    .def_static("get_declared_model_names", &PyWorker::GetDeclaredModelNames)
    .def_static("wait_and_clear", &PyWorker::WaitAndClear)
    .def_static("stop_and_clear", PyWorker::StopAndClear)
    .def_static("enable_pytask_que", PyWorker::EnablePyTaskQueue)
    .def_static("get_py_task", &PyWorker::GetPyTask, py::call_guard<py::gil_scoped_release>())
    .def_static("push_pytask_result", &PyWorker::PushPyTaskResult)
    .def_static("push_pytask_failed", &PyWorker::PushPyTaskFailed)
    .def_static("push_pytask_system_failed", &PyWorker::PushPyTaskSystemFailed)
    .def_static("get_device_type", &PyWorker::GetDeviceType)
    .def_static("notify_failed", &PyWorker::NotifyFailed);

  py::class_<ServableContext, std::shared_ptr<ServableContext>>(m, "ServableContext_")
    .def(py::init<>())
    .def_static("get_instance", &ServableContext::Instance)
    .def("set_device_type_str",
         [](ServableContext &context, const std::string &device_type) {
           auto status = context.SetDeviceTypeStr(device_type);
           if (status != SUCCESS) {
             MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
           }
         })
    .def("set_device_id", &ServableContext::SetDeviceId);

  py::class_<MasterContext, std::shared_ptr<MasterContext>>(m, "MasterContext_")
    .def(py::init<>())
    .def_static("get_instance", &MasterContext::Instance)
    .def("set_max_enqueued_requests", &MasterContext::SetMaxEnqueuedRequests);
}

void PyRegWorkerAgent(pybind11::module *m_ptr) {
  auto &m = *m_ptr;
  py::class_<PyAgent>(m, "WorkerAgent_")
    .def_static("get_agents_config_from_worker", &PyAgent::GetAgentsConfigsFromWorker)
    .def_static("wait_and_clear", &PyAgent::WaitAndClear)
    .def_static("stop_and_clear", &PyAgent::StopAndClear)
    .def_static("notify_failed", &PyAgent::NotifyFailed)
    .def_static("startup_notify_exit", &PyAgent::StartupNotifyExit)
    .def_static("start_agent", &PyAgent::StartAgent);

  py::class_<AgentStartUpConfig>(m, "AgentStartUpConfig_")
    .def(py::init<>())
    .def_readwrite("rank_id", &AgentStartUpConfig::rank_id)
    .def_readwrite("device_id", &AgentStartUpConfig::device_id)
    .def_readwrite("model_file_names", &AgentStartUpConfig::model_file_names)
    .def_readwrite("group_file_names", &AgentStartUpConfig::group_file_names)
    .def_readwrite("rank_table_json_file_name", &AgentStartUpConfig::rank_table_json_file_name)
    .def_readwrite("agent_address", &AgentStartUpConfig::agent_address)
    .def_readwrite("distributed_address", &AgentStartUpConfig::distributed_address)
    .def_readwrite("common_meta", &AgentStartUpConfig::common_meta);
}

class PyExitSignalHandle {
 public:
  static void Start() { ExitSignalHandle::Instance().Start(); }
  static bool HasStopped() { return ExitSignalHandle::Instance().HasStopped(); }
};

// cppcheck-suppress syntaxError
PYBIND11_MODULE(_mindspore_serving, m) {
  PyRegServable(&m);
  PyRegMaster(&m);
  PyRegWorker(&m);
  PyRegWorkerAgent(&m);

  py::class_<PyExitSignalHandle>(m, "ExitSignalHandle_")
    .def_static("start", &PyExitSignalHandle::Start)
    .def_static("has_stopped", &PyExitSignalHandle::HasStopped);

  (void)py::module::import("atexit").attr("register")(py::cpp_function{[&]() -> void {
    Server::Instance().Clear();
    Worker::GetInstance().Clear();
    WorkerAgent::Instance().Clear();
  }});
}

}  // namespace mindspore::serving
