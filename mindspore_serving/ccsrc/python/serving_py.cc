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
#include "python/worker/preprocess_py.h"
#include "python/worker/postprocess_py.h"
#include "python/worker/worker_py.h"
#include "python/worker/servable_py.h"
#include "python/tensor_py.h"
#include "common/servable.h"
#include "common/ssl_config.h"
#include "master/server.h"
#include "master/master_context.h"
#include "master/worker_context.h"
#include "worker/context.h"
#include "python/master/master_py.h"
#include "python/agent/agent_py.h"
#include "common/exit_handle.h"
#include "worker/distributed_worker/worker_agent.h"

namespace mindspore::serving {

void PyRegServable(pybind11::module *m_ptr) {
  auto &m = *m_ptr;
  // avoid as numpy object memory copy in PyTensor::AsPythonData
  py::class_<TensorBase, TensorBasePtr>(m, "Tensor_");

  py::class_<PyPreprocessStorage, std::shared_ptr<PyPreprocessStorage>>(m, "PreprocessStorage_")
    .def(py::init<>())
    .def_static("get_instance", &PyPreprocessStorage::Instance)
    .def("register", &PyPreprocessStorage::Register)
    .def("get_pycpp_preprocess_info", &PyPreprocessStorage::GetPyCppPreprocessInfo);

  py::class_<PyPostprocessStorage, std::shared_ptr<PyPostprocessStorage>>(m, "PostprocessStorage_")
    .def(py::init<>())
    .def_static("get_instance", &PyPostprocessStorage::Instance)
    .def("register", &PyPostprocessStorage::Register)
    .def("get_pycpp_postprocess_info", &PyPostprocessStorage::GetPyCppPostprocessInfo);

  py::enum_<PredictPhaseTag>(m, "PredictPhaseTag_")
    .value("kPredictPhaseTag_Input", PredictPhaseTag::kPredictPhaseTag_Input)
    .value("kPredictPhaseTag_Preproces", PredictPhaseTag::kPredictPhaseTag_Preproces)
    .value("kPredictPhaseTag_Predict", PredictPhaseTag::kPredictPhaseTag_Predict)
    .value("kPredictPhaseTag_Postprocess", PredictPhaseTag::kPredictPhaseTag_Postprocess)
    .export_values();

  py::class_<MethodSignature>(m, "MethodSignature_")
    .def(py::init<>())
    .def_readwrite("method_name", &MethodSignature::method_name)
    .def_readwrite("inputs", &MethodSignature::inputs)
    .def_readwrite("outputs", &MethodSignature::outputs)
    .def_readwrite("preprocess_name", &MethodSignature::preprocess_name)
    .def_readwrite("preprocess_inputs", &MethodSignature::preprocess_inputs)
    .def_readwrite("postprocess_name", &MethodSignature::postprocess_name)
    .def_readwrite("postprocess_inputs", &MethodSignature::postprocess_inputs)
    .def_readwrite("servable_name", &MethodSignature::servable_name)
    .def_readwrite("servable_inputs", &MethodSignature::servable_inputs)
    .def_readwrite("returns", &MethodSignature::returns);

  py::class_<RequestSpec>(m, "RequestSpec_")
    .def(py::init<>())
    .def_readwrite("servable_name", &RequestSpec::servable_name)
    .def_readwrite("version_number", &RequestSpec::version_number)
    .def_readwrite("method_name", &RequestSpec::method_name);

  py::class_<CommonServableMeta>(m, "CommonServableMeta_")
    .def(py::init<>())
    .def_readwrite("servable_name", &CommonServableMeta::servable_name)
    .def_readwrite("inputs_count", &CommonServableMeta::inputs_count)
    .def_readwrite("outputs_count", &CommonServableMeta::outputs_count)
    .def_readwrite("with_batch_dim", &CommonServableMeta::with_batch_dim)
    .def_readwrite("without_batch_dim_inputs", &CommonServableMeta::without_batch_dim_inputs);

  py::class_<LocalServableMeta>(m, "LocalServableMeta_")
    .def(py::init<>())
    .def_readwrite("servable_file", &LocalServableMeta::servable_file)
    .def_readwrite("options", &LocalServableMeta::load_options)
    .def("set_model_format", &LocalServableMeta::SetModelFormat);

  py::class_<DistributedServableMeta>(m, "DistributedServableMeta_")
    .def(py::init<>())
    .def_readwrite("rank_size", &DistributedServableMeta::rank_size)
    .def_readwrite("stage_size", &DistributedServableMeta::stage_size);

  py::class_<ServableMeta>(m, "ServableMeta_")
    .def(py::init<>())
    .def_readwrite("common_meta", &ServableMeta::common_meta)
    .def_readwrite("local_meta", &ServableMeta::local_meta)
    .def_readwrite("distributed_meta", &ServableMeta::distributed_meta);

  py::class_<ServableSignature>(m, "ServableSignature_")
    .def(py::init<>())
    .def_readwrite("servable_meta", &ServableSignature::servable_meta)
    .def_readwrite("methods", &ServableSignature::methods);

  py::class_<PyServableStorage>(m, "ServableStorage_")
    .def_static("register_servable_input_output_info", &PyServableStorage::RegisterInputOutputInfo)
    .def_static("register_method", &PyServableStorage::RegisterMethod)
    .def_static("declare_servable", &PyServableStorage::DeclareServable)
    .def_static("declare_distributed_servable", &PyServableStorage::DeclareDistributedServable);

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
    .def_static("stop_and_clear", &PyMaster::StopAndClear);

  py::class_<ServableStartConfig>(m, "ServableStartConfig_")
    .def(py::init<>())
    .def_readwrite("servable_directory", &ServableStartConfig::servable_directory)
    .def_readwrite("servable_name", &ServableStartConfig::servable_name)
    .def_readwrite("config_version_number", &ServableStartConfig::config_version_number)
    .def_readwrite("device_type", &ServableStartConfig::device_type)
    .def_readwrite("device_ids", &ServableStartConfig::device_ids);

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
  py::class_<TaskContext>(m, "TaskContext_").def(py::init<>());

  py::class_<TaskItem>(m, "TaskItem_")
    .def(py::init<>())
    .def_readwrite("task_type", &TaskItem::task_type)
    .def_readwrite("name", &TaskItem::name)
    .def_property_readonly("instance_list",
                           [](const TaskItem &item) {
                             py::tuple instances(item.instance_list.size());
                             for (size_t i = 0; i < item.instance_list.size(); i++) {
                               instances[i] = PyTensor::AsNumpyTuple(item.instance_list[i]->data);
                             }
                             return instances;
                           })
    .def_readwrite("context_list", &TaskItem::context_list);

  py::class_<PyWorker>(m, "Worker_")
    .def_static("start_servable", &PyWorker::StartServable)
    .def_static("start_distributed_servable", &PyWorker::StartDistributedServable)
    .def_static("get_batch_size", &PyWorker::GetBatchSize)
    .def_static("wait_and_clear", &PyWorker::WaitAndClear)
    .def_static("stop_and_clear", PyWorker::StopAndClear)
    .def_static("get_py_task", &PyWorker::GetPyTask, py::call_guard<py::gil_scoped_release>())
    .def_static("try_get_preprocess_py_task", &PyWorker::TryGetPreprocessPyTask)
    .def_static("try_get_postprocess_py_task", &PyWorker::TryGetPostprocessPyTask)
    .def_static("push_preprocess_result", &PyWorker::PushPreprocessPyResult)
    .def_static("push_preprocess_failed", &PyWorker::PushPreprocessPyFailed)
    .def_static("push_postprocess_result", &PyWorker::PushPostprocessPyResult)
    .def_static("push_postprocess_failed", &PyWorker::PushPostprocessPyFailed)
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
    .def_readwrite("model_file_name", &AgentStartUpConfig::model_file_name)
    .def_readwrite("group_file_name", &AgentStartUpConfig::group_file_name)
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
