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
#include "worker/context.h"
#include "python/master/master_py.h"

namespace mindspore::serving {

PYBIND11_MODULE(_mindspore_serving, m) {
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

  py::class_<ServableMeta>(m, "ServableMeta_")
    .def(py::init<>())
    .def_readwrite("servable_name", &ServableMeta::servable_name)
    .def_readwrite("inputs_count", &ServableMeta::inputs_count)
    .def_readwrite("outputs_count", &ServableMeta::outputs_count)
    .def_readwrite("servable_file", &ServableMeta::servable_file)
    .def_readwrite("with_batch_dim", &ServableMeta::with_batch_dim)
    .def_readwrite("options", &ServableMeta::load_options)
    .def_readwrite("without_batch_dim_inputs", &ServableMeta::without_batch_dim_inputs)
    .def("set_model_format", &ServableMeta::SetModelFormat);

  py::class_<ServableSignature>(m, "ServableSignature_")
    .def(py::init<>())
    .def_readwrite("servable_meta", &ServableSignature::servable_meta)
    .def_readwrite("methods", &ServableSignature::methods);

  py::class_<PyServableStorage>(m, "ServableStorage_")
    .def_static("register_servable_input_output_info", &PyServableStorage::RegisterInputOutputInfo)
    .def_static("register_method", &PyServableStorage::RegisterMethod)
    .def_static("declare_servable", &PyServableStorage::DeclareServable);

  py::class_<TaskContext>(m, "TaskContext_").def(py::init<>());

  py::class_<TaskItem>(m, "TaskItem_")
    .def(py::init<>())
    .def_readwrite("task_type", &TaskItem::task_type)
    .def_readwrite("name", &TaskItem::name)
    .def_property_readonly("instance_list",
                           [](const TaskItem &item) {
                             py::tuple instances(item.instance_list.size());
                             for (size_t i = 0; i < item.instance_list.size(); i++) {
                               instances[i] = PyTensor::AsNumpyTuple(item.instance_list[i].data);
                             }
                             return instances;
                           })
    .def_readwrite("context_list", &TaskItem::context_list);

  py::class_<PyWorker>(m, "Worker_")
    .def_static("start_servable", &PyWorker::StartServable)
    .def_static("start_servable_in_master", &PyWorker::StartServableInMaster)
    .def_static("get_batch_size", &PyWorker::GetBatchSize)
    .def_static("wait_and_clear", &PyWorker::WaitAndClear)
    .def_static("stop_and_clear", PyWorker::StopAndClear)
    .def_static("get_py_task", &PyWorker::GetPyTask, py::call_guard<py::gil_scoped_release>())
    .def_static("try_get_preprocess_py_task", &PyWorker::TryGetPreprocessPyTask)
    .def_static("try_get_postprocess_py_task", &PyWorker::TryGetPostprocessPyTask)
    .def_static("push_preprocess_result", &PyWorker::PushPreprocessPyResult)
    .def_static("push_preprocess_failed", &PyWorker::PushPreprocessPyFailed)
    .def_static("push_postprocess_result", &PyWorker::PushPostprocessPyResult)
    .def_static("push_postprocess_failed", &PyWorker::PushPostprocessPyFailed);

  py::class_<ServableContext, std::shared_ptr<ServableContext>>(m, "Context_")
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

  py::class_<PyMaster, std::shared_ptr<PyMaster>>(m, "Master_")
    .def_static("start_grpc_server", &PyMaster::StartGrpcServer)
    .def_static("start_grpc_master_server", &PyMaster::StartGrpcMasterServer)
    .def_static("start_restful_server", &PyMaster::StartRestfulServer)
    .def_static("wait_and_clear", &PyMaster::WaitAndClear)
    .def_static("stop_and_clear", &PyMaster::StopAndClear);

  (void)py::module::import("atexit").attr("register")(py::cpp_function{[&]() -> void {
    Server::Instance().Clear();
    Worker::GetInstance().Clear();
  }});
}

}  // namespace mindspore::serving
