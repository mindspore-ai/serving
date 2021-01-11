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

#ifndef MINDSPORE_SERVING_TEST_SERVABLE_COMMON_H
#define MINDSPORE_SERVING_TEST_SERVABLE_COMMON_H

#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "common/common_test.h"
#include "master/server.h"
#include "worker/worker.h"
#include "worker/notfiy_master/local_notify.h"
#include "worker/context.h"
#include "master/grpc/grpc_process.h"
#include "mindspore_serving/proto/ms_service.pb.h"

namespace mindspore {
namespace serving {

#define ExpectContainMsg(error_msg, expected_msg)                                                     \
  {                                                                                                   \
    auto error_msg_str = error_msg;                                                                   \
    EXPECT_TRUE(error_msg_str.find(expected_msg) != std::string::npos);                               \
    if (error_msg_str.find(expected_msg) == std::string::npos) {                                      \
      std::cout << "error_msg: " << error_msg_str << ", expected_msg: " << expected_msg << std::endl; \
    }                                                                                                 \
  }

class TestMasterWorker : public UT::Common {
 public:
  TestMasterWorker() = default;
  void Init(std::string servable_dir, std::string servable_name, int version_number, std::string model_file) {
    servable_dir_ = servable_dir;
    servable_name_ = servable_name;
    version_number_ = version_number;
    model_file_ = model_file;

    servable_name_path_ = servable_dir_ + "/" + servable_name_;
    version_number_path_ = servable_name_path_ + "/" + std::to_string(version_number_);
    model_name_path_ = version_number_path_ + "/" + model_file_;

    __mode_t access_mode = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH;
    mkdir(servable_dir_.c_str(), access_mode);
    mkdir(servable_name_path_.c_str(), access_mode);
    mkdir(version_number_path_.c_str(), access_mode);
    std::ofstream fp(model_name_path_);
    fp << "model content";
    fp.close();
    model_name_path_list_.emplace(model_name_path_);
    version_number_path_list_.emplace(version_number_path_);
    servable_name_path_list_.emplace(servable_name_path_);
    servable_dir_list_.emplace(servable_dir_);
  }

  virtual void SetUp() {}
  virtual void TearDown() {
    for (auto &item : model_name_path_list_) {
      remove(item.c_str());
    }
    for (auto &item : version_number_path_list_) {
      rmdir(item.c_str());
    }
    for (auto &item : servable_name_path_list_) {
      rmdir(item.c_str());
    }
    for (auto &item : servable_dir_list_) {
      rmdir(item.c_str());
    }
    Worker::GetInstance().Clear();
    Server::Instance().Clear();
  }

  void StartAddServable() {
    auto status = StartServable(servable_dir_, servable_name_, 0);
    ASSERT_TRUE(status.IsSuccess());
  }

  void RegisterAddServable(bool with_batch_dim = false) {
    DeclareServable(servable_name_, model_file_, "mindir", with_batch_dim);

    // register_method
    RegisterMethod(servable_name_, "add_common", {"x1", "x2"}, {"y"}, 2, 1);
  }

  static Status StartServable(const std::string &servable_dir, const std::string &servable_name, int version_number) {
    auto notify_master = std::make_shared<LocalNotifyMaster>();
    ServableContext::Instance()->SetDeviceId(0);
    ServableContext::Instance()->SetDeviceTypeStr("Ascend");
    Status status = Worker::GetInstance().StartServable(servable_dir, servable_name, version_number, notify_master);
    return status;
  }
  static void DeclareServable(const std::string &servable_name, const std::string &servable_file,
                              const std::string &model_type, bool with_batch_dim = false) {
    ServableMeta servable_meta;
    servable_meta.servable_name = servable_name;
    servable_meta.servable_file = servable_file;
    servable_meta.SetModelFormat(model_type);
    servable_meta.with_batch_dim = with_batch_dim;
    // declare_servable
    ServableStorage::Instance().DeclareServable(servable_meta);
  }
  static Status RegisterMethod(const std::string &servable_name, const std::string &method_name,
                               const std::vector<std::string> &input_names,
                               const std::vector<std::string> &output_names, size_t servable_input_count,
                               size_t servable_output_count) {
    auto status =
      ServableStorage::Instance().RegisterInputOutputInfo(servable_name, servable_input_count, servable_output_count);
    if (status != SUCCESS) {
      return status;
    }

    MethodSignature method_signature;
    method_signature.servable_name = servable_name;
    method_signature.method_name = method_name;
    method_signature.inputs = input_names;
    method_signature.outputs = output_names;
    // method input 0 and input 1 as servable input
    method_signature.servable_inputs = {{kPredictPhaseTag_Input, 0}, {kPredictPhaseTag_Input, 1}};
    // servable output as method output
    method_signature.returns = {{kPredictPhaseTag_Predict, 0}};
    ServableStorage::Instance().RegisterMethod(method_signature);
    return SUCCESS;
  }
  std::string servable_dir_;
  std::string servable_name_;
  int version_number_ = 0;
  std::string model_file_;
  std::string model_name_path_;
  std::string version_number_path_;
  std::string servable_name_path_;
  std::set<std::string> servable_dir_list_;
  std::set<std::string> model_name_path_list_;
  std::set<std::string> version_number_path_list_;
  std::set<std::string> servable_name_path_list_;
};

class TestMasterWorkerClient : public TestMasterWorker {
 public:
  TestMasterWorkerClient() = default;

  static void InitTensor(proto::Tensor *tensor, const std::vector<int64_t> &shape, proto::DataType data_type,
                         const void *data, size_t data_size) {
    MSI_EXCEPTION_IF_NULL(tensor);
    tensor->set_dtype(data_type);
    auto proto_shape = tensor->mutable_shape();
    for (auto item : shape) {
      proto_shape->add_dims(item);
    }
    tensor->set_data(data, data_size);
  }

  static std::vector<float> InitOneInstanceRequest(proto::PredictRequest *request, const std::string &servable_name,
                                                   const std::string &method_name, int version_number) {
    MSI_EXCEPTION_IF_NULL(request);
    auto request_servable_spec = request->mutable_servable_spec();
    request_servable_spec->set_name(servable_name);
    request_servable_spec->set_method_name(method_name);
    request_servable_spec->set_version_number(version_number);

    std::vector<float> x1_data = {1.1, 2.2, 3.3, 4.4};
    std::vector<float> x2_data = {1.2, 2.3, 3.4, 4.5};
    std::vector<float> y_data;
    for (size_t i = 0; i < x1_data.size(); i++) {
      y_data.push_back(x1_data[i] + x2_data[i]);
    }
    auto instance = request->add_instances();
    auto &input_map = (*instance->mutable_items());
    // input x1
    InitTensor(&input_map["x1"], {2, 2}, proto::MS_FLOAT32, x1_data.data(), x1_data.size() * sizeof(float));
    // input x2
    InitTensor(&input_map["x2"], {2, 2}, proto::MS_FLOAT32, x2_data.data(), x2_data.size() * sizeof(float));
    return y_data;
  }
  template <class IN_DT = float, class OUT_DT = float>
  static std::vector<std::vector<OUT_DT>> InitMultiInstancesRequest(proto::PredictRequest *request,
                                                                    const std::string &servable_name,
                                                                    const std::string &method_name, int version_number,
                                                                    size_t instances_count) {
    MSI_EXCEPTION_IF_NULL(request);
    auto request_servable_spec = request->mutable_servable_spec();
    request_servable_spec->set_name(servable_name);
    request_servable_spec->set_method_name(method_name);
    request_servable_spec->set_version_number(version_number);

    auto data_type = proto::MS_FLOAT32;
    if (std::string(typeid(IN_DT).name()) == std::string(typeid(int32_t).name())) {
      data_type = proto::MS_INT32;
    }

    std::vector<std::vector<OUT_DT>> y_data_list;
    for (size_t k = 0; k < instances_count; k++) {
      std::vector<float> x1_data_org = {1.1, 2.2, 3.3, 4.4};
      std::vector<float> x2_data_org = {6.6, 7.7, 8.8, 9.9};

      std::vector<IN_DT> x1_data;
      std::vector<IN_DT> x2_data;

      std::vector<OUT_DT> y_data;
      for (size_t i = 0; i < x1_data_org.size(); i++) {
        x1_data.push_back(static_cast<IN_DT>(x1_data_org[i] * (k + 1)));
        x2_data.push_back(static_cast<IN_DT>(x2_data_org[i] * (k + 1)));
        y_data.push_back(static_cast<OUT_DT>(x1_data[i] + x2_data[i]));
      }
      y_data_list.push_back(y_data);

      auto instance = request->add_instances();
      auto &input_map = (*instance->mutable_items());
      // input x1
      InitTensor(&input_map["x1"], {2, 2}, data_type, x1_data.data(), x1_data.size() * sizeof(IN_DT));
      // input x2
      InitTensor(&input_map["x2"], {2, 2}, data_type, x2_data.data(), x2_data.size() * sizeof(IN_DT));
    }
    return y_data_list;
  }

  template <class IN_DT = float, class OUT_DT = float>
  static std::vector<std::vector<OUT_DT>> InitMultiInstancesShape2Request(proto::PredictRequest *request,
                                                                          const std::string &servable_name,
                                                                          const std::string &method_name,
                                                                          int version_number, size_t instances_count) {
    MSI_EXCEPTION_IF_NULL(request);
    auto request_servable_spec = request->mutable_servable_spec();
    request_servable_spec->set_name(servable_name);
    request_servable_spec->set_method_name(method_name);
    request_servable_spec->set_version_number(version_number);

    auto data_type = proto::MS_FLOAT32;
    if (std::string(typeid(IN_DT).name()) == std::string(typeid(int32_t).name())) {
      data_type = proto::MS_INT32;
    }

    std::vector<std::vector<OUT_DT>> y_data_list;
    for (size_t k = 0; k < instances_count; k++) {
      std::vector<float> x1_data_org = {1.1, 2.2};
      std::vector<float> x2_data_org = {8.8, 9.9};

      std::vector<IN_DT> x1_data;
      std::vector<IN_DT> x2_data;

      std::vector<OUT_DT> y_data;
      for (size_t i = 0; i < x1_data_org.size(); i++) {
        x1_data.push_back(static_cast<IN_DT>(x1_data_org[i] * (k + 1)));
        x2_data.push_back(static_cast<IN_DT>(x2_data_org[i] * (k + 1)));
        y_data.push_back(x1_data[i] + x2_data[i]);
      }
      y_data_list.push_back(y_data);

      auto instance = request->add_instances();
      auto &input_map = (*instance->mutable_items());
      // input x1
      InitTensor(&input_map["x1"], {2}, data_type, x1_data.data(), x1_data.size() * sizeof(IN_DT));
      // input x2
      InitTensor(&input_map["x2"], {2}, data_type, x2_data.data(), x2_data.size() * sizeof(IN_DT));
    }
    return y_data_list;
  }

  template <class OUT_DT>
  static void CheckMultiInstanceResult(const proto::PredictReply &reply,
                                       const std::vector<std::vector<OUT_DT>> &y_data_list,
                                       size_t instances_count) {  // checkout output
    ASSERT_EQ(reply.instances_size(), instances_count);
    ASSERT_EQ(reply.error_msg_size(), 0);
    auto data_type = proto::MS_FLOAT32;
    if (std::string(typeid(OUT_DT).name()) == std::string(typeid(int32_t).name())) {
      data_type = proto::MS_INT32;
    }
    std::vector<int64_t> shape;
    if (y_data_list[0].size() == 4) {
      shape = {2, 2};
    } else {
      shape = {2};
    }
    for (size_t k = 0; k < instances_count; k++) {
      auto &output_instance = reply.instances(k);
      ASSERT_EQ(output_instance.items_size(), 1);
      auto &output_items = output_instance.items();
      ASSERT_EQ(output_items.begin()->first, "y");
      auto &output_tensor = output_items.begin()->second;

      CheckTensor(output_tensor, shape, data_type, y_data_list[k].data(), y_data_list[k].size() * sizeof(OUT_DT));
    }
  }

  template <class OUT_DT>
  static void CheckInstanceResult(const proto::PredictReply &reply, const std::vector<OUT_DT> &y_data) {
    // checkout output
    ASSERT_EQ(reply.instances_size(), 1);
    ASSERT_EQ(reply.error_msg_size(), 0);
    auto data_type = proto::MS_FLOAT32;
    if (std::string(typeid(OUT_DT).name()) == std::string(typeid(int32_t).name())) {
      data_type = proto::MS_INT32;
    }
    std::vector<int64_t> shape;
    if (y_data.size() == 4) {
      shape = {2, 2};
    } else {
      shape = {2};
    }
    auto &output_instance = reply.instances(0);
    ASSERT_EQ(output_instance.items_size(), 1);
    auto &output_items = output_instance.items();
    ASSERT_EQ(output_items.begin()->first, "y");
    auto &output_tensor = output_items.begin()->second;

    CheckTensor(output_tensor, shape, data_type, y_data.data(), y_data.size() * sizeof(OUT_DT));
  }

  static void CheckTensor(const proto::Tensor &output_tensor, const std::vector<int64_t> &shape,
                          proto::DataType data_type, const void *data, size_t data_size) {
    EXPECT_EQ(output_tensor.dtype(), data_type);
    // check shape [2,2]
    auto &output_tensor_shape = output_tensor.shape();
    ASSERT_EQ(output_tensor_shape.dims_size(), shape.size());
    std::vector<int64_t> proto_shape;
    for (size_t i = 0; i < output_tensor_shape.dims_size(); i++) {
      proto_shape.push_back(output_tensor_shape.dims(i));
    }
    EXPECT_EQ(proto_shape, shape);

    // check data
    ASSERT_EQ(output_tensor.data().size(), data_size);
    switch (data_type) {
      case proto::MS_FLOAT32: {
        auto data_len = data_size / sizeof(float);
        auto real_data = reinterpret_cast<const float *>(output_tensor.data().data());
        auto expect_data = reinterpret_cast<const float *>(data);
        for (size_t i = 0; i < data_len; i++) {
          EXPECT_EQ(real_data[i], expect_data[i]);
          if (real_data[i] != expect_data[i]) {
            break;
          }
        }
        break;
      }
      case proto::MS_INT32: {
        auto data_len = data_size / sizeof(int32_t);
        auto real_data = reinterpret_cast<const int32_t *>(output_tensor.data().data());
        auto expect_data = reinterpret_cast<const int32_t *>(data);
        for (size_t i = 0; i < data_len; i++) {
          EXPECT_EQ(real_data[i], expect_data[i]);
          if (real_data[i] != expect_data[i]) {
            break;
          }
        }
        break;
      }
      default:
        FAIL();
    }
  }
  static grpc::Status Dispatch(const proto::PredictRequest &request, proto::PredictReply *reply) {
    MSServiceImpl impl(Server::Instance().GetDispatcher());
    grpc::ServerContext context;
    return impl.Predict(&context, &request, reply);
  }
};

}  // namespace serving
}  // namespace mindspore
#endif  // MINDSPORE_SERVING_TEST_SERVABLE_COMMON_H
