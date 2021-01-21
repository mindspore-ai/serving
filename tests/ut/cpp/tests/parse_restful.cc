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
#include "common/common_test.h"
#include "master/server.h"
#include "common/tensor_base.h"
#define private public
#include "master/restful/http_process.h"
#undef private

using std::string;
using std::vector;
namespace mindspore {
namespace serving {
class TestParseInput : public UT::Common {
 public:
  TestParseInput() = default;
};

class TestParseReply : public UT::Common {
 public:
  TestParseReply() = default;
};

TEST_F(TestParseInput, test_parse_SUCCESS) {
  nlohmann::json js = R"(
    {"instances":[
        {
          "key_tag":"scalar",
          "key_int": 1,
          "key_bool": false,
          "key_float": 2.3,
          "key_str": "ut_test",
          "key_bytes": {"b64": "dXRfdGVzdA==", "type": "bytes"}
        },
        {
          "key_tag":"tensor",
          "key_int": [1,2,3],
          "key_bool":[[true, false], [false, true]],
          "key_float":[[1.1, 2.2]],
          "key_str":["ut_test"],
          "key_bytes":{"b64":"dXRfdGVzdA=="}
        },
        {
          "key_tag":"b64",
          "key_str_format1":"ut_test",
          "key_str_foramt2":{"b64":"dXRfdGVzdA==", "type":"str"},
          "key_bytes_int16":{"b64":"AQACAAIAAwADAAQA", "type":"int16", "shape":[3,2]},
          "key_bytes_fp16":{"b64":"ZjxmQJpCZkQ=", "type":"fp16", "shape":[2,2]},
          "key_bytes_bool":{"b64":"AQA=", "type":"bool", "shape":[1,2]}
        }
      ]
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_EQ(status.StatusCode(), SUCCESS);
  ASSERT_EQ(predict_request.instances().size(), 3);
  for (int32_t i = 0; i < predict_request.instances().size(); i++) {
    auto &cur_instance = predict_request.instances(i);
    auto &items = cur_instance.items();
    if (i == 0) {
      ASSERT_EQ(items.size(), 6);
      for (const auto &item : items) {
        ProtoTensor pb_tensor(const_cast<proto::Tensor *>(&item.second));
        if (item.first == "key_int") {
          ASSERT_EQ(pb_tensor.data_type(), DataType::kMSI_Int32);
          const int32_t *data = reinterpret_cast<const int32_t *>(pb_tensor.data());
          ASSERT_EQ(*data, 1);
        } else if (item.first == "key_bool") {
          ASSERT_EQ(pb_tensor.data_type(), DataType::kMSI_Bool);
          const bool *data = reinterpret_cast<const bool *>(pb_tensor.data());
          ASSERT_EQ(*data, false);
        } else if (item.first == "key_float") {
          ASSERT_EQ(pb_tensor.data_type(), DataType::kMSI_Float32);
          const float *data = reinterpret_cast<const float *>(pb_tensor.data());
          ASSERT_FLOAT_EQ(*data, 2.3);
        } else if (item.first == "key_str") {
          ASSERT_EQ(pb_tensor.data_type(), DataType::kMSI_String);
          auto str_nums = pb_tensor.bytes_data_size();
          ASSERT_EQ(str_nums, 1);
          std::string value;
          size_t length;
          const uint8_t *ptr = nullptr;
          pb_tensor.get_bytes_data(0, &ptr, &length);
          value.resize(length);
          memcpy_s(value.data(), length, reinterpret_cast<const char *>(ptr), length);
          ASSERT_EQ(value, "ut_test");
        } else if (item.first == "key_bytes") {
          ASSERT_EQ(pb_tensor.data_type(), DataType::kMSI_Bytes);
          auto str_nums = pb_tensor.bytes_data_size();
          ASSERT_EQ(str_nums, 1);
          std::string value;
          size_t length;
          const uint8_t *ptr = nullptr;
          pb_tensor.get_bytes_data(0, &ptr, &length);
          value.resize(length);
          memcpy_s(value.data(), length, reinterpret_cast<const char *>(ptr), length);
          ASSERT_EQ(value, "ut_test");
        }
      }
    } else if (i == 1) {
      ASSERT_EQ(items.size(), 6);
      for (const auto &item : items) {
        ProtoTensor pb_tensor(const_cast<proto::Tensor *>(&item.second));
        auto shape = pb_tensor.shape();
        if (item.first == "key_int") {
          ASSERT_EQ(pb_tensor.data_type(), DataType::kMSI_Int32);
          ASSERT_EQ(shape.size(), 1);
          ASSERT_EQ(shape[0], 3);
          vector<int32_t> expected_value = {1, 2, 3};
          for (int i = 0; i < 3; i++) {
            const int32_t *data = reinterpret_cast<const int32_t *>(pb_tensor.data()) + i;
            ASSERT_EQ(*data, expected_value[i]);
          }
        } else if (item.first == "key_bool") {
          ASSERT_EQ(pb_tensor.data_type(), DataType::kMSI_Bool);
          ASSERT_EQ(shape.size(), 2);
          ASSERT_EQ(shape[0], 2);
          ASSERT_EQ(shape[1], 2);
          vector<vector<bool>> expected_value = {{true, false}, {false, true}};
          for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
              const bool *data = reinterpret_cast<const bool *>(pb_tensor.data()) + i * 2 + j;
              ASSERT_EQ(*data, expected_value[i][j]);
            }
          }
        } else if (item.first == "key_float") {
          ASSERT_EQ(pb_tensor.data_type(), DataType::kMSI_Float32);
          ASSERT_EQ(shape.size(), 2);
          ASSERT_EQ(shape[0], 1);
          ASSERT_EQ(shape[1], 2);
          vector<vector<float>> expected_value = {{1.1, 2.2}};
          for (int i = 0; i < 1; i++) {
            for (int j = 0; j < 2; j++) {
              const float *data = reinterpret_cast<const float *>(pb_tensor.data()) + i * 1 + j;
              ASSERT_FLOAT_EQ(*data, expected_value[i][j]);
            }
          }
        } else if (item.first == "key_str") {
          ASSERT_EQ(pb_tensor.data_type(), DataType::kMSI_String);
          ASSERT_EQ(shape.size(), 1);
          ASSERT_EQ(shape[0], 1);
          auto str_nums = pb_tensor.bytes_data_size();
          ASSERT_EQ(str_nums, 1);
          std::string value;
          size_t length;
          const uint8_t *ptr = nullptr;
          pb_tensor.get_bytes_data(0, &ptr, &length);
          value.resize(length);
          memcpy_s(value.data(), length, reinterpret_cast<const char *>(ptr), length);
          ASSERT_EQ(value, "ut_test");
        } else if (item.first == "key_bytes") {
          ASSERT_EQ(pb_tensor.data_type(), DataType::kMSI_Bytes);
          auto str_nums = pb_tensor.bytes_data_size();
          ASSERT_EQ(str_nums, 1);
          std::string value;
          size_t length;
          const uint8_t *ptr = nullptr;
          pb_tensor.get_bytes_data(0, &ptr, &length);
          value.resize(length);
          memcpy_s(value.data(), length, reinterpret_cast<const char *>(ptr), length);
          ASSERT_EQ(value, "ut_test");
        }
      }
    } else if (i == 2) {
      ASSERT_EQ(items.size(), 6);
      for (const auto &item : items) {
        ProtoTensor pb_tensor(const_cast<proto::Tensor *>(&item.second));
        auto shape = pb_tensor.shape();
        if (item.first == "key_str_format1") {
          ASSERT_EQ(pb_tensor.data_type(), DataType::kMSI_String);
          auto str_nums = pb_tensor.bytes_data_size();
          ASSERT_EQ(str_nums, 1);
          std::string value;
          size_t length;
          const uint8_t *ptr = nullptr;
          pb_tensor.get_bytes_data(0, &ptr, &length);
          value.resize(length);
          memcpy_s(value.data(), length, reinterpret_cast<const char *>(ptr), length);
          ASSERT_EQ(value, "ut_test");
        } else if (item.first == "key_str_format2") {
          ASSERT_EQ(pb_tensor.data_type(), DataType::kMSI_String);
          auto str_nums = pb_tensor.bytes_data_size();
          ASSERT_EQ(str_nums, 1);
          std::string value;
          size_t length;
          const uint8_t *ptr = nullptr;
          pb_tensor.get_bytes_data(0, &ptr, &length);
          value.resize(length);
          memcpy_s(value.data(), length, reinterpret_cast<const char *>(ptr), length);
          ASSERT_EQ(value, "ut_test");
        } else if (item.first == "key_bytes_int16") {
          ASSERT_EQ(pb_tensor.data_type(), DataType::kMSI_Int16);
          ASSERT_EQ(shape.size(), 2);
          ASSERT_EQ(shape[0], 3);
          ASSERT_EQ(shape[1], 2);
          vector<vector<int16_t>> expected_value = {{1, 2}, {2, 3}, {3, 4}};
          for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
              const int16_t *data = reinterpret_cast<const int16_t *>(pb_tensor.data()) + i * 2 + j;
              ASSERT_FLOAT_EQ(*data, expected_value[i][j]);
            }
          }
        } else if (item.first == "key_bytes_fp16") {
          ASSERT_EQ(pb_tensor.data_type(), DataType::kMSI_Float16);
          ASSERT_EQ(shape.size(), 2);
          ASSERT_EQ(shape[0], 2);
          ASSERT_EQ(shape[1], 2);
        } else if (item.first == "key_bytes_bool") {
          ASSERT_EQ(pb_tensor.data_type(), DataType::kMSI_Bool);
          ASSERT_EQ(shape.size(), 2);
          ASSERT_EQ(shape[0], 1);
          ASSERT_EQ(shape[1], 2);
          vector<vector<bool>> expected_value = {{true, false}};
          for (int i = 0; i < 1; i++) {
            for (int j = 0; j < 2; j++) {
              const bool *data = reinterpret_cast<const bool *>(pb_tensor.data()) + i * 2 + j;
              ASSERT_FLOAT_EQ(*data, expected_value[i][j]);
            }
          }
        }
      }
    }
  }
}

TEST_F(TestParseInput, test_instances_empty_FAIL) {
  nlohmann::json js = R"(
    {"":
        {
          "key_tag":"scalar",
          "key_int": 1,
          "key_bool": false,
          "key_float": 2.3,
          "key_str": "ut_test",
          "key_bytes": {"b64": "dXRfdGVzdA==", "type": "bytes"}
        }
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_NE(status.StatusCode(), SUCCESS);
}

TEST_F(TestParseInput, test_instances_incorrect_FAIL) {
  nlohmann::json js = R"(
    {"instance":
        {
          "key_tag":"scalar",
          "key_int": 1,
          "key_bool": false,
          "key_float": 2.3,
          "key_str": "ut_test",
          "key_bytes": {"b64": "dXRfdGVzdA==", "type": "bytes"}
        }
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_NE(status.StatusCode(), SUCCESS);
}

TEST_F(TestParseInput, test_key_empty_FAIL) {
  nlohmann::json js = R"(
    {"instances":
        {
          "":"scalar",
          "key_int": 1,
          "key_bool": false,
          "key_float": 2.3,
          "key_str": "ut_test",
          "key_bytes": {"b64": "dXRfdGVzdA==", "type": "bytes"}
        }
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_NE(status.StatusCode(), SUCCESS);
}

TEST_F(TestParseInput, test_value_empty_SUCCESS) {
  nlohmann::json js = R"(
    {"instances":
        {
          "key_tag":"",
          "key_int": 1,
          "key_bool": false,
          "key_float": 2.3,
          "key_str": "ut_test",
          "key_bytes": {"b64": "dXRfdGVzdA==", "type": "bytes"}
        }
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_EQ(status.StatusCode(), SUCCESS);
}

TEST_F(TestParseInput, test_obj_unknown_key_FAIL) {
  nlohmann::json js = R"(
    {"instances":
        {
          "key_tag":"",
          "key_int": 1,
          "key_bool": false,
          "key_float": 2.3,
          "key_str": "ut_test",
          "key_bytes": {"b64": "dXRfdGVzdA==", "type": "bytes", "type1":"bytes"}
        }
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_NE(status.StatusCode(), SUCCESS);
}

TEST_F(TestParseInput, test_obj_nob64_key_FAIL) {
  nlohmann::json js = R"(
    {"instances":
        {
          "key_tag":"",
          "key_int": 1,
          "key_bool": false,
          "key_float": 2.3,
          "key_str": "ut_test",
          "key_bytes": {"base64": "dXRfdGVzdA==", "type": "bytes"}
        }
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_NE(status.StatusCode(), SUCCESS);
}

TEST_F(TestParseInput, test_obj_illegal_b64value_FAIL) {
  nlohmann::json js = R"(
    {"instances":
        {
          "key_tag":"",
          "key_int": 1,
          "key_bool": false,
          "key_float": 2.3,
          "key_str": "ut_test",
          "key_bytes": {"base64": "dXRfdGVzdA", "type": "bytes"}
        }
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_NE(status.StatusCode(), SUCCESS);
}

TEST_F(TestParseInput, test_obj_unknown_type_FAIL) {
  nlohmann::json js = R"(
    {"instances":
        {
          "key_tag":"",
          "key_int": 1,
          "key_bool": false,
          "key_float": 2.3,
          "key_str": "ut_test",
          "key_bytes": {"base64": "dXRfdGVzdA==", "type": "INt"}
        }
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_NE(status.StatusCode(), SUCCESS);
}

TEST_F(TestParseInput, test_obj_error_shape_format_FAIL) {
  nlohmann::json js = R"(
    {"instances":
        {
          "key_tag":"",
          "key_int": 1,
          "key_bool": false,
          "key_float": 2.3,
          "key_str": "ut_test",
          "key_bytes_int16":{"b64":"AQACAAIAAwADAAQA", "type":"int16", "shape":3}
        }
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_NE(status.StatusCode(), SUCCESS);
}

TEST_F(TestParseInput, test_obj_error_shape_format2_FAIL) {
  nlohmann::json js = R"(
    {"instances":
        {
          "key_tag":"",
          "key_int": 1,
          "key_bool": false,
          "key_float": 2.3,
          "key_str": "ut_test",
          "key_bytes_int16":{"b64":"AQACAAIAAwADAAQA", "type":"int16", "shape":[[3],[2]]}
        }
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_NE(status.StatusCode(), SUCCESS);
}

TEST_F(TestParseInput, test_obj_error_shape_value_FAIL) {
  nlohmann::json js = R"(
    {"instances":
        {
          "key_tag":"",
          "key_int": 1,
          "key_bool": false,
          "key_float": 2.3,
          "key_str": "ut_test",
          "key_bytes_int16":{"b64":"AQACAAIAAwADAAQA", "type":"int16", "shape":[3.0,2.0]}
        }
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_NE(status.StatusCode(), SUCCESS);
}

TEST_F(TestParseInput, test_obj_error_shape_value2_FAIL) {
  nlohmann::json js = R"(
    {"instances":
        {
          "key_tag":"",
          "key_int": 1,
          "key_bool": false,
          "key_float": 2.3,
          "key_str": "ut_test",
          "key_bytes_int16":{"b64":"AQACAAIAAwADAAQA", "type":"int16", "shape":[3,3]}
        }
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_NE(status.StatusCode(), SUCCESS);
}

TEST_F(TestParseInput, test_obj_error_shape_value3_FAIL) {
  nlohmann::json js = R"(
    {"instances":
        {
          "key_tag":"",
          "key_int": 1,
          "key_bool": false,
          "key_float": 2.3,
          "key_str": "ut_test",
          "key_bytes_int16":{"b64":"AQACAAIAAwADAAQA", "type":"int16", "shape":[3,-2]}
        }
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_NE(status.StatusCode(), SUCCESS);
}

TEST_F(TestParseInput, test_tensor_value_empty_FAIL) {
  nlohmann::json js = R"(
    {"instances":
       {
          "key_tag":"tensor",
          "key_int": [],
          "key_bool":[[true, false], [false, true]],
          "key_float":[[1.1, 2.2]],
          "key_str":["ut_test"],
          "key_bytes":{"b64":"dXRfdGVzdA=="}
        }
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_NE(status.StatusCode(), SUCCESS);
}

TEST_F(TestParseInput, test_tensor_value_diff_type_FAIL) {
  nlohmann::json js = R"(
    {"instances":
       {
          "key_tag":"tensor",
          "key_int": [1, 2.0],
          "key_bool":[[true, false], [false, true]],
          "key_float":[[1.1, 2.2]],
          "key_str":["ut_test"],
          "key_bytes":{"b64":"dXRfdGVzdA=="}
        }
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_NE(status.StatusCode(), SUCCESS);
}

TEST_F(TestParseInput, test_tensor_value_diff_dimention_FAIL) {
  nlohmann::json js = R"(
    {"instances":
       {
          "key_tag":"tensor",
          "key_int": [1, 2],
          "key_bool":[[true, false], [false]],
          "key_float":[[1.1, 2.2]],
          "key_str":["ut_test"],
          "key_bytes":{"b64":"dXRfdGVzdA=="}
        }
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_NE(status.StatusCode(), SUCCESS);
}

TEST_F(TestParseInput, test_tensor_multi_object_FAIL) {
  nlohmann::json js = R"(
    {"instances":
       {
          "key_tag":"tensor",
          "key_int": [1, 2],
          "key_bool":[[true, false], [false, true]],
          "key_float":[[1.1, 2.2]],
          "key_str":["ut_test"],
          "key_bytes":[{"b64":"dXRfdGVzdA=="}, {"b64":"dXRfdGVzdA=="}]
        }
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_NE(status.StatusCode(), SUCCESS);
}

TEST_F(TestParseReply, test_reply_SUCCESS) {
  nlohmann::json js = R"(
    {"instances":[
        {
          "key_tag":"scalar",
          "key_int": 1,
          "key_bool": false,
          "key_float": 2.3,
          "key_str": "ut_test",
          "key_bytes": {"b64": "dXRfdGVzdA==", "type": "bytes"}
        },
        {
          "key_tag":"tensor",
          "key_int": [1,2,3],
          "key_bool":[[true, false], [false, true]],
          "key_float":[[1.1, 2.2]],
          "key_str":["ut_test"]
        }
      ]
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status(INVALID_INPUTS);
  status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_EQ(status.StatusCode(), SUCCESS);

  nlohmann::json out_js;
  proto::PredictReply reply;
  auto instance_ptr = reply.add_instances();
  auto &map_item = *(instance_ptr->mutable_items());
  // test scalar:
  // scalar:key_int
  proto::Tensor tensor_int;
  ProtoTensor pb_tensor_int(&tensor_int);
  DataType type_int = kMSI_Int32;
  pb_tensor_int.set_data_type(type_int);
  pb_tensor_int.set_shape({1});
  pb_tensor_int.resize_data(pb_tensor_int.GetTypeSize(type_int));
  auto data_int = reinterpret_cast<int32_t *>(pb_tensor_int.mutable_data());
  *data_int = 1;
  map_item["key_int"] = tensor_int;

  // scalar: key_bool
  proto::Tensor tensor_bool;
  ProtoTensor pb_tensor_bool(&tensor_bool);
  DataType type_bool = kMSI_Bool;
  pb_tensor_bool.set_data_type(type_bool);
  pb_tensor_bool.resize_data(pb_tensor_bool.GetTypeSize(type_bool));
  auto data_bool = reinterpret_cast<bool *>(pb_tensor_bool.mutable_data());
  *data_bool = false;
  map_item["key_bool"] = tensor_bool;

  // scalar: key_float
  proto::Tensor tensor_float;
  ProtoTensor pb_tensor_float(&tensor_float);
  DataType type_float = kMSI_Float32;
  pb_tensor_float.set_data_type(type_float);
  pb_tensor_float.set_shape({1});
  pb_tensor_float.resize_data(pb_tensor_float.GetTypeSize(type_float));
  auto data_float = reinterpret_cast<float *>(pb_tensor_float.mutable_data());
  *data_float = 2.3;
  map_item["key_float"] = tensor_float;

  // scalar: key_str
  string value = "ut_test";
  proto::Tensor tensor_str;
  ProtoTensor pb_tensor_str(&tensor_str);
  DataType type_str = kMSI_String;
  pb_tensor_str.set_data_type(type_str);
  pb_tensor_str.add_bytes_data(reinterpret_cast<uint8_t *>(value.data()), value.length());
  map_item["key_str"] = tensor_str;

  // scalar: key_bytes
  string value_bytes = "ut_test";
  proto::Tensor tensor_bytes;
  ProtoTensor pb_tensor_bytes(&tensor_bytes);
  DataType type_bytes = kMSI_Bytes;
  pb_tensor_bytes.set_data_type(type_bytes);
  pb_tensor_bytes.add_bytes_data(reinterpret_cast<uint8_t *>(value_bytes.data()), value_bytes.length());
  map_item["key_bytes"] = tensor_bytes;

  // test tensor:
  auto instance_ptr2 = reply.add_instances();
  auto &map_item2 = *(instance_ptr2->mutable_items());

  // tensor int:
  vector<int32_t> tensor_value_int = {1, 2, 3};
  proto::Tensor tensor_int2;
  ProtoTensor pb_tensor_int2(&tensor_int2);
  DataType type_int2 = kMSI_Int32;
  pb_tensor_int2.set_data_type(type_int2);
  pb_tensor_int2.set_shape({3});
  pb_tensor_int2.resize_data(pb_tensor_int2.GetTypeSize(type_int2) * 3);
  for (int i = 0; i < 3; i++) {
    auto data_int2 = reinterpret_cast<int32_t *>(pb_tensor_int2.mutable_data()) + i;
    *data_int2 = tensor_value_int[i];
  }
  map_item2["key_int"] = tensor_int2;

  // tensor: key_bool
  vector<vector<bool>> tensor_value_bool = {{true, false}, {false, true}};
  proto::Tensor tensor_bool2;
  ProtoTensor pb_tensor_bool2(&tensor_bool2);
  DataType type_bool2 = kMSI_Bool;
  pb_tensor_bool2.set_data_type(type_bool2);
  pb_tensor_bool2.set_shape({2, 2});
  pb_tensor_bool2.resize_data(pb_tensor_bool2.GetTypeSize(type_bool2) * 4);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      auto data_bool2 = reinterpret_cast<bool *>(pb_tensor_bool2.mutable_data()) + i * 2 + j;
      *data_bool2 = tensor_value_bool[i][j];
    }
  }
  map_item2["key_bool"] = tensor_bool2;

  // tensor: key_float
  vector<vector<float>> tensor_value_float = {{1.1, 2.2}};
  proto::Tensor tensor_float2;
  ProtoTensor pb_tensor_float2(&tensor_float2);
  DataType type_float2 = kMSI_Float32;
  pb_tensor_float2.set_data_type(type_float2);
  pb_tensor_float2.set_shape({1, 2});
  pb_tensor_float2.resize_data(pb_tensor_float2.GetTypeSize(type_float2) * 2);
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      auto data_float2 = reinterpret_cast<float *>(pb_tensor_float2.mutable_data()) + i * 1 + j;
      *data_float2 = tensor_value_float[i][j];
    }
  }
  map_item2["key_float"] = tensor_float2;

  // tensor: key_str
  vector<string> tensor_value_str = {"ut_test", "ut_test2"};
  proto::Tensor tensor_str2;
  ProtoTensor pb_tensor_str2(&tensor_str2);
  DataType type_str2 = kMSI_String;
  pb_tensor_str2.set_data_type(type_str2);
  pb_tensor_str2.set_shape({2});
  for (int i = 0; i < 2; i++) {
    pb_tensor_str2.add_bytes_data(reinterpret_cast<uint8_t *>(tensor_value_str[i].data()),
                                  tensor_value_str[i].length());
  }
  map_item2["key_str"] = tensor_str2;

  Status status2 = restful_service.ParseReply(reply, &out_js);
  ASSERT_EQ(status2.StatusCode(), SUCCESS);
  string out_str = out_js.dump();
  std::cout << "Parse reply out:" << out_str << std::endl;

  ASSERT_TRUE(out_js.is_object());
  for (auto &item : out_js.items()) {
    ASSERT_EQ(item.key(), "instances");
    ASSERT_TRUE(item.value().is_array());
    ASSERT_EQ(item.value().size(), 2);
    int sum = 0;
    // array
    for (auto &element : item.value()) {
      ASSERT_TRUE(element.is_object());
      if (element.size() == 5) {
        int count = 0;
        // object
        std::cout << "===start====" << std::endl;
        for (auto &it : element.items()) {
          if (it.key() == "key_int") {
            ASSERT_TRUE(it.value().is_array());
            ASSERT_EQ(it.value().size(), 1);
            auto array_items = it.value().items();
            auto int_val = *(array_items.begin());
            ASSERT_TRUE(int_val.value().is_number_integer());
            ASSERT_EQ(int_val.value(), 1);
            count++;
          } else if (it.key() == "key_bool") {
            ASSERT_TRUE(it.value().is_boolean());
            ASSERT_EQ(it.value(), false);
            count++;
          } else if (it.key() == "key_float") {
            ASSERT_TRUE(it.value().is_array());
            ASSERT_EQ(it.value().size(), 1);
            auto array_items = it.value().items();
            auto float_val = *(array_items.begin());
            ASSERT_FLOAT_EQ(float_val.value(), 2.3);
            count++;
          } else if (it.key() == "key_str") {
            ASSERT_TRUE(it.value().is_string());
            ASSERT_EQ(it.value(), "ut_test");
            count++;
          } else if (it.key() == "key_bytes") {
            ASSERT_TRUE(it.value().is_object());
            ASSERT_EQ(it.value()["b64"], "dXRfdGVzdA==");
            count++;
          }
        }
        ASSERT_EQ(count, 5);
        sum++;
      } else if (element.size() == 4) {
        int count = 0;
        // object
        for (auto &it : element.items()) {
          if (it.key() == "key_int") {
            ASSERT_TRUE(it.value().is_array());
            ASSERT_EQ(it.value().size(), 3);
            ASSERT_EQ(it.value()[0], 1);
            ASSERT_EQ(it.value()[1], 2);
            ASSERT_EQ(it.value()[2], 3);
            count++;
          } else if (it.key() == "key_bool") {
            ASSERT_TRUE(it.value().is_array());
            ASSERT_EQ(it.value().size(), 2);
            ASSERT_TRUE(it.value()[0].is_array());
            ASSERT_EQ(it.value()[0].size(), 2);
            ASSERT_EQ(it.value()[0][0], true);
            ASSERT_EQ(it.value()[0][1], false);
            ASSERT_EQ(it.value()[1].size(), 2);
            ASSERT_EQ(it.value()[1][0], false);
            ASSERT_EQ(it.value()[1][1], true);
            count++;
          } else if (it.key() == "key_float") {
            ASSERT_TRUE(it.value().is_array());
            ASSERT_EQ(it.value().size(), 1);
            ASSERT_TRUE(it.value()[0].is_array());
            ASSERT_EQ(it.value()[0].size(), 2);
            ASSERT_FLOAT_EQ(it.value()[0][0], 1.1);
            ASSERT_FLOAT_EQ(it.value()[0][1], 2.2);
            count++;
          } else if (it.key() == "key_str") {
            ASSERT_TRUE(it.value().is_array());
            ASSERT_EQ(it.value().size(), 2);
            ASSERT_EQ(it.value()[0], "ut_test");
            ASSERT_EQ(it.value()[1], "ut_test2");
            count++;
          }
        }
        ASSERT_EQ(count, 4);
        sum++;
      }
    }
    ASSERT_EQ(sum, 2);
  }
}

TEST_F(TestParseReply, test_reply_instances_num_not_match_FAIL) {
  nlohmann::json js = R"(
    {"instances":[
        {
          "key_tag":"scalar",
          "key_int": 1,
          "key_bool": false,
          "key_float": 2.3,
          "key_str": "ut_test",
          "key_bytes": {"b64": "dXRfdGVzdA==", "type": "bytes"}
        }
      ]
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status(INVALID_INPUTS);
  status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_EQ(status.StatusCode(), SUCCESS);

  nlohmann::json out_js;
  proto::PredictReply reply;
  auto instance_ptr = reply.add_instances();
  auto &map_item = *(instance_ptr->mutable_items());
  // test scalar:
  // scalar:key_int
  proto::Tensor tensor_int;
  ProtoTensor pb_tensor_int(&tensor_int);
  DataType type_int = kMSI_Int32;
  pb_tensor_int.set_data_type(type_int);
  pb_tensor_int.set_shape({1});
  pb_tensor_int.resize_data(pb_tensor_int.GetTypeSize(type_int));
  auto data_int = reinterpret_cast<int32_t *>(pb_tensor_int.mutable_data());
  *data_int = 1;
  map_item["key_int"] = tensor_int;

  // scalar: key_bool
  proto::Tensor tensor_bool;
  ProtoTensor pb_tensor_bool(&tensor_bool);
  DataType type_bool = kMSI_Bool;
  pb_tensor_bool.set_data_type(type_bool);
  pb_tensor_bool.resize_data(pb_tensor_bool.GetTypeSize(type_bool));
  auto data_bool = reinterpret_cast<bool *>(pb_tensor_bool.mutable_data());
  *data_bool = false;
  map_item["key_bool"] = tensor_bool;

  // scalar: key_float
  proto::Tensor tensor_float;
  ProtoTensor pb_tensor_float(&tensor_float);
  DataType type_float = kMSI_Float32;
  pb_tensor_float.set_data_type(type_float);
  pb_tensor_float.set_shape({1});
  pb_tensor_float.resize_data(pb_tensor_float.GetTypeSize(type_float));
  auto data_float = reinterpret_cast<float *>(pb_tensor_float.mutable_data());
  *data_float = 2.3;
  map_item["key_float"] = tensor_float;

  // scalar: key_str
  string value = "ut_test";
  proto::Tensor tensor_str;
  ProtoTensor pb_tensor_str(&tensor_str);
  DataType type_str = kMSI_String;
  pb_tensor_str.set_data_type(type_str);
  pb_tensor_str.add_bytes_data(reinterpret_cast<uint8_t *>(value.data()), value.length());
  map_item["key_str"] = tensor_str;

  // scalar: key_bytes
  string value_bytes = "ut_test";
  proto::Tensor tensor_bytes;
  ProtoTensor pb_tensor_bytes(&tensor_bytes);
  DataType type_bytes = kMSI_Bytes;
  pb_tensor_bytes.set_data_type(type_bytes);
  pb_tensor_bytes.add_bytes_data(reinterpret_cast<uint8_t *>(value_bytes.data()), value_bytes.length());
  map_item["key_bytes"] = tensor_bytes;

  // test tensor:
  auto instance_ptr2 = reply.add_instances();
  auto &map_item2 = *(instance_ptr2->mutable_items());

  // tensor int:
  vector<int32_t> tensor_value_int = {1, 2, 3};
  proto::Tensor tensor_int2;
  ProtoTensor pb_tensor_int2(&tensor_int2);
  DataType type_int2 = kMSI_Int32;
  pb_tensor_int2.set_data_type(type_int2);
  pb_tensor_int2.set_shape({3});
  pb_tensor_int2.resize_data(pb_tensor_int2.GetTypeSize(type_int2) * 3);
  for (int i = 0; i < 3; i++) {
    auto data_int2 = reinterpret_cast<int32_t *>(pb_tensor_int2.mutable_data()) + i;
    *data_int2 = tensor_value_int[i];
  }
  map_item2["key_int"] = tensor_int2;

  // tensor: key_bool
  vector<vector<bool>> tensor_value_bool = {{true, false}, {false, true}};
  proto::Tensor tensor_bool2;
  ProtoTensor pb_tensor_bool2(&tensor_bool2);
  DataType type_bool2 = kMSI_Bool;
  pb_tensor_bool2.set_data_type(type_bool2);
  pb_tensor_bool2.set_shape({2, 2});
  pb_tensor_bool2.resize_data(pb_tensor_bool2.GetTypeSize(type_bool2) * 4);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      auto data_bool2 = reinterpret_cast<bool *>(pb_tensor_bool2.mutable_data()) + i * 2 + j;
      *data_bool2 = tensor_value_bool[i][j];
    }
  }
  map_item2["key_bool"] = tensor_bool2;

  // tensor: key_float
  vector<vector<float>> tensor_value_float = {{1.1, 2.2}};
  proto::Tensor tensor_float2;
  ProtoTensor pb_tensor_float2(&tensor_float2);
  DataType type_float2 = kMSI_Float32;
  pb_tensor_float2.set_data_type(type_float2);
  pb_tensor_float2.set_shape({1, 2});
  pb_tensor_float2.resize_data(pb_tensor_float2.GetTypeSize(type_float2) * 2);
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      auto data_float2 = reinterpret_cast<float *>(pb_tensor_float2.mutable_data()) + i * 1 + j;
      *data_float2 = tensor_value_float[i][j];
    }
  }
  map_item2["key_float"] = tensor_float2;

  // tensor: key_str
  vector<string> tensor_value_str = {"ut_test", "ut_test2"};
  proto::Tensor tensor_str2;
  ProtoTensor pb_tensor_str2(&tensor_str2);
  DataType type_str2 = kMSI_String;
  pb_tensor_str2.set_data_type(type_str2);
  pb_tensor_str2.set_shape({2});
  for (int i = 0; i < 2; i++) {
    pb_tensor_str2.add_bytes_data(reinterpret_cast<uint8_t *>(tensor_value_str[i].data()),
                                  tensor_value_str[i].length());
  }
  map_item2["key_str"] = tensor_str2;

  Status status2 = restful_service.ParseReply(reply, &out_js);
  ASSERT_NE(status2.StatusCode(), SUCCESS);
}

TEST_F(TestParseReply, test_reply_error_num_not_match_FAIL) {
  nlohmann::json js = R"(
    {"instances":[
        {
          "key_tag":"scalar",
          "key_int": 1,
          "key_bool": false,
          "key_float": 2.3,
          "key_str": "ut_test",
          "key_bytes": {"b64": "dXRfdGVzdA==", "type": "bytes"}
        }
      ]
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status(INVALID_INPUTS);
  status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_EQ(status.StatusCode(), SUCCESS);

  nlohmann::json out_js;
  proto::PredictReply reply;
  auto error_msg = reply.add_error_msg();
  error_msg->set_error_msg("error1");

  auto error_msg2 = reply.add_error_msg();
  error_msg2->set_error_msg("error2");

  Status status2 = restful_service.ParseReply(reply, &out_js);
  ASSERT_NE(status2.StatusCode(), SUCCESS);
}

TEST_F(TestParseReply, test_reply_type_not_set_FAIL) {
  nlohmann::json js = R"(
    {"instances":[
        {
          "key_tag":"scalar",
          "key_int": 1,
          "key_bool": false,
          "key_float": 2.3,
          "key_str": "ut_test",
          "key_bytes": {"b64": "dXRfdGVzdA==", "type": "bytes"}
        }
      ]
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status(INVALID_INPUTS);
  status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_EQ(status.StatusCode(), SUCCESS);

  nlohmann::json out_js;
  proto::PredictReply reply;
  auto instance_ptr = reply.add_instances();
  auto &map_item = *(instance_ptr->mutable_items());
  // test scalar:
  // scalar:key_int
  proto::Tensor tensor_int;
  ProtoTensor pb_tensor_int(&tensor_int);
  pb_tensor_int.set_shape({1});
  pb_tensor_int.resize_data(pb_tensor_int.GetTypeSize(kMSI_Int32));
  auto data_int = reinterpret_cast<int32_t *>(pb_tensor_int.mutable_data());
  *data_int = 1;
  map_item["key_int"] = tensor_int;

  Status status2 = restful_service.ParseReply(reply, &out_js);
  ASSERT_NE(status2.StatusCode(), SUCCESS);
}

TEST_F(TestParseReply, test_reply_type_fp16_FAIL) {
  nlohmann::json js = R"(
    {"instances":[
        {
          "key_tag":"scalar",
          "key_int": 1,
          "key_bool": false,
          "key_float": 2.3,
          "key_str": "ut_test",
          "key_bytes": {"b64": "dXRfdGVzdA==", "type": "bytes"}
        }
      ]
    }
  )"_json;

  struct evhttp_request *request = new evhttp_request();
  int size = 100;
  std::shared_ptr<DecomposeEvRequest> request_msg = std::make_shared<DecomposeEvRequest>(request, size);
  request_msg->request_message_ = js;
  std::shared_ptr<RestfulRequest> restful_request = std::make_shared<RestfulRequest>(request_msg);
  proto::PredictRequest predict_request;
  std::shared_ptr<Dispatcher> dispatcher_ = Server::Instance().GetDispatcher();
  RestfulService restful_service(dispatcher_);
  Status status(INVALID_INPUTS);
  status = restful_service.ParseRequest(restful_request, &predict_request);
  ASSERT_EQ(status.StatusCode(), SUCCESS);

  nlohmann::json out_js;
  proto::PredictReply reply;
  auto instance_ptr = reply.add_instances();
  auto &map_item = *(instance_ptr->mutable_items());
  // test scalar:
  // scalar: key_float
  proto::Tensor tensor_float;
  ProtoTensor pb_tensor_float(&tensor_float);
  DataType type_float = kMSI_Float16;
  pb_tensor_float.set_data_type(type_float);
  pb_tensor_float.set_shape({1});
  pb_tensor_float.resize_data(pb_tensor_float.GetTypeSize(type_float));
  map_item["key_float16"] = tensor_float;

  Status status2 = restful_service.ParseReply(reply, &out_js);
  ASSERT_NE(status2.StatusCode(), SUCCESS);
}
}  // namespace serving
}  // namespace mindspore
