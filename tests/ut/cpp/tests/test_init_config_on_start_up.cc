/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "common/tensor_base.h"
#define private public
#include "worker/distributed_worker/distributed_model_loader.h"
#undef private

using std::string;
using std::vector;
namespace mindspore {
namespace serving {
class TestParseRankTableFile : public UT::Common {
 public:
  TestParseRankTableFile() = default;
  virtual void SetUp() {}
  virtual void TearDown() {
    for (auto &item : config_file_list_) {
      remove(item.c_str());
    }
    UT::Common::TearDown();
  }
  std::set<std::string> config_file_list_;
};

TEST_F(TestParseRankTableFile, test_init_config_on_startup_empty_file_failed) {
  std::string empty_rank_table_file = "empty_rank_table_file";
  std::ofstream fp(empty_rank_table_file);
  fp << "empty rank table file";
  fp.close();
  config_file_list_.emplace(empty_rank_table_file);
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->InitConfigOnStartup(empty_rank_table_file);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_init_config_on_startup_success) {
  nlohmann::json rank_table_server_list = R"(
  {
      "server_list": [
          {
              "server_id": "10.155.111.140",
              "device": [
                  {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                  {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"}],
               "host_nic_ip": "reserve"
          }
      ],
      "status": "completed"
  }
  )"_json;
  std::string rank_table_file = "rank_table_file";
  std::ofstream fp(rank_table_file);
  fp << rank_table_server_list;
  fp.close();
  config_file_list_.emplace(rank_table_file);
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->InitConfigOnStartup(rank_table_file);
  ASSERT_EQ(status.StatusCode(), SUCCESS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_server_list_success) {
  nlohmann::json rank_table_server_list = R"(
  {
      "version": "1.0",
      "server_count": "1",
      "server_list": [
          {
              "server_id": "10.155.111.140",
              "device": [
                  {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                  {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"},
                  {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                  {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"},
                  {"device_id": "4","device_ip": "192.1.27.7","rank_id": "4"},
                  {"device_id": "5","device_ip": "192.2.27.7","rank_id": "5"},
                  {"device_id": "6","device_ip": "192.3.27.7","rank_id": "6"},
                  {"device_id": "7","device_ip": "192.4.27.7","rank_id": "7"}],
               "host_nic_ip": "reserve"
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithServerList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), SUCCESS);
  ASSERT_EQ(servable->config_.rank_list.size(), 8);
  uint32_t expect_device_id = 0;
  for (auto &one_rank_config : servable->config_.rank_list) {
    std::string server_ip = one_rank_config.ip;
    uint32_t device_id = one_rank_config.device_id;
    ASSERT_EQ(server_ip, "10.155.111.140");
    ASSERT_EQ(device_id, expect_device_id);
    expect_device_id++;
  }
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_not_server_list_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "version": "1.0",
      "server_count": "1",
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithServerList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_invalid_server_list_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "version": "1.0",
      "server_count": "1",
      "server_list": "0",
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithServerList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_empty_server_list_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "version": "1.0",
      "server_count": "1",
      "server_list": [],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithServerList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_not_server_id_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "version": "1.0",
      "server_count": "1",
      "server_list": [
          {
              "device": [
                  {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                  {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"}],
               "host_nic_ip": "reserve"
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithServerList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_invalid_server_id_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "version": "1.0",
      "server_count": "1",
      "server_list": [
          {
              "server_id": [],
              "device": [
                  {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                  {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"}],
               "host_nic_ip": "reserve"
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithServerList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_empty_server_id_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "version": "1.0",
      "server_count": "1",
      "server_list": [
          {
              "server_id": "",
              "device": [
                  {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                  {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"}],
               "host_nic_ip": "reserve"
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithServerList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_not_device_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "version": "1.0",
      "server_count": "1",
      "server_list": [
          {
              "server_id": "10.155.111.140",
               "host_nic_ip": "reserve"
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithServerList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_invalid_device_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "version": "1.0",
      "server_count": "1",
      "server_list": [
          {
              "server_id": "10.155.111.140",
              "device": "dsfds",
               "host_nic_ip": "reserve"
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithServerList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_empty_device_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "version": "1.0",
      "server_count": "1",
      "server_list": [
          {
              "server_id": "10.155.111.140",
              "device": [],
               "host_nic_ip": "reserve"
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithServerList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_not_device_id_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "version": "1.0",
      "server_count": "1",
      "server_list": [
          {
              "server_id": "10.155.111.140",
              "device": [
                  {"device_ip": "192.1.27.6","rank_id": "0"}],
               "host_nic_ip": "reserve"
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithServerList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_invalid_device_id_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "version": "1.0",
      "server_count": "1",
      "server_list": [
          {
              "server_id": "",
              "device": [
                  {"device_id": "1wdb","device_ip": "192.1.27.6","rank_id": "0"}],
               "host_nic_ip": "reserve"
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithServerList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_not_rank_id_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "version": "1.0",
      "server_count": "1",
      "server_list": [
          {
              "server_id": "10.155.111.140",
              "device": [
                  {"device_id": "0","device_ip": "192.1.27.6"}],
               "host_nic_ip": "reserve"
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithServerList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_invalid_rank_id_failed1) {
  nlohmann::json rank_table_server_list = R"(
  {
      "version": "1.0",
      "server_count": "1",
      "server_list": [
          {
              "server_id": "",
              "device": [
                  {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0wer"}],
               "host_nic_ip": "reserve"
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithServerList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_invalid_rank_id_failed2) {
  nlohmann::json rank_table_server_list = R"(
  {
      "version": "1.0",
      "server_count": "1",
      "server_list": [
          {
              "server_id": "",
              "device": [
                  {"device_id": "0","device_ip": "192.1.27.6","rank_id": "5"}],
               "host_nic_ip": "reserve"
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithServerList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_group_list_success) {
  nlohmann::json rank_table_group_list = R"(
  {
      "board_id": "0x0000",
      "chip_info": "910",
      "deploy_mode": "lab",
      "group_count": "1",
      "group_list": [
          {
              "device_num": "2",
              "server_num": "1",
              "group_name": "",
              "instance_count": "2",
              "instance_list": [
                  {
                      "devices": [{"device_id": "0","device_ip": "192.1.27.6"}],
                      "rank_id": "0",
                      "server_id": "10.155.111.140"
                  },
                  {
                      "devices": [{"device_id": "1","device_ip": "192.2.27.6"}],
                      "rank_id": "1",
                      "server_id": "10.155.111.140"
                  }
              ]
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithGroupList("rank_table_file", rank_table_group_list);
  ASSERT_EQ(status.StatusCode(), SUCCESS);
  ASSERT_EQ(servable->config_.rank_list.size(), 2);
  uint32_t expect_device_id = 0;
  for (auto &one_rank_config : servable->config_.rank_list) {
    std::string server_ip = one_rank_config.ip;
    uint32_t device_id = one_rank_config.device_id;
    ASSERT_EQ(server_ip, "10.155.111.140");
    ASSERT_EQ(device_id, expect_device_id);
    expect_device_id++;
  }
}
TEST_F(TestParseRankTableFile, test_parse_rank_table_file_not_group_list_failed) {
  nlohmann::json rank_table_group_list = R"(
  {
      "board_id": "0x0000",
      "chip_info": "910",
      "deploy_mode": "lab",
      "group_count": "1",
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithGroupList("rank_table_file", rank_table_group_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_invalid_group_list_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "board_id": "0x0000",
      "chip_info": "910",
      "group_count": "1",
      "group_list": "0",
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithGroupList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_empty_group_list_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "board_id": "0x0000",
      "chip_info": "910",
      "group_count": "1",
      "group_list": [],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithGroupList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_not_instance_list_failed) {
  nlohmann::json rank_table_group_list = R"(
  {
      "board_id": "0x0000",
      "chip_info": "910",
      "deploy_mode": "lab",
      "group_count": "1",
      "group_list": [
          {
              "server_num": "1"
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithGroupList("rank_table_file", rank_table_group_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_invalid_instance_list_failed) {
  nlohmann::json rank_table_group_list = R"(
  {
      "board_id": "0x0000",
      "chip_info": "910",
      "deploy_mode": "lab",
      "group_count": "1",
      "group_list": [
          {
              "server_num": "1",
              "instance_list": "0"
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithGroupList("rank_table_file", rank_table_group_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_empty_instance_list_failed) {
  nlohmann::json rank_table_group_list = R"(
  {
      "board_id": "0x0000",
      "chip_info": "910",
      "deploy_mode": "lab",
      "group_count": "1",
      "group_list": [
          {
              "server_num": "1",
              "instance_list": []
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithGroupList("rank_table_file", rank_table_group_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_group_list_not_server_id_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "board_id": "0x0000",
      "group_list": [
          {
              "instance_count": "1",
              "instance_list": [
                  {
                      "devices": [{"device_id": "0","device_ip": "192.1.27.6"}],
                      "rank_id": "0"
                  }
              ]
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithGroupList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_group_list_invalid_server_id_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "board_id": "0x0000",
      "group_list": [
          {
              "instance_count": "1",
              "instance_list": [
                  {
                      "devices": [{"device_id": "0","device_ip": "192.1.27.6"}],
                      "rank_id": "0",
                      "server_id": []
                  }
              ]
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithGroupList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_group_list_empty_server_id_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "board_id": "0x0000",
      "group_list": [
          {
              "instance_count": "1",
              "instance_list": [
                  {
                      "devices": [{"device_id": "0","device_ip": "192.1.27.6"}],
                      "rank_id": "0",
                      "server_id": ""
                  }
              ]
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithGroupList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_group_list_not_devices_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "board_id": "0x0000",
      "group_list": [
          {
              "instance_count": "1",
              "instance_list": [
                  {
                      "rank_id": "0",
                      "server_id": "10.155.111.140"
                  }
              ]
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithGroupList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_group_list_invalid_devices_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "board_id": "0x0000",
      "group_list": [
          {
              "instance_count": "1",
              "instance_list": [
                  {
                      "devices": "rtrt",
                      "rank_id": "0",
                      "server_id": "10.155.111.140"
                  }
              ]
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithGroupList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_group_list_empty_devices_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "board_id": "0x0000",
      "group_list": [
          {
              "instance_count": "1",
              "instance_list": [
                  {
                      "devices": [],
                      "rank_id": "0",
                      "server_id": "10.155.111.140"
                  }
              ]
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithGroupList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_group_list_not_device_id_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "board_id": "0x0000",
      "group_list": [
          {
              "instance_count": "1",
              "instance_list": [
                  {
                      "devices": [{"device_ip": "192.1.27.6"}],
                      "rank_id": "0",
                      "server_id": "10.155.111.140"
                  }
              ]
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithGroupList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_group_list_invalid_device_id_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "board_id": "0x0000",
      "group_list": [
          {
              "instance_count": "1",
              "instance_list": [
                  {
                      "devices": [{"device_id": "wd1gt2", "device_ip": "192.1.27.6"}],
                      "rank_id": "0",
                      "server_id": "10.155.111.140"
                  }
              ]
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithGroupList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_group_list_not_rank_id_failed) {
  nlohmann::json rank_table_server_list = R"(
  {
      "board_id": "0x0000",
      "group_list": [
          {
              "instance_count": "1",
              "instance_list": [
                  {
                      "devices": [{"device_id": "0", "device_ip": "192.1.27.6"}],
                      "server_id": "10.155.111.140"
                  }
              ]
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithGroupList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_group_list_invalid_rank_id_failed1) {
  nlohmann::json rank_table_server_list = R"(
  {
      "board_id": "0x0000",
      "group_list": [
          {
              "instance_count": "1",
              "instance_list": [
                  {
                      "devices": [{"device_id": "0", "device_ip": "192.1.27.6"}],
                      "rank_id": "tfdg5",
                      "server_id": "10.155.111.140"
                  }
              ]
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithGroupList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

TEST_F(TestParseRankTableFile, test_parse_rank_table_file_with_group_list_invalid_rank_id_failed2) {
  nlohmann::json rank_table_server_list = R"(
  {
      "board_id": "0x0000",
      "group_list": [
          {
              "instance_count": "1",
              "instance_list": [
                  {
                      "devices": [{"device_id": "0", "device_ip": "192.1.27.6"}],
                      "rank_id": "7",
                      "server_id": "10.155.111.140"
                  }
              ]
          }
      ],
      "status": "completed"
  }
  )"_json;
  auto servable = std::make_shared<DistributedModelLoader>();
  auto status = servable->ParserRankTableWithGroupList("rank_table_file", rank_table_server_list);
  ASSERT_EQ(status.StatusCode(), INVALID_INPUTS);
}

}  // namespace serving
}  // namespace mindspore
