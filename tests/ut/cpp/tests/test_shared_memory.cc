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

#include "tests/ut/cpp/common/test_servable_common.h"
#include "common/shared_memory.h"

#define private public
#undef private

using std::string;
using std::vector;
namespace mindspore {
namespace serving {

class TestSharedMemory : public UT::Common {
 public:
  void SetUp() override {
    UT::Common::SetUp();
  }
  void TearDown() override {
    UT::Common::TearDown();
  }
};

TEST_F(TestSharedMemory, test_alloc_release_shared_memory_success) {
  SharedMemoryAllocator allocator;
  std::string memory_key_prefix = "test_memory_key";
  uint64_t item_size = 64;
  auto status = allocator.NewMemoryBuffer(memory_key_prefix, item_size, 3);
  ASSERT_TRUE(status == SUCCESS);
  std::string first_memory_key;
  std::vector<SharedMemoryItem> first_shm_list;
  for (int i = 0; i < 3; i++) {
    SharedMemoryItem shm_item;
    status = allocator.AllocMemoryItem(memory_key_prefix, &shm_item);
    ASSERT_TRUE(status == SUCCESS);
    ASSERT_EQ(shm_item.memory_key_prefix, memory_key_prefix);
    ASSERT_EQ(shm_item.size, item_size);
    ASSERT_TRUE(shm_item.memory_key.find(memory_key_prefix) != std::string::npos);
    if (first_memory_key.empty()) {
      first_memory_key = shm_item.memory_key;
    } else {
      ASSERT_EQ(first_memory_key, shm_item.memory_key);
    }
    first_shm_list.push_back(shm_item);
  }
  // new shared memory
  std::string second_memory_key;
  std::vector<SharedMemoryItem> second_shm_list;
  for (int i = 0; i < 3; i++) {
    SharedMemoryItem shm_item;
    status = allocator.AllocMemoryItem(memory_key_prefix, &shm_item);
    ASSERT_TRUE(status == SUCCESS);
    ASSERT_EQ(shm_item.memory_key_prefix, memory_key_prefix);
    ASSERT_EQ(shm_item.size, item_size);
    ASSERT_TRUE(shm_item.memory_key.find(memory_key_prefix) != std::string::npos);
    if (second_memory_key.empty()) {
      second_memory_key = shm_item.memory_key;
    } else {
      ASSERT_EQ(second_memory_key, shm_item.memory_key);
    }
    ASSERT_NE(second_memory_key, first_memory_key);
    second_shm_list.push_back(shm_item);
  }
  // free shared memory and alloc
  {
    auto &free_memory = second_shm_list[1];
    allocator.ReleaseMemoryItem(free_memory);
    SharedMemoryItem shm_item;
    status = allocator.AllocMemoryItem(memory_key_prefix, &shm_item);
    ASSERT_TRUE(status == SUCCESS);
    ASSERT_EQ(shm_item.memory_key, free_memory.memory_key);
    ASSERT_EQ(shm_item.bytes_size, free_memory.bytes_size);
    ASSERT_EQ(shm_item.offset_address, free_memory.offset_address);
    ASSERT_EQ(shm_item.offset, free_memory.offset);
  }
  {
    auto &free_memory = first_shm_list[1];
    allocator.ReleaseMemoryItem(free_memory);
    SharedMemoryItem shm_item;
    status = allocator.AllocMemoryItem(memory_key_prefix, &shm_item);
    ASSERT_TRUE(status == SUCCESS);
    ASSERT_EQ(shm_item.memory_key, free_memory.memory_key);
    ASSERT_EQ(shm_item.bytes_size, free_memory.bytes_size);
    ASSERT_EQ(shm_item.offset_address, free_memory.offset_address);
    ASSERT_EQ(shm_item.offset, free_memory.offset);
  }
}

TEST_F(TestSharedMemory, test_alloc_release_shared_memory_repeat_release_failed) {
  SharedMemoryAllocator allocator;
  std::string memory_key_prefix = "test_memory_key";
  uint64_t item_size = 64;
  auto status = allocator.NewMemoryBuffer(memory_key_prefix, item_size, 3);
  ASSERT_TRUE(status == SUCCESS);
  SharedMemoryItem shm_item;
  status = allocator.AllocMemoryItem(memory_key_prefix, &shm_item);
  ASSERT_TRUE(status == SUCCESS);
  allocator.ReleaseMemoryItem(shm_item);
  try {
    allocator.ReleaseMemoryItem(shm_item);
    FAIL();
  } catch (std::runtime_error &ex) {
    std::string error_msg = ex.what();
    auto index = error_msg.find("Shared memory " + shm_item.memory_key + " has already been in free set, offset: ");
    ASSERT_TRUE(index != std::string::npos);
  }
}

TEST_F(TestSharedMemory, test_alloc_attach_shared_memory_success) {
  SharedMemoryAllocator allocator;
  std::string memory_key_prefix = "test_memory_key";
  uint64_t item_size = 64;
  auto status = allocator.NewMemoryBuffer(memory_key_prefix, item_size, 3);
  ASSERT_TRUE(status == SUCCESS);
  SharedMemoryItem shm_item;
  status = allocator.AllocMemoryItem(memory_key_prefix, &shm_item);
  ASSERT_TRUE(status == SUCCESS);
  SharedMemoryManager attach;
  SharedMemoryAttachItem attach_item;
  status = attach.Attach(shm_item.memory_key, shm_item.bytes_size, shm_item.offset, shm_item.size, &attach_item);
  ASSERT_TRUE(status == SUCCESS);
  ASSERT_NE(shm_item.offset_address, attach_item.offset_address);
  attach_item.offset_address[0] = 0xfe;
  ASSERT_EQ(0xfe, shm_item.offset_address[0]);

  shm_item.offset_address[1] = 0xfa;
  ASSERT_EQ(0xfa, attach_item.offset_address[1]);
  attach.Detach(attach_item.memory_key);
}

TEST_F(TestSharedMemory, test_alloc_twice_attach_shared_memory_success) {
  SharedMemoryAllocator allocator;
  std::string memory_key_prefix = "test_memory_key";
  uint64_t item_size = 64;
  auto status = allocator.NewMemoryBuffer(memory_key_prefix, item_size, 3);
  ASSERT_TRUE(status == SUCCESS);

  SharedMemoryManager attach;
  std::string memory_key;
  // first memory item
  {
    SharedMemoryItem shm_item;
    status = allocator.AllocMemoryItem(memory_key_prefix, &shm_item);
    ASSERT_TRUE(status == SUCCESS);
    SharedMemoryAttachItem attach_item;
    status = attach.Attach(shm_item.memory_key, shm_item.bytes_size, shm_item.offset, shm_item.size, &attach_item);
    ASSERT_TRUE(status == SUCCESS);
    ASSERT_NE(shm_item.offset_address, attach_item.offset_address);
    attach_item.offset_address[0] = 0xfe;
    ASSERT_EQ(0xfe, shm_item.offset_address[0]);
    shm_item.offset_address[1] = 0xfa;
    ASSERT_EQ(0xfa, attach_item.offset_address[1]);
    memory_key = shm_item.memory_key;
  }
  // second memory item
  {
    SharedMemoryItem shm_item;
    status = allocator.AllocMemoryItem(memory_key_prefix, &shm_item);
    ASSERT_TRUE(status == SUCCESS);
    SharedMemoryAttachItem attach_item;
    status = attach.Attach(shm_item.memory_key, shm_item.bytes_size, shm_item.offset, shm_item.size, &attach_item);
    ASSERT_TRUE(status == SUCCESS);
    ASSERT_NE(shm_item.offset_address, attach_item.offset_address);
    attach_item.offset_address[3] = 0xfe;
    ASSERT_EQ(0xfe, shm_item.offset_address[3]);
    shm_item.offset_address[4] = 0xfa;
    ASSERT_EQ(0xfa, attach_item.offset_address[4]);
  }
  attach.Detach(memory_key);
}

TEST_F(TestSharedMemory, test_alloc_re_attach_shared_memory_success) {
  SharedMemoryAllocator allocator;
  std::string memory_key_prefix = "test_memory_key";
  uint64_t item_size = 64;
  auto status = allocator.NewMemoryBuffer(memory_key_prefix, item_size, 3);
  ASSERT_TRUE(status == SUCCESS);

  SharedMemoryManager attach;
  // first memory item
  {
    SharedMemoryItem shm_item;
    status = allocator.AllocMemoryItem(memory_key_prefix, &shm_item);
    ASSERT_TRUE(status == SUCCESS);
    SharedMemoryAttachItem attach_item;
    status = attach.Attach(shm_item.memory_key, shm_item.bytes_size, shm_item.offset, shm_item.size, &attach_item);
    ASSERT_TRUE(status == SUCCESS);
    ASSERT_NE(shm_item.offset_address, attach_item.offset_address);
    attach_item.offset_address[0] = 0xfe;
    ASSERT_EQ(0xfe, shm_item.offset_address[0]);
    shm_item.offset_address[1] = 0xfa;
    ASSERT_EQ(0xfa, attach_item.offset_address[1]);
    attach.Detach(shm_item.memory_key);
  }
  // second memory item
  {
    SharedMemoryItem shm_item;
    status = allocator.AllocMemoryItem(memory_key_prefix, &shm_item);
    ASSERT_TRUE(status == SUCCESS);
    SharedMemoryAttachItem attach_item;
    status = attach.Attach(shm_item.memory_key, shm_item.bytes_size, shm_item.offset, shm_item.size, &attach_item);
    ASSERT_TRUE(status == SUCCESS);
    ASSERT_NE(shm_item.offset_address, attach_item.offset_address);
    attach_item.offset_address[3] = 0xfe;
    ASSERT_EQ(0xfe, shm_item.offset_address[3]);
    shm_item.offset_address[4] = 0xfa;
    ASSERT_EQ(0xfa, attach_item.offset_address[4]);
    attach.Detach(shm_item.memory_key);
  }
}

TEST_F(TestSharedMemory, test_alloc_attach_shared_memory_attach_repeat_success) {
  SharedMemoryAllocator allocator;
  std::string memory_key_prefix = "test_memory_key";
  uint64_t item_size = 64;
  auto status = allocator.NewMemoryBuffer(memory_key_prefix, item_size, 3);
  ASSERT_TRUE(status == SUCCESS);
  SharedMemoryItem shm_item;
  status = allocator.AllocMemoryItem(memory_key_prefix, &shm_item);
  ASSERT_TRUE(status == SUCCESS);
  SharedMemoryManager attach;
  SharedMemoryAttachItem attach_item;
  status = attach.Attach(shm_item.memory_key, shm_item.bytes_size, shm_item.offset, shm_item.size, &attach_item);
  ASSERT_TRUE(status == SUCCESS);
  SharedMemoryAttachItem attach_item2;
  status = attach.Attach(shm_item.memory_key, shm_item.bytes_size, shm_item.offset, shm_item.size, &attach_item2);
  ASSERT_TRUE(status == SUCCESS);
  ASSERT_EQ(attach_item.offset_address, attach_item2.offset_address);
}


TEST_F(TestSharedMemory, test_alloc_attach_shared_memory_detach_repeat_failed) {
  SharedMemoryAllocator allocator;
  std::string memory_key_prefix = "test_memory_key";
  uint64_t item_size = 64;
  auto status = allocator.NewMemoryBuffer(memory_key_prefix, item_size, 3);
  ASSERT_TRUE(status == SUCCESS);
  SharedMemoryItem shm_item;
  status = allocator.AllocMemoryItem(memory_key_prefix, &shm_item);
  ASSERT_TRUE(status == SUCCESS);
  SharedMemoryManager attach;
  SharedMemoryAttachItem attach_item;
  status = attach.Attach(shm_item.memory_key, shm_item.bytes_size, shm_item.offset, shm_item.size, &attach_item);
  ASSERT_TRUE(status == SUCCESS);
  status = attach.Detach(shm_item.memory_key);
  ASSERT_TRUE(status == SUCCESS);
  status = attach.Detach(shm_item.memory_key);
  ASSERT_TRUE(status != SUCCESS);
}

TEST_F(TestSharedMemory, test_alloc_attach_invalid_shared_memory_failed) {
  SharedMemoryAllocator allocator;
  std::string memory_key_prefix = "test_memory_key";
  uint64_t item_size = 64;
  auto status = allocator.NewMemoryBuffer(memory_key_prefix, item_size, 1);
  ASSERT_TRUE(status == SUCCESS);
  SharedMemoryItem shm_item;
  status = allocator.AllocMemoryItem(memory_key_prefix, &shm_item);
  ASSERT_TRUE(status == SUCCESS);

  SharedMemoryManager attach;
  SharedMemoryAttachItem attach_item;
  // invalid memory key
  status = attach.Attach("invalid memory key", shm_item.bytes_size, shm_item.offset, shm_item.size, &attach_item);
  ASSERT_TRUE(status != SUCCESS);

  // invalid memory bytes size
  status = attach.Attach(shm_item.memory_key, 0,  shm_item.offset, shm_item.size, &attach_item);
  ASSERT_TRUE(status != SUCCESS);

  // invalid memory data offset
  status = attach.Attach(shm_item.memory_key, shm_item.bytes_size, shm_item.bytes_size, shm_item.size, &attach_item);
  ASSERT_TRUE(status != SUCCESS);

  // invalid memory data size
  status = attach.Attach(shm_item.memory_key, shm_item.bytes_size, 0, shm_item.bytes_size + 1, &attach_item);
  ASSERT_TRUE(status != SUCCESS);

  // success
  status = attach.Attach(shm_item.memory_key, shm_item.bytes_size,  shm_item.offset, shm_item.size, &attach_item);
  ASSERT_TRUE(status == SUCCESS);
}

}  // namespace serving
}  // namespace mindspore
