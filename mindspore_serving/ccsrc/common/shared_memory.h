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

#ifndef MINDSPORE_SERVING_SHARED_MEMORY_H
#define MINDSPORE_SERVING_SHARED_MEMORY_H

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <queue>
#include <set>
#include <mutex>
#include "common/serving_common.h"
#include "common/buffer_tensor.h"

namespace mindspore {
namespace serving {

struct SharedMemoryItem {
  std::string memory_key_prefix;
  std::string memory_key;   // for shm_open
  uint64_t bytes_size = 0;  // for shm_open
  uint8_t *offset_address = nullptr;
  uint64_t offset = 0;
  uint64_t size = 0;
};

struct SharedMemory {
  std::string memory_key;
  uint64_t bytes_size = 0;
  uint8_t *address = nullptr;
  std::set<uint64_t> free_queue;
};

struct SharedMemoryGroup {
  std::map<std::string, SharedMemory> shm_map;
  std::string memory_key_prefix;
  uint64_t item_size = 0;
  uint64_t item_count = 0;
  uint64_t free_count = 0;
};

class SharedMemoryAllocator {
 public:
  static SharedMemoryAllocator &Instance();
  SharedMemoryAllocator();
  ~SharedMemoryAllocator();
  Status NewMemoryBuffer(const std::string &memory_key_prefix, uint64_t item_size, uint64_t init_item_count);
  Status AllocMemoryItem(const std::string &memory_key_prefix, SharedMemoryItem *shm_item);
  void ReleaseMemoryItem(const SharedMemoryItem &shm_item);

 private:
  std::map<std::string, SharedMemoryGroup> memory_map_;
  std::mutex lock_;
  Status AddShmMemoryBuffer(SharedMemoryGroup *shm_group);
};

class ShmTensor : public BufferTensor {
 public:
  ShmTensor(DataType type, std::vector<int64_t> shape, const SharedMemoryItem &shm_item);
  ~ShmTensor();

 private:
  SharedMemoryItem shm_info_;
};

struct SharedMemoryAttach {
  std::string memory_key;
  uint64_t bytes_size = 0;
  uint8_t *address = nullptr;
};

struct SharedMemoryAttachItem {
  std::string memory_key;  // for shm_open
  uint8_t *offset_address = nullptr;
  uint64_t offset = 0;
  uint64_t size = 0;
};

class SharedMemoryManager {
 public:
  static SharedMemoryManager &Instance();
  SharedMemoryManager();
  ~SharedMemoryManager();
  Status Attach(const std::string &memory_key, uint64_t bytes_size, uint64_t data_offset, uint64_t data_size,
                SharedMemoryAttachItem *shm_info);
  Status Detach(const std::string &memory_key);

 private:
  std::vector<SharedMemoryAttach> attached_shm_list_;
  std::mutex lock_;
  Status Attach(const std::string &memory_key, uint64_t bytes_size, SharedMemoryAttach *attach_mem);
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_SHARED_MEMORY_H
