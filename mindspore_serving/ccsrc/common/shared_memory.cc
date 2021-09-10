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

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "common/shared_memory.h"

namespace mindspore {
namespace serving {

SharedMemoryAllocator &SharedMemoryAllocator::Instance() {
  static SharedMemoryAllocator instance;
  return instance;
}

SharedMemoryAllocator::SharedMemoryAllocator() = default;
SharedMemoryAllocator::~SharedMemoryAllocator() {
  std::unique_lock<std::mutex> lock(lock_);
  for (auto &item : memory_map_) {
    auto &group = item.second;
    for (auto &shm : group.shm_map) {
      auto ret = munmap(shm.second.address, shm.second.bytes_size);
      if (ret == -1) {
        MSI_LOG_ERROR << "Failed to munmap, memory key: " << shm.second.memory_key;
      }
      ret = shm_unlink(shm.second.memory_key.c_str());
      if (ret == -1) {
        MSI_LOG_ERROR << "Failed to shm_unlink " << shm.second.memory_key << ", errno: " << errno;
      }
    }
  }
  memory_map_.clear();
}

Status SharedMemoryAllocator::AddShmMemoryBuffer(SharedMemoryGroup *shm_group) {
  auto item_size = shm_group->item_size;
  auto item_count = shm_group->item_count;
  auto memory_key = shm_group->memory_key_prefix + "_" + std::to_string(shm_group->shm_map.size());
  // maximum 4GB memory
  if (item_size == 0 || item_count == 0 || UINT32_MAX / item_size < item_count) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Invalid item size or item count, item size: " << item_size
                                          << ", item count :" << item_count << ", memory key: " << memory_key;
  }
  auto align_item_size = (item_size + 7) / 8 * 8;
  auto shm_fd = shm_open(memory_key.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (shm_fd == -1) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Failed to shm_open " << memory_key << " , errno: " << errno;
  }

  uint64_t memory_size = align_item_size * item_count;
  auto ret = ftruncate(shm_fd, memory_size);
  if (ret == -1) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Failed to ftruncate " << memory_key << ", errno: " << errno << ", memory size: " << memory_size;
  }
  auto address = mmap(nullptr, memory_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (address == MAP_FAILED) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Failed to mmap " << memory_key << ", errno: " << errno << ", memory size: " << memory_size;
  }
  ret = close(shm_fd);
  if (ret == -1) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Failed to close " << memory_key << ", errno: " << errno;
  }
  SharedMemory &shm = shm_group->shm_map[memory_key];
  shm.memory_key = memory_key;
  shm.address = reinterpret_cast<uint8_t *>(address);
  shm.bytes_size = memory_size;
  uint64_t offset = 0;
  for (uint64_t i = 0; i < item_count; i++) {
    shm.free_queue.emplace(offset);
    offset += align_item_size;
  }
  shm_group->free_count += item_count;

  MSI_LOG_INFO << "New shared memory success, memory key: " << memory_key << ", bytes size: " << memory_size
               << ", item count: " << item_count;
  return SUCCESS;
}

Status SharedMemoryAllocator::NewMemoryBuffer(const std::string &memory_key_prefix, uint64_t item_size,
                                              uint64_t item_count) {
  std::unique_lock<std::mutex> lock(lock_);
  if (memory_map_.find(memory_key_prefix) != memory_map_.end()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Shared memory has already been inited";
  }
  auto &group = memory_map_[memory_key_prefix];
  group.memory_key_prefix = memory_key_prefix;
  group.item_size = item_size;
  group.item_count = item_count;
  group.free_count = 0;
  auto status = AddShmMemoryBuffer(&group);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Alloc shared memory failed, memory key prefix: " << memory_key_prefix;
    return status;
  }
  return SUCCESS;
}

Status SharedMemoryAllocator::AllocMemoryItem(const std::string &memory_key_prefix, SharedMemoryItem *shm_item) {
  std::unique_lock<std::mutex> lock(lock_);
  auto it = memory_map_.find(memory_key_prefix);
  if (it == memory_map_.end()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Cannot find shared memory " << memory_key_prefix;
  }
  auto &group = it->second;
  if (group.free_count == 0) {
    auto status = AddShmMemoryBuffer(&group);
    if (status != SUCCESS) {
      MSI_LOG_ERROR << "Alloc shared memory failed, memory key prefix: " << memory_key_prefix;
      return SUCCESS;
    }
  }
  for (auto &item : group.shm_map) {
    auto &shm = item.second;
    if (!shm.free_queue.empty()) {
      shm_item->memory_key_prefix = memory_key_prefix;
      shm_item->memory_key = shm.memory_key;
      shm_item->bytes_size = shm.bytes_size;
      shm_item->offset = *shm.free_queue.begin();
      shm_item->offset_address = shm.address + shm_item->offset;
      shm_item->size = group.item_size;
      shm.free_queue.erase(shm_item->offset);
      group.free_count -= 1;
      return SUCCESS;
    }
  }
  MSI_LOG_EXCEPTION << "There is no free shared memory";
}

void SharedMemoryAllocator::ReleaseMemoryItem(const SharedMemoryItem &shm_item) {
  std::unique_lock<std::mutex> lock(lock_);
  auto it = memory_map_.find(shm_item.memory_key_prefix);
  if (it == memory_map_.end()) {
    MSI_LOG_WARNING << "Cannot find shared memory prefix " << shm_item.memory_key_prefix;
    return;
  }
  auto shm_it = it->second.shm_map.find(shm_item.memory_key);
  if (shm_it == it->second.shm_map.end()) {
    MSI_LOG_WARNING << "Cannot find shared memory " << shm_item.memory_key;
    return;
  }
  if (shm_it->second.free_queue.count(shm_item.offset) > 0) {
    MSI_LOG_EXCEPTION << "Shared memory " << shm_item.memory_key
                      << " has already been in free set, offset: " << shm_item.offset;
  }
  shm_it->second.free_queue.emplace(shm_item.offset);
  it->second.free_count += 1;
}

ShmTensor::ShmTensor(DataType type, std::vector<int64_t> shape, const SharedMemoryItem &shm_item)
    : BufferTensor(type, shape, shm_item.offset_address, shm_item.size, false), shm_info_(shm_item) {}

ShmTensor::~ShmTensor() { SharedMemoryAllocator::Instance().ReleaseMemoryItem(shm_info_); }

ShmAttachTensor::ShmAttachTensor(DataType type, std::vector<int64_t> shape, const SharedMemoryAttachItem &item)
    : BufferTensor(type, shape, item.offset_address, item.size, false), shm_info_(item) {}

ShmAttachTensor::~ShmAttachTensor() = default;

SharedMemoryManager &SharedMemoryManager::Instance() {
  static SharedMemoryManager instance;
  return instance;
}

SharedMemoryManager::SharedMemoryManager() {}
SharedMemoryManager::~SharedMemoryManager() {
  std::unique_lock<std::mutex> lock(lock_);
  for (auto &item : attached_shm_list_) {
    auto ret = munmap(item.address, item.bytes_size);
    if (ret == -1) {
      MSI_LOG_ERROR << "Failed to munmap, memory key: " << item.memory_key;
    }
  }
  attached_shm_list_.clear();
}

Status SharedMemoryManager::Attach(const std::string &memory_key, uint64_t bytes_size, uint64_t data_offset,
                                   uint64_t data_size, SharedMemoryAttachItem *shm_info) {
  if (data_size > bytes_size || data_offset > bytes_size - data_size) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Invalid memory size info, memory key: " << memory_key << ", bytes size: " << bytes_size
           << ", data offset: " << data_offset << ", data size: " << data_size;
  }
  SharedMemoryAttach attach_mem;
  auto status = Attach(memory_key, bytes_size, &attach_mem);
  if (status != SUCCESS) {
    return status;
  }
  shm_info->memory_key = attach_mem.memory_key;
  shm_info->offset_address = attach_mem.address + data_offset;
  shm_info->offset = data_offset;
  shm_info->size = data_size;
  return SUCCESS;
}

Status SharedMemoryManager::Detach(const std::string &memory_key) {
  std::unique_lock<std::mutex> lock(lock_);
  auto it = std::find_if(attached_shm_list_.begin(), attached_shm_list_.end(),
                         [&memory_key](const SharedMemoryAttach &item) { return memory_key == item.memory_key; });
  if (it == attached_shm_list_.end()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Cannot find shared memory " << memory_key;
  }
  auto ret = munmap(it->address, it->bytes_size);
  if (ret == -1) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Failed to munmap, memory key: " << memory_key;
  }
  attached_shm_list_.erase(it);
  return SUCCESS;
}

Status SharedMemoryManager::Attach(const std::string &memory_key, uint64_t bytes_size, SharedMemoryAttach *attach_mem) {
  std::unique_lock<std::mutex> lock(lock_);
  for (auto &item : attached_shm_list_) {
    if (item.memory_key == memory_key) {
      *attach_mem = item;
      return SUCCESS;
    }
  }
  auto shm_fd = shm_open(memory_key.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
  if (shm_fd == -1) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Failed to shm_open " << memory_key << " , errno: " << errno;
  }
  auto address = mmap(nullptr, bytes_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (address == MAP_FAILED) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Failed to mmap " << memory_key << ", errno: " << errno << ", memory size: " << bytes_size;
  }
  auto ret = close(shm_fd);
  if (ret == -1) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Failed to close " << memory_key << ", errno: " << errno;
  }
  attach_mem->memory_key = memory_key;
  attach_mem->bytes_size = bytes_size;
  attach_mem->address = static_cast<uint8_t *>(address);
  attached_shm_list_.push_back(*attach_mem);
  return SUCCESS;
}

}  // namespace serving
}  // namespace mindspore
