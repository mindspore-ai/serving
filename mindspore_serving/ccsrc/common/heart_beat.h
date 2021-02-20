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

#ifndef MINDSPORE_SERVING_HEART_BEAT_H
#define MINDSPORE_SERVING_HEART_BEAT_H

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <condition_variable>
#include <thread>
#include <functional>
#include <chrono>
#include <utility>
#include "common/serving_common.h"
#include "common/grpc_server.h"
#include "proto/ms_service.pb.h"
#include "proto/ms_service.grpc.pb.h"
namespace mindspore::serving {
using TimerCallback = std::function<void()>;

class MS_API Timer {
 public:
  Timer() {}
  ~Timer() { StopTimer(); }
  void StartTimer(int64_t millisecond, TimerCallback callback) {
    auto timer_run = [this, millisecond, callback]() {
      std::unique_lock<std::mutex> lk(cv_m_);
      if (cv_.wait_for(lk, std::chrono::milliseconds(millisecond)) == std::cv_status::timeout) {
        callback();
      }
    };
    thread_ = std::thread(timer_run);
  }
  void StopTimer() {
    cv_.notify_one();
    if (thread_.joinable()) {
      try {
        thread_.join();
      } catch (const std::system_error &) {
      } catch (...) {
      }
    }
  }

 private:
  std::mutex cv_m_;
  std::thread thread_;
  std::condition_variable cv_;
};

template <class SendStub, class RecvStub>
class MS_API Watcher {
 public:
  explicit Watcher(const std::string host_address) { host_address_ = host_address; }
  void StartWatch(const std::string &address) {
    auto it = watchee_map_.find(address);
    if (it != watchee_map_.end()) {
      MSI_LOG(INFO) << "watchee exist: " << address;
      return;
    }
    WatcheeContext context;
    auto channel = GrpcServer::CreateChannel(address);
    context.stub_ = SendStub::NewStub(channel);
    context.timer_ = std::make_shared<Timer>();
    watchee_map_.insert(make_pair(address, context));
    MSI_LOG(INFO) << "Begin to send ping to " << address;
    // add timer
    watchee_map_[address].timer_->StartTimer(max_time_out_ / max_ping_times_,
                                             std::bind(&Watcher::RecvPongTimeOut, this, address));
    SendPing(address);
  }
  void StopWatch(const std::string &address) {
    // clear map and timer
    auto it = watchee_map_.find(address);
    if (it == watchee_map_.end()) {
      MSI_LOG(INFO) << "watchee not exist: " << address;
      return;
    }
    watchee_map_[address].timer_->StopTimer();
    watchee_map_.erase(address);
  }

  void SendPing(const std::string &address) {
    watchee_map_[address].timeouts_ += 1;
    // send async message
    PingAsync(address);
  }

  void RecvPing(const std::string &address) {
    // recv message
    if (watcher_map_.count(address)) {
      watcher_map_[address].timer_->StopTimer();
    } else {
      WatcherContext context;
      auto channel = GrpcServer::CreateChannel(address);
      context.stub_ = RecvStub::NewStub(channel);
      context.timer_ = std::make_shared<Timer>();
      watcher_map_.insert(make_pair(address, context));
      MSI_LOG(INFO) << "Begin to send pong to " << address;
    }
    // add timer
    watcher_map_[address].timer_->StartTimer(max_time_out_, std::bind(&Watcher::RecvPingTimeOut, this, address));
    // send async message
    PongAsync(address);
  }

  void RecvPong(const std::string &address) {
    // recv message
    if (watchee_map_.count(address)) {
      watchee_map_[address].timeouts_ = 0;
    } else {
      MSI_LOG(INFO) << "Recv Pong after timeout or stop";
    }
  }

  void RecvPongTimeOut(const std::string &address) {
    if (watchee_map_[address].timeouts_ >= max_ping_times_) {
      // add exit handle
      MSI_LOG(INFO) << "Recv Pong Time Out from " << address;
      watchee_map_.erase(address);
      return;
    }
    SendPing(address);
  }

  void RecvPingTimeOut(const std::string &address) {
    MSI_LOG(INFO) << "Recv Ping Time Out from " << address;
    // add exit handle
    watcher_map_.erase(address);
  }
  void PingAsync(const std::string &address) {
    proto::PingRequest request;
    proto::PingReply reply;
    request.set_address(address);
    grpc::ClientContext context;
    const int32_t TIME_OUT = 100;
    std::chrono::system_clock::time_point deadline =
      std::chrono::system_clock::now() + std::chrono::microseconds(TIME_OUT);
    context.set_deadline(deadline);
    (void)watchee_map_[address].stub_->Ping(&context, request, &reply);
    MSI_LOG(INFO) << "Finish send ping";
  }

  void PongAsync(const std::string &address) {
    proto::PongRequest request;
    proto::PongReply reply;
    request.set_address(address);
    grpc::ClientContext context;
    const int32_t TIME_OUT = 100;
    std::chrono::system_clock::time_point deadline =
      std::chrono::system_clock::now() + std::chrono::microseconds(TIME_OUT);
    context.set_deadline(deadline);
    (void)watcher_map_[address].stub_->Pong(&context, request, &reply);
    MSI_LOG(INFO) << "Finish send pong";
  }

 private:
  struct WatcheeContext {
    uint64_t timeouts_ = 0;
    std::shared_ptr<Timer> timer_ = nullptr;
    std::shared_ptr<typename SendStub::Stub> stub_ = nullptr;
  };
  struct WatcherContext {
    uint64_t timeouts_ = 0;
    std::shared_ptr<Timer> timer_ = nullptr;
    std::shared_ptr<typename RecvStub::Stub> stub_ = nullptr;
  };
  std::string host_address_;
  uint64_t max_ping_times_ = 10;
  uint64_t max_time_out_ = 10000;  // 10s
  std::unordered_map<std::string, WatcheeContext> watchee_map_;
  std::unordered_map<std::string, WatcherContext> watcher_map_;
};
}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_HEART_BEAT_H
