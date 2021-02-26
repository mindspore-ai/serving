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
  ~Timer() {
    is_stoped_.store(true);
    cv_.notify_all();
    if (thread_.joinable()) {
      try {
        thread_.join();
      } catch (const std::system_error &) {
      } catch (...) {
      }
    }
  }

  void StartTimer(int64_t millisecond, TimerCallback callback) {
    auto timer_run = [this, millisecond, callback]() {
      while (!is_stoped_.load()) {
        std::unique_lock<std::mutex> lk(cv_m_);
        if (cv_.wait_for(lk, std::chrono::milliseconds(millisecond)) == std::cv_status::timeout) {
          callback();
        }
      }
    };
    thread_ = std::thread(timer_run);
  }
  void StopTimer() {
    is_stoped_.store(true);
    cv_.notify_all();
  }

 private:
  std::mutex cv_m_;
  std::thread thread_;
  std::condition_variable cv_;
  std::atomic<bool> is_stoped_ = false;
};

template <class SendStub, class RecvStub>
class MS_API Watcher {
 public:
  explicit Watcher(const std::string host_address) { host_address_ = host_address; }
  ~Watcher() {
    if (ping_running_) {
      ping_cq_.Shutdown();
      if (ping_thread_.joinable()) {
        try {
          ping_thread_.join();
        } catch (const std::system_error &) {
        } catch (...) {
        }
      }
    }
    ping_running_ = false;
    if (pong_running_) {
      pong_cq_.Shutdown();
      if (pong_thread_.joinable()) {
        try {
          pong_thread_.join();
        } catch (const std::system_error &) {
        } catch (...) {
        }
      }
    }
    pong_running_ = false;
  }
  void StartWatch(const std::string &address) {
    if (ping_running_ == false) {
      ping_thread_ = std::thread(&Watcher::AsyncPingRpc, this);
      ping_running_ = true;
    }
    auto it = watchee_map_.find(address);
    if (it != watchee_map_.end()) {
      MSI_LOG(INFO) << "watchee exist: " << address;
      watchee_map_[address].timer_ = std::make_shared<Timer>();
    } else {
      WatcheeContext context;
      auto channel = GrpcServer::CreateChannel(address);
      context.stub_ = SendStub::NewStub(channel);
      context.timer_ = std::make_shared<Timer>();
      watchee_map_.insert(make_pair(address, context));
    }
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
    if (pong_running_ == false) {
      pong_thread_ = std::thread(&Watcher::AsyncPongRpc, this);
      pong_running_ = true;
    }
    std::unique_lock<std::mutex> lock{m_lock_};
    // recv message
    if (watcher_map_.count(address)) {
      watcher_map_[address].timer_->StopTimer();
      watcher_map_[address].timer_ = std::make_shared<Timer>();
    } else {
      WatcherContext context;
      auto channel = GrpcServer::CreateChannel(address);
      context.stub_ = RecvStub::NewStub(channel);
      context.timer_ = std::make_shared<Timer>();
      watcher_map_.insert(make_pair(address, context));
      MSI_LOG(INFO) << "Begin to send pong to " << address;
    }
    // send async message
    PongAsync(address);
    // add timer
    watcher_map_[address].timer_->StartTimer(max_time_out_, std::bind(&Watcher::RecvPingTimeOut, this, address));
  }

  void RecvPong(const std::string &address) {
    std::unique_lock<std::mutex> lock{m_lock_};
    // recv message
    if (watchee_map_.count(address)) {
      watchee_map_[address].timeouts_ = 0;
    } else {
      MSI_LOG(INFO) << "Recv Pong after timeout or stop";
    }
  }

  void RecvPongTimeOut(const std::string &address) {
    std::unique_lock<std::mutex> lock{m_lock_};
    if (watchee_map_[address].timeouts_ >= max_ping_times_) {
      // add exit handle
      MSI_LOG(ERROR) << "Recv Pong Time Out from " << address << ", host address is " << host_address_;
      watchee_map_[address].timer_->StopTimer();
      // need erase map
      return;
    }
    SendPing(address);
  }

  void RecvPingTimeOut(const std::string &address) {
    MSI_LOG(ERROR) << "Recv Ping Time Out from " << address << ", host address is " << host_address_;
    // add exit handle
    watcher_map_[address].timer_->StopTimer();
    // need erase map
  }
  void PingAsync(const std::string &address) {
    proto::PingRequest request;
    request.set_address(host_address_);
    AsyncPingCall *call = new AsyncPingCall;
    call->response_reader = watchee_map_[address].stub_->PrepareAsyncPing(&call->context, request, &ping_cq_);
    call->response_reader->StartCall();
    call->response_reader->Finish(&call->reply, &call->status, call);
  }

  void PongAsync(const std::string &address) {
    proto::PongRequest request;
    request.set_address(host_address_);
    AsyncPongCall *call = new AsyncPongCall;
    call->response_reader = watcher_map_[address].stub_->PrepareAsyncPong(&call->context, request, &pong_cq_);
    call->response_reader->StartCall();
    call->response_reader->Finish(&call->reply, &call->status, call);
  }
  void AsyncPingRpc() {
    void *got_tag;
    bool ok = false;
    while (ping_cq_.Next(&got_tag, &ok)) {
      AsyncPingCall *call = static_cast<AsyncPingCall *>(got_tag);
      if (!call->status.ok()) {
        MSI_LOG_DEBUG << "RPC failed: " << call->status.error_code() << ", " << call->status.error_message();
      }
      delete call;
    }
  }
  void AsyncPongRpc() {
    void *got_tag;
    bool ok = false;
    while (pong_cq_.Next(&got_tag, &ok)) {
      AsyncPongCall *call = static_cast<AsyncPongCall *>(got_tag);
      if (!call->status.ok()) {
        MSI_LOG_DEBUG << "RPC failed: " << call->status.error_code() << ", " << call->status.error_message();
      }
      delete call;
    }
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
  struct AsyncPingCall {
    grpc::ClientContext context;
    grpc::Status status;
    proto::PingReply reply;
    std::shared_ptr<grpc::ClientAsyncResponseReader<proto::PingReply>> response_reader;
  };
  struct AsyncPongCall {
    grpc::ClientContext context;
    grpc::Status status;
    proto::PongReply reply;
    std::shared_ptr<grpc::ClientAsyncResponseReader<proto::PongReply>> response_reader;
  };
  std::string host_address_;
  uint64_t max_ping_times_ = 20;
  uint64_t max_time_out_ = 20000;  // 20s
  std::unordered_map<std::string, WatcheeContext> watchee_map_;
  std::unordered_map<std::string, WatcherContext> watcher_map_;
  std::mutex m_lock_;
  grpc::CompletionQueue ping_cq_;
  std::thread ping_thread_;
  bool ping_running_ = false;
  grpc::CompletionQueue pong_cq_;
  std::thread pong_thread_;
  bool pong_running_ = false;
};
}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_HEART_BEAT_H
