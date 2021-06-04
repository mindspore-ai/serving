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

#include "master/restful/restful_request.h"
#include <event2/buffer.h>
#include <event2/http.h>
#include <evhttp.h>
#include <algorithm>
#include <utility>

namespace {
const char kUrlKeyModel[] = "model";
const char kUrlKeyVersion[] = "version";
const char kUrlSplit[] = "/";
const char kUrlKeyEnd[] = ":";
}  // namespace

namespace mindspore {
namespace serving {

DecomposeEvRequest::DecomposeEvRequest(struct evhttp_request *request, int max_msg_size)
    : event_request_(request), max_msg_size_(max_msg_size) {}

DecomposeEvRequest::~DecomposeEvRequest() {
  if (event_request_ && evhttp_request_is_owned(event_request_)) {
    evhttp_request_free(event_request_);
  }
}

std::string DecomposeEvRequest::UrlQuery(const std::string &url, const std::string &key) {
  std::string::size_type start_pos(0);
  if (key == kUrlKeyEnd) {
    if ((start_pos = url_.find(kUrlKeyEnd)) != std::string::npos) {
      return url_.substr(start_pos + 1, url_.size());
    }
  }

  int key_size = key.size() + 1;
  std::string::size_type end_pos(0);
  if ((start_pos = url.find(key)) != std::string::npos) {
    end_pos = std::min(url.find(kUrlSplit, start_pos + key_size), url.find(kUrlKeyEnd, start_pos + key_size));
    if (end_pos == std::string::npos) {
      return url.substr(start_pos + key_size);
    }
    return url.substr(start_pos + key_size, end_pos - start_pos - key_size);
  }
  return "";
}

Status DecomposeEvRequest::GetPostMessageToJson() {
  Status status(SUCCESS);
  std::string message;
  size_t input_size = evbuffer_get_length(event_request_->input_buffer);
  if (input_size == 0) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "http message invalid";
  } else if (input_size > max_msg_size_) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "http message is bigger than " << max_msg_size_;
  } else {
    message.resize(input_size);
    auto src_data = evbuffer_pullup(event_request_->input_buffer, -1);
    if (src_data == nullptr) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "get http message failed.";
    }
    if (memcpy_s(message.data(), input_size, src_data, input_size) != EOK) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "copy http message failed.";
    }
  }
  MSI_TIME_STAMP_START(ParseJson)
  try {
    request_message_ = nlohmann::json::parse(message);
  } catch (nlohmann::json::exception &e) {
    std::string json_exception = e.what();
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Illegal JSON format." + json_exception;
  }
  MSI_TIME_STAMP_END(ParseJson)

  return status;
}

Status DecomposeEvRequest::CheckRequestMethodValid() {
  Status status(SUCCESS);
  switch (evhttp_request_get_command(event_request_)) {
    case EVHTTP_REQ_POST:
      request_method_ = "POST";
      return status;
    default:
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "http message only support POST right now";
  }
}

Status DecomposeEvRequest::Decompose() {
  Status status(SUCCESS);
  status = CheckRequestMethodValid();
  if (status != SUCCESS) {
    return status;
  }

  status = GetPostMessageToJson();
  if (status != SUCCESS) {
    return status;
  }

  // eg: /model/resnet/version/1:predict
  url_ = evhttp_request_get_uri(event_request_);
  if (url_.empty()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "evhttp url is empty.";
  }
  MSI_LOG_INFO << "url_: " << url_;

  model_name_ = UrlQuery(url_, kUrlKeyModel);
  if (model_name_.empty()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "please check url, the keyword:[model] must contain.";
  }
  MSI_LOG_INFO << "model_name_: " << model_name_;
  if (url_.find(kUrlKeyVersion) != std::string::npos) {
    auto version_str = UrlQuery(url_, kUrlKeyVersion);
    try {
      auto version = std::stol(version_str);
      if (version < 0 || version >= UINT32_MAX) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
               << "please check url, version number range failed, request version number " << version_str;
      }
      version_ = static_cast<uint32_t>(version);
    } catch (const std::invalid_argument &) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
             << "please check url, the keyword:[version] value invalid, request version number " << version_str;
    } catch (const std::out_of_range &) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
             << "please check url, version number range failed, request version number " << version_str;
    }
    MSI_LOG_INFO << "version_: " << version_;
  }

  service_method_ = UrlQuery(url_, kUrlKeyEnd);
  if (service_method_.empty()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "please check url, the keyword:[service method] must contain.";
  }
  MSI_LOG_INFO << "service_method_: " << service_method_;
  return status;
}

RestfulRequest::RestfulRequest(std::shared_ptr<DecomposeEvRequest> request)
    : decompose_event_request_(std::move(request)) {}

RestfulRequest::~RestfulRequest() {
  if (replay_buffer_ != nullptr) {
    evbuffer_free(replay_buffer_);
  }
}

Status RestfulRequest::RestfulReplayBufferInit() {
  replay_buffer_ = evbuffer_new();
  if (replay_buffer_ == nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "create restful replay buffer fail";
  }
  return SUCCESS;
}

Status RestfulRequest::RestfulReplay(const std::string &replay) {
  if (replay_buffer_ == nullptr) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "replay_buffer_ is nullptr";
  }
  if (decompose_event_request_ == nullptr) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "decompose_event_request_ is nullptr";
  }
  if (decompose_event_request_->event_request_ == nullptr) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "decompose_event_request_->event_request_ is nullptr";
  }
  evbuffer_add(replay_buffer_, replay.data(), replay.size());
  evhttp_send_reply(decompose_event_request_->event_request_, HTTP_OK, "Client", replay_buffer_);
  return SUCCESS;
}

void RestfulRequest::ErrorMessage(Status status) {
  nlohmann::json error_json = {{"error_msg", status.StatusMessage()}};
  std::string out_error_str = error_json.dump();
  (void)RestfulReplay(out_error_str);
}

}  // namespace serving
}  // namespace mindspore
