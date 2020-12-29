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

static const char UrlKeyModel[] = "model";
static const char UrlKeyVersion[] = "version";
static const char UrlSplit[] = "/";
static const char UrlKeyEnd[] = ":";

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
  if (key == UrlKeyEnd) {
    if ((start_pos = url_.find(UrlKeyEnd)) != std::string::npos) {
      return url_.substr(start_pos + 1, url_.size());
    }
  }

  int key_size = key.size() + 1;
  std::string::size_type end_pos(0);
  if ((start_pos = url.find(key)) != std::string::npos) {
    end_pos = std::min(url.find(UrlSplit, start_pos + key_size), url.find(UrlKeyEnd, start_pos + key_size));
    if (end_pos != std::string::npos) {
      return url.substr(start_pos + key_size, end_pos - start_pos - key_size);
    }
  }
  return "";
}

Status DecomposeEvRequest::GetPostMessageToJson() {
  Status status(SUCCESS);
  std::string message;
  size_t input_size = evbuffer_get_length(event_request_->input_buffer);
  if (input_size == 0) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "http message invalid");
    return status;
  } else if (input_size > max_msg_size_) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "http message is bigger than " + std::to_string(max_msg_size_));
    return status;
  } else {
    message.resize(input_size);
    auto src_data = evbuffer_pullup(event_request_->input_buffer, -1);
    if (src_data == nullptr) {
      ERROR_INFER_STATUS(status, FAILED, "get http message failed.");
      return status;
    }
    if (memcpy_s(message.data(), input_size, src_data, input_size) != EOK) {
      ERROR_INFER_STATUS(status, FAILED, "copy http message failed.");
      return status;
    }
  }
  MSI_TIME_STAMP_START(ParseJson)
  try {
    request_message_ = nlohmann::json::parse(message);
  } catch (nlohmann::json::exception &e) {
    std::string json_exception = e.what();
    std::string error_message = "Illegal JSON format." + json_exception;
    ERROR_INFER_STATUS(status, INVALID_INPUTS, error_message);
    return status;
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
      ERROR_INFER_STATUS(status, INVALID_INPUTS, "http message only support POST right now");
      return status;
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
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "evhttp url is empty.");
    return status;
  }
  MSI_LOG_INFO << "url_: " << url_;

  model_name_ = UrlQuery(url_, UrlKeyModel);
  if (model_name_.empty()) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "please check url, the keyword:[model] must contain.");
    return status;
  }
  MSI_LOG_INFO << "model_name_: " << model_name_;
  if (url_.find(UrlKeyVersion) != std::string::npos) {
    auto version_str = UrlQuery(url_, UrlKeyVersion);
    try {
      version_ = std::stol(version_str);
    } catch (const std::invalid_argument &) {
      ERROR_INFER_STATUS(status, INVALID_INPUTS, "please check url, the keyword:[version] must contain.");
      return status;
    } catch (const std::out_of_range &) {
      ERROR_INFER_STATUS(status, INVALID_INPUTS, "please check url, the keyword:[version] out of range.");
      return status;
    }
    MSI_LOG_INFO << "version_: " << version_;
  }

  service_method_ = UrlQuery(url_, UrlKeyEnd);
  if (service_method_.empty()) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "please check url, the keyword:[service method] must contain.");
    return status;
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
  Status status(SUCCESS);
  replay_buffer_ = evbuffer_new();
  if (replay_buffer_ == nullptr) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "creat restful replay buffer fail");
    return status;
  }
  return status;
}

Status RestfulRequest::RestfulReplay(const std::string &replay) {
  Status status(SUCCESS);
  if (replay_buffer_ == nullptr) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "replay_buffer_ is nullptr");
    return status;
  }
  if (decompose_event_request_ == nullptr) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "replay_buffer_ is nullptr");
    return status;
  }
  if (decompose_event_request_->event_request_ == nullptr) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "replay_buffer_ is nullptr");
    return status;
  }
  evbuffer_add(replay_buffer_, replay.data(), replay.size());
  evhttp_send_reply(decompose_event_request_->event_request_, HTTP_OK, "Client", replay_buffer_);
  return status;
}

Status RestfulRequest::ErrorMessage(Status status) {
  Status error_status(SUCCESS);
  nlohmann::json error_json = {{"error_message", status.StatusMessage()}};
  std::string out_error_str = error_json.dump();
  if ((error_status = RestfulReplay(out_error_str)) != SUCCESS) {
    return error_status;
  }
  return error_status;
}

}  // namespace serving
}  // namespace mindspore
