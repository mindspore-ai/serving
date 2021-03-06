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

#ifndef MINDSPORE_SERVING_MASTER_HTTP_HANDLE_H
#define MINDSPORE_SERVING_MASTER_HTTP_HANDLE_H

#include <string>
#include <memory>
#include "common/serving_common.h"
#include "master/restful/restful_request.h"

using nlohmann::json;
namespace mindspore {
namespace serving {
Status HandleRestfulRequest(const std::shared_ptr<RestfulRequest> &restful_request);

size_t Base64Encode(const uint8_t *input, size_t length, uint8_t *output);
size_t Base64Decode(const uint8_t *target, size_t target_length, uint8_t *origin);
size_t GetB64TargetSize(size_t origin_len);
size_t GetB64OriginSize(size_t target_len, size_t tail_size);
size_t GetTailEqualSize(const std::string &str);

}  // namespace serving
}  // namespace mindspore
#endif  // MINDSPORE_SERVING_MASTER_HTTP_HANDLE_H
