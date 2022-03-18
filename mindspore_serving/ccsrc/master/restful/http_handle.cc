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

#include "master/restful/http_handle.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <vector>

#include "master/restful/http_process.h"
#include "master/server.h"

namespace mindspore {
namespace serving {
static std::vector<unsigned char> encode_table = {
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
  'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
  's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'};
static std::vector<unsigned char> decode_table = {
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 62,
  255, 255, 255, 63,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  255, 255, 255, 255, 255, 255, 255, 0,
  1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,
  23,  24,  25,  255, 255, 255, 255, 255, 255, 26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  255, 255, 255, 255, 255};

size_t Base64Encode(const uint8_t *input, size_t length, uint8_t *output) {
  if (length == 0) {
    return 0;
  }

  size_t i, j;
  for (i = 0, j = 0; i + 3 <= length; i += 3) {
    output[j++] = encode_table[input[i] >> 2];
    output[j++] = encode_table[((input[i] << 4) & 0x30) | (input[i + 1] >> 4)];
    output[j++] = encode_table[((input[i + 1] << 2) & 0x3c) | (input[i + 2] >> 6)];
    output[j++] = encode_table[input[i + 2] & 0x3f];
  }

  if (i < length) {
    uint32_t left_num = length - i;
    if (left_num == 1) {
      output[j++] = encode_table[input[i] >> 2];
      output[j++] = encode_table[(input[i] << 4) & 0x30];
      output[j++] = '=';
      output[j++] = '=';
    } else {
      output[j++] = encode_table[input[i] >> 2];
      output[j++] = encode_table[((input[i] << 4) & 0x30) | (input[i + 1] >> 4)];
      output[j++] = encode_table[(input[i + 1] << 2) & 0x3c];
      output[j++] = '=';
    }
  }
  return j;
}

size_t Base64Decode(const uint8_t *target, size_t target_length, uint8_t *origin) {
  if (target_length == 0 || target_length % 4 != 0) {
    return 0;
  }
  size_t i, j = 0;
  uint8_t value[4];
  for (i = 0; i < target_length; i += 4) {
    for (size_t k = 0; k < 4; k++) {
      value[k] = decode_table[target[i + k]];
    }

    // value[2], value[3]:may be '='
    if (value[0] >= 64 || value[1] >= 64) {
      MSI_LOG_EXCEPTION << "Decode value is not more than max value 64";
    }

    origin[j++] = (value[0] << 2) | (value[1] >> 4);

    if (value[2] >= 64) {
      break;
    } else if (value[3] >= 64) {
      origin[j++] = (value[1] << 4) | (value[2] >> 2);
      break;
    } else {
      origin[j++] = (value[1] << 4) | (value[2] >> 2);
      origin[j++] = (value[2] << 6) | value[3];
    }
  }
  return j;
}

size_t GetB64TargetSize(size_t origin_len) {
  size_t target_size = 0;
  if (origin_len % 3 == 0) {
    target_size = (origin_len / 3) * 4;
  } else {
    target_size = (origin_len / 3 + 1) * 4;
  }
  return target_size;
}

size_t GetB64OriginSize(size_t target_len, size_t tail_size) {
  size_t origin_length = 0;
  if (target_len == 0 || target_len % 4 != 0) {
    return origin_length;
  }
  origin_length = 3 * (target_len / 4) - tail_size;
  return origin_length;
}

size_t GetTailEqualSize(const std::string &str) {
  size_t length = str.size();
  if (length % 4 != 0) {
    return UINT32_MAX;
  }
  size_t count = 0;
  if (length >= 1 && str[length - 1] == '=') {
    count++;
  }
  if (length >= 2 && str[length - 2] == '=') {
    count++;
  }
  return count;
}
}  // namespace serving
}  // namespace mindspore
