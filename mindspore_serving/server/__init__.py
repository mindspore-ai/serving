# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
MindSpore Serving is a lightweight and high-performance service module that helps MindSpore developers efficiently
deploy online inference services in the production environment.

MindSpore Serving server API, which can be used to start servables, gRPC and RESTful server. A servable corresponds to
the service provided by a model. The client sends inference tasks and receives inference results through gRPC and
RESTful server.
"""

from .master import start_grpc_server, start_restful_server, stop, SSLConfig
from ._server import start_servables, ServableStartConfig
from . import register
from . import distributed

__all__ = []
__all__.extend([
    "start_grpc_server",
    "start_restful_server",
    "stop",
    "start_servables",
    'ServableStartConfig',
    "SSLConfig"
])

__all__.extend(register.__all__)
__all__.extend(distributed.__all__)
