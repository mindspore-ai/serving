from agent.agent_multi_post_method import *
from worker.worker_to_agent import *
import mindspore_lite as ms
import numpy as np
import logging

from config.serving_config import AgentConfig


def test_protocol():
    inputs = np.random.randint(0, 100, size=[4, 10])
    print(inputs)
    proto_request = create_proto_request(inputs, 0)
    request, inc_flag = parse_proto_request(proto_request)
    print(request, inc_flag)

    proto_reply = create_proto_reply(inputs)
    result = parse_proto_reply(proto_reply)
    print(result.tensors)


if __name__ == "__main__":
    startup_agents(AgentConfig.ctx_setting,
                   AgentConfig.inc_setting,
                   AgentConfig.post_model_setting,
                   AgentConfig.npu_nums,
                   AgentConfig.prefill_model,
                   AgentConfig.decode_model,
                   AgentConfig.argmax_model,
                   AgentConfig.topk_model,
                   AgentConfig.AgentPorts[0])