from agent.agent_multi_post_method import *
from multiprocessing import Queue

from config.serving_config import AgentConfig, ModelName


if __name__ == "__main__":
    startup_queue = Queue(1024)
    startup_agents(AgentConfig.ctx_setting,
                   AgentConfig.inc_setting,
                   AgentConfig.post_model_setting,
                   len(AgentConfig.AgentPorts),
                   AgentConfig.prefill_model,
                   AgentConfig.decode_model,
                   AgentConfig.argmax_model,
                   AgentConfig.topk_model,
                   startup_queue)

    started_agents = 0
    while True:
        value = startup_queue.get()
        print("agent : %f started" % value)
        started_agents = started_agents + 1
        if started_agents >= len(AgentConfig.AgentPorts):
            print("all agents started")
            break

    # server_app_post.init_server_app()
    # server_app_post.warmup_model(ModelName)
    # server_app_post.run_server_app()
