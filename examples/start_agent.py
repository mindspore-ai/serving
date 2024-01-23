import argparse
import sys
from multiprocessing import Queue
from mindspore_serving.agent.agent_multi_post_method import startup_agents
from mindspore_serving.config.config import ServingConfig, check_valid_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help='YAML config files')
    args = parser.parse_args()
    startup_queue = Queue(1024)
    config = ServingConfig(args.config)
    if not check_valid_config(config):
        sys.exit(1)
    print("load yaml sucess!")

    startup_agents(config, startup_queue)

    started_agents = 0
    while True:
        value = startup_queue.get()
        print("agent : %f started" % value)
        started_agents = started_agents + 1
        if started_agents >= len(config.serving_config.agent_ports):
            print("all agents started")
            break
