import os
import time
import psutil
import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help='YAML config files')
    args = parser.parse_args()
    # start agent
    print("----starting agents----")
    p_agent = subprocess.Popen([f'python examples/start_agent.py --config {args.config}> agent.log 2>&1 &'],
                               shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    agents_finished = False
    while not agents_finished:
        # check if agent pid is alive
        if not psutil.pid_exists(p_agent.pid) or (psutil.pid_exists(p_agent.pid) not in psutil.pids()):
            raise RuntimeError('there occurs some error when starting agent, check the agent.log')
        # 监控agent端口是否启动, 暂时删除，310 上会有部分不影响功能的报错，导致启动进程停止
        # err_out = os.popen("grep \"ERROR\" ./agent.log")
        # for line in err_out.read().splitlines():
        #     # raise RuntimeError(line)
        #     print("error line:", line)
        output = os.popen("grep \"all agents\" ./agent.log")
        res = output.read()
        for line in res.splitlines():
            if 'all agents started' in line:
                agents_finished = True
                break
        time.sleep(1)
    print("----agents are ready----")

    print("----starting server----")
    p_server = subprocess.Popen([f'python examples/server_app_post.py --config {args.config}> server.log 2>&1 &'],
                                shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    server_finished = False
    while not server_finished:
        # check if server pid is alive
        if not psutil.pid_exists(p_server.pid) or (psutil.pid_exists(p_server.pid) not in psutil.pids()):
            raise RuntimeError('there occurs some error when starting serving, check the server.log')
        # 监控agent端口是否启动
        # err_out = os.popen("grep \"ERROR\" ./server.log")
        # for line in err_out.read().splitlines():
        #     raise RuntimeError(line)
        output = os.popen("grep \"Uvicorn running on\" ./server.log")
        res = output.read()
        for line in res.splitlines():
            if line is not None:
                server_finished = True
                break
        time.sleep(1)
    print("----server is ready----")
