import os
import time

if __name__ == "__main__":
    # start agent
    print("----starting agents----")
    _ = os.popen("python start_agent.py > agent.log 2>&1 &")
    agents_finished = False
    while not agents_finished:
        # 监控agent端口是否启动
        output = os.popen("grep \"all agents\" ./agent.log")
        res = output.read()
        for line in res.splitlines():
            print(line)
            if line == 'all agents started':
                agents_finished = True
                break
        time.sleep(1)
    print("----agents are ready----")
    print("----starting serving----")

    _ = os.popen("python client/server_app_post.py > server.log 2>&1 &")
    server_finished = False
    while not server_finished:
        # 监控agent端口是否启动
        output = os.popen("grep \"server port is\" ./server.log")
        res = output.read()
        for line in res.splitlines():
            print(line)
            if line is not None:
                server_finished = True
                break
        time.sleep(1)
    print("----serving is ready----")
