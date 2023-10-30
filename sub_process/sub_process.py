import logging
import time
import os
import signal
import threading
import psutil


def _send_exit_signal_to_children(subprocess_list):
    """Send exit signal to all child processes, and terminate all child processes when they are still alive
    in some seconds later"""

    def wait_exit(wait_seconds, msg):
        for i in range(wait_seconds):
            all_exit = True
            for process in subprocess_list:
                if process.is_alive():
                    logging.warning(f"There are still child processes that have not exited and {msg} in "
                                    f"{wait_seconds - i} seconds.")
                    time.sleep(1)
                    all_exit = False
                    break
            if all_exit:
                logging.info(f"All Child process exited")
                return True
        return False

    if wait_exit(3, "SIGINT will be sent"):
        return
    # Send signal SIGINT
    for index, process in enumerate(subprocess_list):
        if process.is_alive():
            logging.warning(f"Send signal SIGINT to {index}")
            try:
                child_process = psutil.Process(process.pid)
                children_of_child = child_process.children(recursive=True)
                for item in children_of_child:
                    os.kill(item.pid, signal.SIGINT)
            # pylint: disable=broad-except
            except Exception as e:
                logging.warning(f"Get exception when send signal SIGINT to children of child {index}, exception: {e}")
            os.kill(process.pid, signal.SIGINT)

    if wait_exit(10, "will be forcibly killed"):
        return

    for index, process in enumerate(subprocess_list):
        if process.is_alive():
            logging.warning(f"Kill Child process {index}")
            try:
                child_process = psutil.Process(process.pid)
                children_of_child = child_process.children(recursive=True)
                for item in children_of_child:
                    os.kill(item.pid, signal.SIGKILL)
            # pylint: disable=broad-except
            except Exception as e:
                logging.warning(f"Get exception when send signal SIGKILL to children of child {index}, exception: {e}")
            os.kill(process.pid, signal.SIGKILL)


def listen_agents_after_startup(subprocess_list):
    def wait_child_exit():
        # TODO 捕捉退出信号
        while True:
            for index, sub in enumerate(subprocess_list):
                if not sub.is_alive():
                    logging.warning(f"Child {index}, pid={sub.pid} has exited")
                    # TODO 通知distributed worker agents退出
                    return
            time.sleep(0.1)

    def listening_thread_fun():
        wait_child_exit()
        # kill all children
        _send_exit_signal_to_children(subprocess_list)

    thread = threading.Thread(target=listening_thread_fun)
    thread.start()