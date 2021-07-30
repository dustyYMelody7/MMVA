from .videoProcess import VideoProcess
from .videoProcess_docker import VideoProcess as VideoProcess_docker


def check_file(log_file: str):

    with open(log_file) as f:
        data = f.readlines()
    for item in data:
        if "finished" in item:
            return True
    return False
