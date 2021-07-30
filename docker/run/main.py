print("""
        __    __  _________     __ __      __           ____               
       / /   /  |/  / ____/    / //_/___ _/ /__  ____  / __/___  _________ 
      / /   / /|_/ / /  ______/ ,< / __ `/ / _ \/ __ \/ /_/ __ \/ ___/ __ \\
     / /___/ /  / / /__/_____/ /| / /_/ / /  __/ / / / __/ /_/ / /  / / / /
    /_____/_/  /_/\____/    /_/ |_\__,_/_/\___/_/ /_/_/  \____/_/  /_/ /_/ 
                                                                           
""")

import os
import time
import signal
import traceback
import configparser

import multiprocessing as mp

from utils.logging import get_log_hold
from utils.common import init_redis

FLAG = False

# 选择导入模块
try:
    from frProcess import FR as MyProcess

    logger = get_log_hold("face" + os.path.basename(__file__).split('.')[0])
    logger.info("Loading face recognition model.")
    MODE = "face"
    FLAG = True
except ModuleNotFoundError as e:
    pass
try:
    from brProcess import BR as MyProcess

    logger = get_log_hold("banner" + os.path.basename(__file__).split('.')[0])
    logger.info("Loading banner recognition model.")
    MODE = "banner"
    FLAG = True
except ModuleNotFoundError as e:
    pass
try:
    from lrProcess import LR as MyProcess

    logger = get_log_hold("logo" + os.path.basename(__file__).split('.')[0])
    logger.info("Loading logo recognition model.")
    MODE = "logo"
    FLAG = True
except ModuleNotFoundError as e:
    pass
try:
    from nrProcess import NR as MyProcess

    logger = get_log_hold("nsfw" + os.path.basename(__file__).split('.')[0])
    logger.info("Loading nsfw recognition model.")
    MODE = "nsfw"
    FLAG = True
except ModuleNotFoundError as e:
    pass
try:
    from orProcess import OR as MyProcess

    logger = get_log_hold("ocr" + os.path.basename(__file__).split('.')[0])
    logger.info("Loading ocr recognition model.")
    MODE = "ocr"
    FLAG = True
except ModuleNotFoundError as e:
    pass

if not FLAG:
    raise ModuleNotFoundError("There is no recognition process module!")


# 从给定的gpu_id中挑选一个未占满进程（满进程指config文件中的单卡最大识别进程数量）的gpu给识别程序
def get_gpu_id(devices: list, device_list: list, num_process: int):

    for item in devices:
        if device_list.count(item) < num_process:
            return item
    return None


def main():
    config_file = './config.cfg'
    # config_files = [config_file, './config.cfg']
    conf = configparser.ConfigParser()
    conf.read(config_file, encoding='utf-8')

    num_process = int(conf.get('general', 'max_re_process'))
    max_repeat_times = int(conf.get('recognition', 'max_repeat_times'))
    # 用来记录所使用的显卡资源
    devices = [int(item) for item in conf.get('recognition', 'devices').split(',')]
    mp.set_start_method('spawn')
    # 用来记录每个进程对应的gpu_id
    device_list = [-1] * (num_process * len(devices))
    # 进程记录
    process_list = [None] * (num_process * len(devices))
    # process_list = [[]] * len(devices)
    # 记录失败进程创建记录次数，目的是为了防止由于显卡资源不足重复失败的创建进程
    re_time = 0
    # 这两个flag都是用来记录一次日志文件的
    flag: bool = True
    flag_redis: bool = True
    redis_dict = {"host": conf.get('redis', 'host'),
                  "port": int(conf.get('redis', 'port')),
                  "passwd": conf.get('redis', 'passwd'),
                  "start": conf.get('redis', 'start'),
                  'maximun_memory': int(conf.get('redis', 'maximun_memory')) * (1024 ** 3),
                  "db": 0}
    redis = init_redis(logger=logger, redis_dict=redis_dict)
    # 消除重启错误
    redis.pop(mode='info', name=redis_dict['start'])
    while True:
        # 这里可能会产生危险
        # 更新配置文件
        if redis is not None and (redis.get('start') is None or redis.get('start') == '0'):
            for item in process_list:
                if item is not None:
                    item.join()
                    # item.terminate()
                    # 这里的做法是强制使用系统指令对进程杀死
                    pid = item.pid
                    os.kill(pid, signal.SIGSTOP)
                    logger.info("Force kill process, PID -> {}.".format(pid))
            if flag_redis:
                logger.info("Waiting to initialize the config file!")
                flag_redis = False
            continue
        # if redis is not None:
        #     redis.close()
        #     del redis
        # redis = None

        # 是否开启进程
        if conf.get('re_status', MODE) not in ['on', 'On', 'ON', 'oN', 'up', 'Up', 'UP', 'uP']:
            # 避免重复写日志
            if flag:
                logger.info(f"Mode: {MODE} not up!")
                flag = False
            continue
        # flag = True
        if len(process_list) > (num_process * len(devices)):
            logger.error("The program runs out of control, please contact the administrator!")
            raise RuntimeError("[ERROR]: The program runs out of control, please contact the administrator!")
        # 这里已经避免了item为None的情况
        if process_list.count(None) == 0:
            for i, item in enumerate(process_list):
                if item.is_alive():
                    continue
                else:
                    try:
                        item.join()
                        # item.terminate()
                        # 这里的做法是强制使用系统指令对进程杀死
                        pid = item.pid
                        os.kill(pid, signal.SIGSTOP)
                        logger.info("Killed process, PID -> {}.".format(pid))
                        # print("[INFO]: kill process PID ->", pid)
                    except Exception as e:
                        logger.error(traceback.format_exc())
                        # print("[ERROR]:", traceback.format_exc())
                    finally:
                        # 防止资源泄露
                        del item
                        # 无论进程是否删除成功，将失败进程创建记录销毁
                        re_time = 0
                        # 写在finally里的是值要检测到有未存活的进程则无论如何需要丢弃
                        # 销毁gpu id的list记录日志
                        device_list[i] = -1
                        process_list[i] = None
        # 如果超过重复创建次数则跳过创建
        elif re_time < max_repeat_times:
            for item in process_list:
                if item is not None:
                    continue
                index = process_list.index(item)

                gpu_id = get_gpu_id(devices, device_list, num_process)
                # 如果gpu_id为None则表示process_list是满的
                if gpu_id is None:
                    continue
                try:
                    re_process = MyProcess(config_file, gpu_id)
                    re_process.start()
                    logger.info("New process, PID -> {}".format(re_process.pid))
                    # print("[INFO]: new process PID ->", re_process.pid)
                    device_list[index] = gpu_id
                    process_list[index] = re_process
                    # re_time = 0
                except RuntimeError:
                    error = traceback.format_exc()
                    if "CUDA" in error:
                        # 如果由于CUDA问题导致创建失败则失败进程创建记录加1
                        re_time += 1
                        logger.warning("CUDA error, repeat times {}, restart a new one!".format(re_time))
                        logger.error(error)
                        # print("[WARNING]: CUDA error, repeat times {}, restart a new one!".format(re_time))
                    else:
                        logger.error(error)
                        # print("[ERROR]:", error)
                except:
                    # print("[ERROR]: Unknown error, please contact the administrator!")
                    logger.error("Unknown error, please contact the administrator!")
                    logger.error(traceback.format_exc())
                    # print("[ERROR]:", traceback.format_exc())
                    break
        # 这个else只会在重复创建任务达到峰值以后防止程序死掉（死循环，不做任何操作）采用的reset方法
        else:
            time.sleep(600)
            re_time = 0
        # 停止等待一会，防止无用的cpu消耗
        time.sleep(2)


if __name__ == '__main__':
    main()
