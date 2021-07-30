import configparser

import multiprocessing as mp

from tqdm import tqdm
from utils.mysqlTools import get_video_info
from videoProcess.videoProcess import VideoProcess
"""
启动videoProcess线程需要做的事情“
    1、读取mysql数据库的movie的数据，获取video_path *
    2、控制线程的启动、资源释放 *
    3、交换数据控制，对进入的不同video分配recognize进程 *****
"""

def run(video_id ,video_path, config_file):

    vd_proc = VideoProcess(video_id, video_path, config_file)

    vd_proc.run()


def videoProcess(video_id, video_dict, conf):

    result_list = []

    def log_result(result):
        result_list.append(result)

    max_num_process = int(conf.get('video', 'max_num_process'))

    pool = mp.Pool(processes=max_num_process)
    for item in tqdm(video_id):
        print(item)
        handler = pool.apply_async(run, args=(item, video_dict[item], config_file,), callback=log_result)
    pool.close()
    pool.join()
    print(result_list)



if __name__ == '__main__':
    config_file = 'config/config.cfg'
    conf = configparser.ConfigParser()
    conf.read(config_file, encoding='utf-8')
    video_dict = get_video_info(conf)

    video_id = list(video_dict.keys())[0:20]

    videoProcess(video_id, video_dict, conf)


