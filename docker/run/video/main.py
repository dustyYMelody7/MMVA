print("""
        __    __  _________     __ __      __           ____               
       / /   /  |/  / ____/    / //_/___ _/ /__  ____  / __/___  _________ 
      / /   / /|_/ / /  ______/ ,< / __ `/ / _ \/ __ \/ /_/ __ \/ ___/ __ \\
     / /___/ /  / / /__/_____/ /| / /_/ / /  __/ / / / __/ /_/ / /  / / / /
    /_____/_/  /_/\____/    /_/ |_\__,_/_/\___/_/ /_/_/  \____/_/  /_/ /_/ 

""")
import os
import configparser

import multiprocessing as mp

from flask import Flask, request, json, jsonify
from utils.mysqlTools import get_one_item
from utils.logging import get_log_hold
from videoProcess import check_file
from videoProcess import VideoProcess_docker as VideoProcess
"""
启动videoProcess线程需要做的事情“
    1、读取mysql数据库的movie的数据，获取video_path *
    2、控制线程的启动、资源释放 *
    3、交换数据控制，对进入的不同video分配recognize进程 *****
"""

VIDEO_DICT = None
CONF = None
logger = get_log_hold("video" + os.path.basename(__file__).split('.')[0])


def run(video_id, video_path, config_file):

    vd_proc = VideoProcess(video_id, video_path, config_file)
    vd_proc.run()


def video_process(video_id):
    config_file = 'config.cfg'
    conf = configparser.ConfigParser()
    conf.read(config_file, encoding='utf-8')

    result_list = []

    def log_result(result):
        result_list.append(result)
    output_dir = conf.get('general', 'output_dir')
    max_num_process = int(conf.get('general', 'max_video_process'))

    config_files = "config.cfg"

    pool = mp.Pool(processes=max_num_process)
    for item in video_id:
        log_file = os.path.join(output_dir, item, item + '.log')
        if os.path.isfile(log_file):
            if check_file(log_file):
                logger.warning(f"Video: {item} has been detected!")
                # print(f"[WARNING]: Video: {item} has been detected!")
                continue
        # print(item)
        logger.info(f"Video: {item} push!")
        handler = pool.apply_async(
            run, args=(item, get_one_item(conf=conf, table=conf.get('mysql', 'video_table'),
                                          column_name='video_id', value=item)[-1][-1], config_files,),
            callback=log_result)
    pool.close()
    pool.join()
    # print(result_list)

# 更新配置文件
def write_info(info: dict):
    module = ['ocr', 'face', 'nsfw', 'logo', 'banner', 'audio']
    config_file = './config.cfg'
    conf = configparser.ConfigParser()
    conf.read(config_file, encoding='utf-8')
    open_model = []
    for item in info:
        if item == 'id':
            continue
        open_model.append(item)
        redis_name = item + '_redis'
        key = 'db'
        value = info[item]
        conf.set(redis_name, key, value)
    for item in module:
        conf.set('re_status', item, 'off')
    for item in open_model:
        conf.set('re_status', item, 'on')
    with open(config_file, 'w') as f:
        conf.write(f)
    logger.info("update config file succeed!")


app = Flask(__name__)


@app.route('/start', methods=['POST'])
def post():
    request_data = json.loads(request.data)
    write_info(request_data)
    video_process(request_data['id'])
    logger.info("Post Finished!")
    # print("[INFO]: Post Finished!")
    return jsonify({'status': 'finished!'})


if __name__ == '__main__':
    #global CONF, VIDEO_DICT
    port = 9933
    app.run(host="0.0.0.0", port=port, debug=False)
    logger.info("Flask have started! Listening on 0.0.0.0:%d !" % port)
    # print("Flask have started! Listening on 0.0.0.0:%d !" % port)


