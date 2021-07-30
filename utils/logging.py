import os
import cv2
import pymysql
import logging
import traceback

from .mysqlTools import MySqlHold

import numpy as np

# 统一将log文件捆绑在了log文件夹下面
# 未作分类
LOG_ROOT = './log'
if not os.path.exists(LOG_ROOT):
    os.mkdir(LOG_ROOT)


def get_log_hold(running_file_name: str, display: bool = True):
    """
    Log holder generator
    :param running_file_name: running file name
    :param display: weather display log in the foreground
    :return: None
    """
    os.makedirs(os.path.join(LOG_ROOT, running_file_name), exist_ok=True)
    log_file = os.path.join(LOG_ROOT, running_file_name, "log.log")
    if not os.path.isfile(log_file):
        with open(log_file, 'w'):
            pass
    logging.getLogger().setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(running_file_name)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if display:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def save_record(video_id: str, save_dir: str, mode: str, name: str, time_: str):
    """
    Generate log file for this project,all record will write in to a file.
    :param video_id: video id
    :param save_dir: log file save dir
    :param mode: recognition mode
    :param name: recognition name
    :param time_: timestamp
    :return: None
    """
    # logger.info('save record!')
    save_file = os.path.join(save_dir, video_id + '.log')
    record = f"{video_id} {mode} {name} {time_}"
    if os.path.isfile(save_file):
        with open(save_file, 'a', encoding='utf-8') as f:
            f.writelines(record + '\n')
    else:
        with open(save_file, 'w', encoding='utf-8') as f:
            f.writelines(record + '\n')


def report_finished(mysql: MySqlHold,
                    logger: logging.Logger,
                    video_id: str,
                    abs_save_dir: str,
                    record_table: str):
    """
    Only report finished status.
    :param mysql: mysql holder
    :param logger: logger holder
    :param video_id: video id
    :param abs_save_dir: I don't know this argument.
    :param record_table: mysql record table
    :return: None
    """
    insert_command = f"insert into {record_table} " \
                     f"(video_id, timestamp, person_name, save_file, mode) " \
                     f"values(\'{video_id}\', " \
                     f"\'finished\', " \
                     f"\'finished\'," \
                     f" \'{video_id}\'," \
                     f" \'face\' )"
    save_record(video_id=video_id,
                save_dir=abs_save_dir,
                mode='video',
                name='finished',
                time_='finished')
    logger.info("{} detect finished!".format(video_id))
    # print("[INFO]: {} detect finished!".format(video_id))
    try:
        mysql.execute_command(insert_command)
    except pymysql.err.IntegrityError:
        logger.warning("Record: {} is exists!".format(insert_command))


def report_result(record_table: str,
                  mode: str,
                  save_dir: str,
                  video_id: str,
                  timestamp: list,
                  name_list: list,
                  batch_image: np.ndarray,
                  mysql: MySqlHold,
                  logger: logging.Logger,
                  num_faces_record: list = None,
                  code_dir: str = None):
    """
    report result to mysql.
    :param record_table: mysql record table name
    :param mode: recognition mode
    :param save_dir: image save dir
    :param video_id: video id
    :param timestamp: record timestamp
    :param name_list: recognition result name list
    :param batch_image: batch images
    :param mysql: mysql holder
    :param logger: logger holder
    :param num_faces_record: number of recognition face
    :param code_dir: code path in the HOST.
    :return: None
    """
    # logger.info("Report_result!")
    # print("[INFO]: Report_result!")
    record_table = record_table
    # 构造宿主机路径， /code_dir/output_dir/video_id
    host_save_dir = None
    if code_dir is not None:
        host_save_dir = os.path.join(code_dir, save_dir.split('/')[-2], save_dir.split('/')[-1])


    def report_union(mode, mysql, time_index: int, file_index: int, count_):
        name = name_list[count_] if isinstance(count_, int) else count_
        time_ = timestamp[time_index].replace(':', '-')
        image_file = os.path.join(save_dir,
                                  name + '_' + time_ + '_' + str(file_index) + '_' + mode + '.jpg')
        # 宿主机文件路径
        host_image_file = None
        if host_save_dir is not None:
            host_image_file = os.path.join(host_save_dir,
                                      name + '_' + time_ + '_' + str(file_index) + '_' + mode + '.jpg')
        try:
            if batch_image is not None and not os.path.exists(image_file):
                cv2.imwrite(image_file, batch_image[time_index])
        except UnicodeEncodeError:
            logger.error(f"{save_dir}, {name}, {time_}, {mode}")
            logger.error(f"{num_faces_record}, {type(num_faces_record)}")
            logger.error(f"{str(file_index)}")
            logger.error(traceback.format_exc())
        insert_command = f"insert into {record_table} " \
                         f"(video_id, timestamp, person_name, save_file, mode) " \
                         f"values(\'{video_id}\', " \
                         f"\'{timestamp[time_index]}\', " \
                         f"\'{name}\'," \
                         f" \'{host_image_file if host_image_file is not None else image_file}\'," \
                         f" \'{mode}\' )"
        save_record(video_id=video_id, save_dir=save_dir, mode=mode, name=name, time_=time_)
        logger.info("Insert one record: {}, {}, {}".format(video_id, timestamp[time_index], name))
        # print("[INFO]: Insert one record: {}, {}, {}".format(video_id, timestamp[time_index], name))
        try:
            mysql.execute_command(insert_command)
        except pymysql.err.IntegrityError:
            logger.warning("Record: {} is exists!".format(insert_command))
            # print("[WARNING]: Record: {} is exists!".format(insert_command))

    if num_faces_record is not None:
        count = 0
        for i, item in enumerate(num_faces_record):
            for idx in range(item):
                if name_list[count] != 'Unknown' and name_list[count] is not None:
                    report_union(mode=mode, mysql=mysql, time_index=i, file_index=idx, count_=count)
                # 是应该放在结尾，这样保证了下一次能扫描到
                count += 1
    else:
        for i, item in enumerate(name_list):
            if item is not None:
                report_union(mode=mode, mysql=mysql, time_index=i, file_index=i, count_=item)
