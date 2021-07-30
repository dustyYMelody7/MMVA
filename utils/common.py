import os
import time
import json
import redis
import logging
import pymysql
import traceback
import configparser

import functools as fun

from typing import Union
from .mysqlTools import MySqlHold
from .redisTools import RedisHold
from .logging import get_log_hold, report_result, report_finished

from multiprocessing import Process


# 独立出来是为了短线重连
def init_mysql(logger: logging.Logger, mysql_dict: dict) -> MySqlHold:
    """
    Generate mysql handler
    :param logger: log handler
    :param mysql_dict: all information for mysql
    :return: MySqlHold
    """
    mysql = MySqlHold(host=mysql_dict['host'],
                      user=mysql_dict['user'],
                      password=mysql_dict['passwd'],
                      database=mysql_dict['db'],
                      port=mysql_dict['port'])
    logger.info("Mysql init!")
    return mysql


def init_redis(logger: logging.Logger, redis_dict: dict) -> RedisHold:
    """
    Generate redis handler
    :param logger: log handler
    :param redis_dict: all information for redis
    :return: RedisHold
    """
    redis = RedisHold(host=redis_dict['host'],
                      port=redis_dict['port'],
                      db_name=redis_dict['db'],
                      passwd=redis_dict['passwd'],
                      maximun_memory=redis_dict['maximun_memory'],
                      logger=logger)
    logger.info("Redis init!")
    return redis


def configurable(function):
    """

    :param function:
    :return:
    """

    assert function.__name__ == '__run_init', "@configurable should only be used for __run_init!"

    @fun.wraps(function)
    def wrapped(self):
        for config_file in self.config_files:
            if "common" in config_file:
                conf = configparser.ConfigParser()
                conf.read(config_file, encoding='utf-8')
                self._mysql_dict = {"port": int(conf.get('mysql', 'port')),
                                    "user": conf.get('mysql', 'user'),
                                    "host": conf.get('mysql', 'host'),
                                    "db": conf.get('mysql', 'db'),
                                    "passwd": conf.get('mysql', 'passwd'),
                                    "video_table": conf.get('mysql', 'video_table'),
                                    "record_table": conf.get('mysql', 'record_table'),
                                    "face_feature_table": conf.get('mysql', 'face_feature_table')}
                self._redis_dict = {"host": conf.get('redis', 'host'),
                                    "port": int(conf.get('redis', 'port')),
                                    "passwd": conf.get('redis', 'passwd'),
                                    "info_name": conf.get('redis', 'info_name')}
                self._output_dir = conf.get('general', 'output_dir')
                self._code_dir = conf.get('general', 'code_dir')
            else:
                conf = configparser.ConfigParser()
                conf.read(config_file, encoding='utf-8')
                self._redis_dict.update({'db': conf.get('redis', 'db')})
                self._redis = init_redis(self.logger, self._redis_dict)
                self._mysql = init_mysql(self.logger, self._mysql_dict)

    return wrapped


def get_data(redis_read: RedisHold,
             redis_info_name: str,
             logger: logging.Logger,
             redis_write: Union[RedisHold, None] = None,
             save_dir: str = './output',
             sleep_time: int = 5):

    video_info = redis_read.pop(mode='info', name=redis_info_name)
    if video_info is None:
        return None
    video_id, timestamp, batch_size, image_width, image_height, mode = video_info
    abs_save_dir = os.path.join(save_dir, video_id)

    # 数据流水线切换
    def write_info_to_redis(batch_image_=None):
        ######
        # 为什么这个地方一定得加这个判断，如果把判断压缩到while里就不能实现finished信号的传输。
        ######
        if batch_image_ is None:
            redis_write.rpush(name=redis_info_name, data=json.dumps({'id': video_id,
                                                                     'timestamp': timestamp,
                                                                     'mode': mode,
                                                                     'batch_size': batch_size,
                                                                     'image_width': image_width,
                                                                     'image_height': image_height}))
            return
        while not (redis_write.rpush(name=redis_info_name, data=json.dumps({'id': video_id,
                                                                            'timestamp': timestamp,
                                                                            'mode': mode,
                                                                            'batch_size': batch_size,
                                                                            'image_width': image_width,
                                                                            'image_height': image_height})) and
                   redis_write.rpush(name=video_id, data=batch_image_.tobytes())):
            time.sleep(sleep_time)
            logger.warning("Re-push data.")
    try:
        if not os.path.exists(abs_save_dir):
            os.makedirs(abs_save_dir)
    except FileExistsError as e:
        logger.warning("Dir {} existed!".format(abs_save_dir))
    if mode == 'audio':
        return batch_size, video_id, timestamp, abs_save_dir
    elif mode == 'video':
        batch_image = redis_read.pop(mode='image',
                                     name=video_id,
                                     batch_size=batch_size,
                                     image_width=image_width,
                                     image_height=image_height)
        # 插入到下一条数据库
        if redis_write is not None:
            write_info_to_redis(batch_image)
            # logger.info("transform all value to next process.")
        return batch_image, video_id, timestamp, abs_save_dir
    elif mode == 'finished':
        # 这里是为了仅在最后一个程序中报到finished
        if redis_write is not None:
            write_info_to_redis()
            # logger.info("record finished status to next process.")
            return None
        return 'finished', video_id, 'finished', abs_save_dir
    logger.error("The 'mode' parameter is not accepted!")
    return None


class ReBase(Process):

    def __init__(self, config_files: str, gpu_id: int = 0):
        super(ReBase, self).__init__()
        # assert len(config_files) == 2, "Config file must include common.cfg,config.cfg"
        self.config_files = config_files
        self.logger = None
        self._gpu_id = gpu_id
        self._model = self._mysql = self._redis_read = self._redis_write = self._output_dir \
            = self._mysql_dict = self._redis_dict_write = self._redis_read_dict = None
        self.mode = None
        self._file = os.path.dirname(__file__)

    # @configurable
    def _init_private_redis(self):
        conf = configparser.ConfigParser()
        conf.read(self.config_files, encoding='utf-8')
        redis_name = self.mode + "_redis"
        redis_dict = {"host": conf.get('redis', 'host'),
                      "port": int(conf.get('redis', 'port')),
                      "passwd": conf.get('redis', 'passwd'),
                      "info_name": conf.get('redis', 'info_name'),
                      'maximun_memory': int(conf.get('redis', 'maximun_memory')) * (1024 ** 3)}
        db = [int(item.strip()) for item in conf.get(redis_name, 'db').split(',')]
        self._sleep_time = int(conf.get(redis_name, 'sleep_time'))
        # assert len(db) == 2, "config.cfg must have two redis database!"
        if len(redis_dict) == 0:
            raise ValueError("'redis_dict' is empty!")
        # 读数据的database
        redis_dict.update({'db': db[0]})
        self._redis_read_dict = redis_dict.copy()
        self._redis_read = init_redis(self.logger, self._redis_read_dict)
        # 因为最后一个流水线不需要写数据，所以这里加判断最后一个进程不做处理
        if len(db) > 1:
            # 写数据的database
            redis_dict.update({'db': db[1]})
            self._redis_write_dict = redis_dict.copy()
            self._redis_write = init_redis(self.logger, self._redis_write_dict)
        else:
            self._redis_write_dict = None
            self._redis_write_dict = None

    def _run_init(self, redis=True):
        assert self.logger is not None, "self.logger is not init!"
        redis_dict = {}
        conf = db = None
        # 这里有个隐形bug，如果config_files = [config.cfg, common.cfg]会报错
        conf = configparser.ConfigParser()
        conf.read(self.config_files, encoding='utf-8')
        self._mysql_dict = {"port": int(conf.get('mysql', 'port')),
                            "user": conf.get('mysql', 'user'),
                            "host": conf.get('mysql', 'host'),
                            "db": conf.get('mysql', 'db'),
                            "passwd": conf.get('mysql', 'passwd'),
                            "video_table": conf.get('mysql', 'video_table'),
                            "record_table": conf.get('mysql', 'record_table'),
                            "face_feature_table": conf.get('mysql', 'face_feature_table')}
        self._output_dir = conf.get('general', 'output_dir')
        self._code_dir = conf.get('general', 'code_dir')
        if redis:
            self._init_private_redis()
        self._mysql = init_mysql(self.logger, self._mysql_dict)
        # 确定绝对路径
        self._output_dir = os.path.join(self._file, self._output_dir)
        try:
            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)
        except FileExistsError as e:
            self.logger.warning("Dir {} exist".format(self._output_dir))
        del db, conf

    def run(self):
        self._run_init()
        assert self._model is not None, "You need init self._model before start process!"
        assert self.mode is not None, "You need init self.mode before start process!"
        try:
            while True:
                # start = None
                try:
                    name_list = None
                    num_faces_record = None
                    data = get_data(redis_read=self._redis_read,
                                    redis_info_name=self._redis_read_dict['info_name'],
                                    logger=self.logger,
                                    redis_write=self._redis_write,
                                    save_dir=self._output_dir,
                                    sleep_time=self._sleep_time)
                    if data is None:
                        continue
                    batch_image, video_id, timestamp, abs_save_dir = data
                    if batch_image is None:
                        continue
                    # start = time.time()
                    if isinstance(batch_image, str):
                        try:
                            report_finished(mysql=self._mysql,
                                            logger=self.logger,
                                            video_id=video_id,
                                            abs_save_dir=abs_save_dir,
                                            record_table=self._mysql_dict['record_table'])
                        # 这里主要解决mysql重传机制
                        except pymysql.err.OperationalError:
                            self._mysql.close()
                            del self._mysql
                            error = traceback.format_exc()
                            if "Lock wait timeout exceeded; try restarting transaction" in error:
                                self.logger.warning(
                                    "The mysql connection was interrupted unexpectedly and is reconnecting!")
                            else:
                                self.logger.error(error)
                                break
                            # print("[WARNING]: The mysql connection was interrupted unexpectedly and is reconnecting.")
                            self._mysql = init_mysql(self.logger, self._mysql_dict)
                            report_finished(mysql=self._mysql,
                                            logger=self.logger,
                                            video_id=video_id,
                                            abs_save_dir=abs_save_dir,
                                            record_table=self._mysql_dict['record_table'])
                    else:
                        # self.logger.info("Get data!")
                        name_list = self._model.detect(batch_image=batch_image)
                        num_faces_record = None
                        # 只有face_module才会触发下述条件
                        if self.mode == 'face' and isinstance(name_list, tuple):
                            name_list, num_faces_record = name_list
                        if name_list is None:
                            continue
                        try:
                            report_result(record_table=self._mysql_dict['record_table'],
                                          mysql=self._mysql,
                                          logger=self.logger,
                                          save_dir=abs_save_dir,
                                          video_id=video_id,
                                          batch_image=batch_image,
                                          timestamp=timestamp,
                                          mode=self.mode,
                                          num_faces_record=num_faces_record,
                                          name_list=name_list,
                                          code_dir=self._code_dir)
                        except pymysql.err.OperationalError:
                            self._mysql.close()
                            del self._mysql
                            error = traceback.format_exc()
                            if "Lock wait timeout exceeded; try restarting transaction" in error:
                                self.logger.warning(
                                    "The mysql connection was interrupted unexpectedly and is reconnecting!")
                            else:
                                self.logger.error(error)
                                break
                            # print("[WARNING]: The mysql connection was interrupted unexpectedly and is reconnecting.")
                            self._mysql = init_mysql(self.logger, self._mysql_dict)
                            report_result(record_table=self._mysql_dict['record_table'],
                                          mysql=self._mysql,
                                          logger=self.logger,
                                          save_dir=abs_save_dir,
                                          video_id=video_id,
                                          batch_image=batch_image,
                                          timestamp=timestamp,
                                          mode=self.mode,
                                          num_faces_record=num_faces_record,
                                          name_list=name_list,
                                          code_dir=self._code_dir)
                    del data, batch_image, video_id, timestamp, abs_save_dir
                except redis.ConnectionError or redis.TimeoutError:
                    self._redis_read.close()
                    self.logger.warning("Maybe redis connection was interrupted unexpectedly and is reconnecting!")
                    if self._redis_write_dict is not None:
                        self._redis_write.close()
                    del self._redis_read, self._redis_write
                    self._redis_read = init_redis(self.logger, self._redis_read_dict)
                    if self._redis_write_dict is not None:
                        self._redis_write = init_redis(self.logger, self._redis_write_dict)
                # finally:
                    # if start is not None:
                    #     self.logger.info("using: {}".format(time.time() - start))
        # except RuntimeError:
        #     error = traceback.format_exc()
        #     if "CUDA" in error:
        #         self.logger.warning(" CUDA error, restart a new one!")
        #         # print("[WARNING]: CUDA error, restart a new one!")
        #     else:
        #         self.logger.error(error)
        #         # print("[ERROR]:", error)
        except:
            self.logger.error(traceback.format_exc())
            # print("[ERROR]:", traceback.format_exc())
        finally:
            self.terminate()

    def __release(self):

        try:
            if self._mysql is not None:
                self._mysql.close()
                self.logger.info("Mysql connect closed!")
                # print("[INFO]: Mysql connect closed!")
            if self._redis_read is not None:
                self._redis_read.close()
                self.logger.info("Read redis connect closed!")
            if self._redis_write is not None:
                self._redis_write.close()
                self.logger.info("Write redis connect closed!")
                # print("[INFO]: Redis connect closed!")
        except:
            self.logger.error(traceback.format_exc())

    def terminate(self) -> None:
        self.__release()
        super(ReBase, self).terminate()
