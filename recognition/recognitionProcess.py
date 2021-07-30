import os
import time
import signal
import pymysql
import traceback
import configparser

import numpy as np

from multiprocessing import Process

from utils.mysqlTools import MySqlHold
from utils.utils import get_feature
from utils.redisTools import RedisHold
from utils.logging import get_log_hold, save_record, report_result
from .ocr import OCRModule
from .face import FaceModule
# from .general_det import GeneralDet
# from .logo import LogoModule
from .nsfw import NSFWModule
# from .audio import AudioModule

ON_STATUS_LIST = ['on', 'ON', 'On', 'oN']
logger = get_log_hold(os.path.basename(__file__).split('.')[0])


# 记录log文件
# def save_record(video_id: str, save_dir: str, mode: str, name: str, time_: str):
#     logger.info('save record!')
#     save_file = os.path.join(save_dir, video_id + '.log')
#     record = f"{video_id} {mode} {name} {time_}"
#     if os.path.isfile(save_file):
#         with open(save_file, 'a', encoding='utf-8') as f:
#             f.writelines(record)
#     else:
#         with open(save_file, 'w', encoding='utf-8') as f:
#             f.writelines(record)


class RecognitionProcess(Process):

    def __init__(self, config_file=os.path.join(os.path.dirname(__file__), '../config/config.cfg'), gpu_id: int = 0):

        super(RecognitionProcess, self).__init__()
        if not os.path.isfile(config_file):
            raise ValueError("[ERROR]: Argument of 'config_file': {} doesn't exists.".format(config_file))

        self.config_file = config_file
        self.gpu_id = gpu_id

    # 独立出来是为了短线重连
    def __init_mysql(self, conf: configparser.ConfigParser):
        self.__mysql = MySqlHold(host=conf.get('mysql', 'host'),
                                 user=conf.get('mysql', 'user'),
                                 password=conf.get('mysql', 'passwd'),
                                 database=conf.get('mysql', 'db'),
                                 port=int(conf.get('mysql', 'port')))

    def __init_redis(self, conf: configparser.ConfigParser):
        self.__redis = RedisHold(host=conf.get('redis', 'host'),
                                 port=int(conf.get('redis', 'port')),
                                 db_name=int(conf.get('redis', 'db')),
                                 passwd=conf.get('redis', 'passwd'),
                                 logger=logger)

    def __runing_init(self):
        logger.info('Running init!')
        # print("[INFO]: Running init!")
        conf = configparser.ConfigParser()
        conf.read(self.config_file)

        # module status
        face_re_status = conf.get('re_status', 'face')
        text_re_status = conf.get('re_status', 'ocr')
        # logo_re_status = conf.get('re_status', 'logo')
        # banner_re_status = conf.get('re_status', 'banner')
        nsfw_re_status = conf.get('re_status', 'nsfw')
        # self.__audio_re_status = conf.get('re_status', 'audio')

        # mysql init
        try:
            self.__init_mysql(conf)
            self.record_table = conf.get('mysql', 'record_table')
        except ConnectionRefusedError as e:
            logger.error("Connect error, your mysql.service doesn't active!"
                         "\nPlease make sure your mysql.service active!")
            # print("[ERROR]: Connect error, your mysql.service doesn't active!"
            #       "\nPlease make sure your mysql.service active!")
            exit(1)

        # redis init
        try:
            self.__init_redis(conf)
            self.__redis_info_name = conf.get('redis', 'info_name')
        except ConnectionRefusedError as e:
            logger.error("Connect error, your redis-server.service doesn't active!"
                         "\nPlease make sure your redis-server.service active!")
            # print("[ERROR]: Connect error, your redis-server.service doesn't active!"
            #       "\nPlease make sure your redis-server.service active!")
            exit(1)

        self.__method = {}
        # module init
        if face_re_status in ON_STATUS_LIST:
            face_module = FaceModule(conf, self.__mysql, gpu_id=self.gpu_id)
            self.__method.update({'face': face_module})

        _, name_list = get_feature(self.__mysql, conf.get('mysql', 'face_feature_table'))
        if len(name_list) <= 0:
            raise RuntimeError("[ERROR]: Please check your mysql table: {}, "
                               "there is nothing!".format('face_feature_table'))
        if text_re_status in ON_STATUS_LIST:
            text_module = OCRModule(conf=conf, name_list=name_list, gpu_id=self.gpu_id)
            self.__method.update({'ocr': text_module})

        # if banner_re_status in ON_STATUS_LIST:
        #     banner_module = GeneralDet(configpath=conf.get('banner', 'config_file'), gpu_id=self.gpu_id)
        #     self.__method.update({'banner': banner_module})
        #
        # if logo_re_status in ON_STATUS_LIST:
        #     logo_module = GeneralDet(configpath=conf.get('logo', 'config_file'), gpu_id=self.gpu_id)
        #     self.__method.update({'logo': logo_module})

        if nsfw_re_status in ON_STATUS_LIST:
            nsfw_module = NSFWModule(conf=conf, gpu_id=self.gpu_id)
            self.__method.update({'nsfw': nsfw_module})

        # if self.__audio_re_status in ON_STATUS_LIST:
        #     self.__audio_module = AudioModule(conf=conf, name_list=name_list)
        # output path
        self.output_dir = conf.get('general', 'output_dir')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self.conf = conf

    # 识别进程代工厂，用于减少代码量
    def __recognize_factor(
            self, video_id: str,
            save_dir: str,
            timestamp: list,
            batch_image: np.ndarray,
            method: str = 'face'):
        """
        factor function.
        :param video_id: video id.
        :param save_dir: image save dir.
        :param timestamp: timestamp record.
        :param batch_image: batch images.
        :param method: recognition method.
        :return: None
        """

        # 如果不在里面就代表未开启模块！
        if method not in list(self.__method.keys()):
            return None

        if batch_image is None:
            return None
        model = self.__method[method]
        name_list = model.detect(batch_image=batch_image)
        num_faces_record = None
        # 只有face_module才会触发下述条件
        if isinstance(name_list, tuple):
            name_list, num_faces_record = name_list
        if name_list is None:
            return
        report_result(record_table=self.record_table,
                      mysql=self.__mysql,
                      logger=logger,
                      save_dir=save_dir,
                      video_id=video_id,
                      batch_image=batch_image,
                      timestamp=timestamp,
                      mode=method,
                      num_faces_record=num_faces_record,
                      name_list=name_list)

    # def __face_recognize(self, batch_image: np.ndarray, video_id: str, save_dir: str, timestamp: list):
    #     if self.__face_re_status not in ON_STATUS_LIST:
    #         print("[WARNING]: Face recognition module is not turned on!")
    #         return None
    #     print("[INFO]: Face detect!")
    #     if batch_image is None:
    #         return None
    #     mode = 'face'
    #     name_list, num_faces_record = self.__face_module.detect(batch_image=batch_image)
    #     if name_list is None:
    #         return None
    #     self.__report_result(save_dir=save_dir,
    #                          video_id=video_id,
    #                          batch_image=batch_image,
    #                          timestamp=timestamp,
    #                          mode=mode,
    #                          num_faces_record=num_faces_record,
    #                          name_list=name_list)
    #
    # def __text_recognize(self, batch_image: np.ndarray, video_id: str, save_dir: str, timestamp: list):
    #     if self.__text_re_status not in ON_STATUS_LIST:
    #         print("[WARNING]: Word recognition module is not turned on!")
    #         return None
    #     print("[INFO]: Word detect!")
    #     if batch_image is None:
    #         return None
    #     mode = 'ocr'
    #     name_list = self.__text_module.detect(batch_image=batch_image)
    #     if name_list is None:
    #         return None
    #     self.__report_result(save_dir=save_dir,
    #                          video_id=video_id,
    #                          batch_image=batch_image,
    #                          timestamp=timestamp,
    #                          mode=mode,
    #                          num_faces_record=None,
    #                          name_list=name_list)
    #
    # def __logo_recognize(self, batch_image: np.ndarray, video_id: str, save_dir: str, timestamp: list):
    #     if self.__logo_re_status not in ON_STATUS_LIST:
    #         print("[WARNING]: Logo recognition module is not turned on!")
    #         return None
    #     if batch_image is None:
    #         return None
    #     name_list = self.__logo_module.detect(batch_image=batch_image)
    #     if name_list is None:
    #         return None
    #     mode = 'logo'
    #     self.__report_result(save_dir=save_dir,
    #                          video_id=video_id,
    #                          batch_image=batch_image,
    #                          timestamp=timestamp,
    #                          mode=mode,
    #                          num_faces_record=None,
    #                          name_list=name_list)
    #
    # def __banner_recognize(self, batch_image: np.ndarray, video_id: str, save_dir: str, timestamp: list):
    #     if self.__banner_re_status not in ON_STATUS_LIST:
    #         print("[WARNING]: Banner recognition module is not turned on!")
    #         return None
    #     if batch_image is None:
    #         return None
    #     mode = 'banner'
    #     name_list = self.__banner_module.detect(batch_image=batch_image)
    #     if name_list is None:
    #         return None
    #     self.__report_result(save_dir=save_dir,
    #                          video_id=video_id,
    #                          batch_image=batch_image,
    #                          timestamp=timestamp,
    #                          mode=mode,
    #                          num_faces_record=None,
    #                          name_list=name_list)
    #
    # def __nsfw_recognize(self, batch_image: np.ndarray, video_id: str, save_dir: str, timestamp: list):
    #     if self.__nsfw_re_status not in ON_STATUS_LIST:
    #         print("[WARNING]: NSFW recognition module is not turned on!")
    #         return None
    #     if batch_image is None:
    #         return None
    #     mode = 'nsfw'
    #     name_list = self.__nsfw_module.detect(batch_image=batch_image)
    #     if name_list is None:
    #         return None
    #     self.__report_result(save_dir=save_dir,
    #                          video_id=video_id,
    #                          batch_image=batch_image,
    #                          timestamp=timestamp,
    #                          mode=mode,
    #                          num_faces_record=None,
    #                          name_list=name_list)
    #
    # def __audio_recognize(self, audio_file: list) -> bool:
    #     if self.__audio_re_status not in ON_STATUS_LIST:
    #         logger.warning("Audio recognition module is not turned on!")
    #         # print("[WARNING]: Audio recognition module is not turned on!")
    #         return False
    #     if len(audio_file) < 1:
    #         logger.warning("No source!")
    #         # print("[WARNING]: No source!")
    #         return False
    #     if self.__audio_module.is_alive():
    #         logger.warning("Audio recognition process is running! waiting to recognize: {}".format(len(audio_file)))
    #         # print("[WARNING]: Audio recognition process is running! waiting to recognize: {}".format(len(audio_file)))
    #         return False
    #     else:
    #         logger.info("Audio recognition process is finished.")
    #         self.__release_audio()
    #     self.__audio_module.set_wav_path(audio_file[-1])
    #     self.__audio_module.start()
    #     return True

    # def __release_audio(self):
    #     try:
    #         self.__audio_module.join()
    #         self.__audio_module.terminate()
    #         pid = self.__audio_module.pid
    #         os.kill(pid, signal.SIGSTOP)
    #     except:
    #         logger.error(traceback.format_exc())
    #         # print(traceback.format_exc())

    def run(self):
        self.__runing_init()
        logger.info("Finished init!")
        # print("[INFO]: Finished init!")
        try:
            audio_list = []
            logger.info("Waiting to detect!")
            # print("[INFO]: Waiting to detect!")
            while True:
                video_info = self.__redis.pop(mode='info', name=self.__redis_info_name)
                # print("[INFO]:", video_info)
                if video_info is None:
                    continue
                # 后期优化方案：
                #############
                # 1、将从redis读数据的过程独立出来成为进程
                # 2、读数据进程read_process与主识别进程数据交换采用queue的方式进行
                #    主进程只用扫描queue决定是否继续执行
                #############
                start_ = time.time()
                # 当音频识别时batch_size值用作音频地址
                video_id, timestamp, batch_size, image_width, image_height, mode = video_info
                abs_save_dir = os.path.join(os.path.dirname(__file__), '..', self.output_dir, video_id)
                try:
                    if not os.path.isdir(abs_save_dir):
                        os.mkdir(abs_save_dir)
                except FileExistsError as e:
                    logger.warning("Dir {} exist".format(abs_save_dir))
                    # print("[WARNING]: Dir {} exist".format(abs_save_dir))

                # video_detect
                try:
                    # print("[INFO]: Read using:", time.time() - start)
                    # start = time.time()
                    # self.__recognize_factor(batch_image=batch_image,
                    #                         video_id=video_id,
                    #                         save_dir=abs_save_dir,
                    #                         timestamp=timestamp,
                    #                         method='face')
                    # # self.__face_recognize(batch_image=batch_image,
                    # #                       video_id=video_id,
                    # #                       save_dir=abs_save_dir,
                    # #                       timestamp=timestamp)
                    # print("[INFO]: Face using:", time.time() - start)
                    # start = time.time()
                    # self.__recognize_factor(batch_image=batch_image,
                    #                         video_id=video_id,
                    #                         save_dir=abs_save_dir,
                    #                         timestamp=timestamp,
                    #                         method='ocr')
                    # # self.__text_recognize(batch_image=batch_image,
                    # #                       video_id=video_id,
                    # #                       save_dir=abs_save_dir,
                    # #                       timestamp=timestamp)
                    # print("[INFO]: Word using:", time.time() - start)
                    # start = time.time()
                    # self.__recognize_factor(batch_image=batch_image,
                    #                         video_id=video_id,
                    #                         save_dir=abs_save_dir,
                    #                         timestamp=timestamp,
                    #                         method='banner')
                    # print("[INFO]: Banner using:", time.time() - start)
                    # start = time.time()
                    # self.__recognize_factor(batch_image=batch_image,
                    #                         video_id=video_id,
                    #                         save_dir=abs_save_dir,
                    #                         timestamp=timestamp,
                    #                         method='logo')
                    # print("[INFO]: Logo using:", time.time() - start)
                    # start = time.time()
                    # self.__recognize_factor(batch_image=batch_image,
                    #                         video_id=video_id,
                    #                         save_dir=abs_save_dir,
                    #                         timestamp=timestamp,
                    #                         method='nsfw')
                    # print("[INFO]: Logo using:", time.time() - start)
                    # # self.__logo_recognize(batch_image=batch_image,
                    # #                       video_id=video_id,
                    # #                       save_dir=abs_save_dir,
                    # #                       timestamp=timestamp)
                    # # self.__banner_recognize(batch_image=batch_image,
                    # #                         video_id=video_id,
                    # #                         save_dir=abs_save_dir,
                    # #                         timestamp=timestamp)
                    if mode == 'video':
                        start = time.time()
                        batch_image = self.__redis.pop(mode='image',
                                                          name=video_id,
                                                          batch_size=batch_size,
                                                          image_width=image_width,
                                                          image_height=image_height)
                        logger.info("Reading used: {}".format(time.time() - start))
                        # print("[INFO]: Reading used:", time.time() - start)
                        if batch_image is None:
                            continue
                        # 循环检测方法
                        for item in list(self.__method.keys()):
                            self.__recognize_factor(batch_image=batch_image,
                                                    video_id=video_id,
                                                    save_dir=abs_save_dir,
                                                    timestamp=timestamp,
                                                    method=item)
                    # elif mode == 'audio':
                    #     audio_list.append(batch_size)
                    #     if self.__audio_recognize(audio_file=audio_list):
                    #         audio_list.pop(-1)
                    elif mode == 'finished':
                        insert_command = f"insert into {self.record_table} " \
                                         f"(video_id, timestamp, person_name, save_file, mode) " \
                                         f"values(\'{video_id}\', " \
                                         f"\'finished\', " \
                                         f"\'finished\'," \
                                         f" \'finished\'," \
                                         f" \'face\' )"
                        save_record(video_id=video_id,
                                    save_dir=abs_save_dir,
                                    mode='video',
                                    name='finished',
                                    time_='finished')
                        logger.info("{} detect finished!".format(video_id))
                        # print("[INFO]: {} detect finished!".format(video_id))
                        try:
                            self.__mysql.execute_command(insert_command)
                        except pymysql.err.IntegrityError:
                            logger.warning("Record: {} is exists!".format(insert_command))
                            # print("[WARNING]: Record: {} is exists!".format(insert_command))
                except pymysql.err.OperationalError:
                    self.__mysql.close()
                    del self.__mysql
                    error = traceback.format_exc()
                    if "Lock wait timeout exceeded; try restarting transaction" in error:
                        logger.warning("The mysql connection was interrupted unexpectedly and is reconnecting!")
                    else:
                        logger.error(error)
                        break
                    # print("[WARNING]: The mysql connection was interrupted unexpectedly and is reconnecting.")
                    self.__init_mysql(self.conf)
                logger.info("RE used: {}".format(time.time() - start_))
                # print("[INFO]: RE using:", time.time() - start_)
        except RuntimeError:
            error = traceback.format_exc()
            if "CUDA" in error:
                logger.warning(" CUDA error, restart a new one!")
                # print("[WARNING]: CUDA error, restart a new one!")
            else:
                logger.error(error)
                # print("[ERROR]:", error)
        except:
            logger.error(traceback.format_exc())
            # print("[ERROR]:", traceback.format_exc())
        finally:
            self.terminate()

    def __release(self):

        try:
            if not self.__mysql is None:
                self.__mysql.close()
                logger.info("Mysql connect closed!")
                # print("[INFO]: Mysql connect closed!")
            if not self.__redis is None:
                self.__redis.close()
                logger.info("Redis connect closed!")
                # print("[INFO]: Redis connect closed!")
            # self.__release_audio()
        except:
            logger.error(traceback.format_exc())
            # print("[ERROR]:", traceback.format_exc())

    def terminate(self) -> None:
        self.__release()
        super(RecognitionProcess, self).terminate()
