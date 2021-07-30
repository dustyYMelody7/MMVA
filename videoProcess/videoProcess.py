import os
import cv2
import json
import time
import datetime
import traceback
import subprocess
import configparser

import numpy as np
from utils.redisTools import RedisHold
from utils.logging import get_log_hold

logger = get_log_hold(os.path.basename(__file__).split('.')[0])


class VideoProcess:

    def __init__(self, video_id: str, video_path: str, config_file: str):
        """
        init.
        :param video_path: video path.
        """
        conf = configparser.ConfigParser()
        conf.read(config_file, encoding='utf-8')
        self.__skip_num = int(conf.get('video', 'skip_num'))
        self.__batch_size = int(conf.get('video', 'batch_size'))
        self.cap = cv2.VideoCapture(video_path)
        self.start = conf.get('redis', 'start')
        self.__redis = RedisHold(host=conf.get('redis', 'host'),
                                 port=int(conf.get('redis', 'port')),
                                 db_name=int(conf.get('redis', 'db')),
                                 passwd=conf.get('redis', 'passwd'),
                                 maximun_memory=int(conf.get('redis', 'maximun_memory')) * (1024 ** 3))
        self.__sleep_time = int(conf.get('redis', 'sleep_time'))

        self.video_info_key = conf.get('redis', 'info_name')
        self.video_id = video_id
        self.scale = int(conf.get('video', 'scale'))
        self.wav_save_dir = os.path.join(conf.get('general', 'output_dir'), video_id, 'wav')
        try:
            os.makedirs(self.wav_save_dir)
        except FileExistsError as e:
            logger.warning("{} dir exist!".format(self.wav_save_dir))
            # print("[WARNING]: {} dir exist!".format(self.wav_save_dir))
        except:
            logger.error(traceback.format_exc())
            # print("[ERROR]:", traceback.format_exc())

        try:
            self.__redis.delete(self.video_id)
        except:
            logger.error(traceback.format_exc())
            # print(traceback.format_exc())
        # 音频识别开启
        audio_dir = self.split_audio(video_path)
        data = {'id': self.video_id,
                'timestamp': '',
                'mode': 'audio',
                'batch_size': audio_dir,
                'image_width': '',
                'image_height': ''
                }
        self.__redis.rpush(name=self.video_info_key, data=json.dumps(data))
        self.__log_file = os.path.join(conf.get('general', 'output_dir'), video_id, video_id + '.log')

    def split_audio(self, video_path):
        audio_dir = os.path.join(self.wav_save_dir, self.video_id + '.wav')
        command = ["ffmpeg",
                   "-y", "-i",
                   video_path,
                   "-ac", str(1),
                   "-ar", str(16000),
                   "-loglevel", "error",
                   audio_dir]
        subprocess.check_output(command, stdin=open(os.devnull), shell=False)
        return audio_dir

    def run(self):
        if os.path.isfile(self.__log_file):
            logger.warning(f"Video: {self.video_id} has been detected!")
            # print(f"[WARNING]: Video: {self.video_id} has been detected!")
            return
        logger.info("Runing!")
        # print("[INFO]: Runing!")
        count = 0
        batch = []
        timestamp = []
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        # 用于判断是否开启后台程序
        flag = True
        try:
            # print(self.cap.isOpened())
            while self.cap.isOpened():
                succeed, frame = self.cap.read()
                count += 1
                if not succeed:
                    break
                if not count % self.__skip_num == 0:
                    continue
                frame = cv2.resize(frame, (frame.shape[1] // self.scale, frame.shape[0] // self.scale))
                batch.append(frame)
                timestamp.append(str(datetime.timedelta(seconds=count/float(fps))))
                if len(batch) >= self.__batch_size:
                    batch = np.asarray(batch, dtype=np.uint8)
                    # print("batch_shape", batch.shape)
                    data = {'id': self.video_id,
                            'timestamp': timestamp,
                            'mode': 'video',
                            'batch_size': batch.shape[0],
                            'image_width': frame.shape[1],
                            'image_height': frame.shape[0]
                            }
                    # 这里是为了保证两句执行切正确完成，可以短路错误但不能短路正确
                    while not (self.__redis.rpush(name=self.video_info_key, data=json.dumps(data)) and
                               self.__redis.rpush(name=self.video_id, data=batch.tobytes())):
                        # 休眠5分钟
                        time.sleep(self.__sleep_time)
                        logger.warning("Re-push data.")
                    if flag:
                        self.__redis.set(name=self.start, value='1')
                        flag = False
                        time.sleep(60)
                    del batch
                    del timestamp
                    batch = []
                    timestamp = []
        except:
            logger.error(traceback.format_exc())
            # print("[ERROR]", traceback.format_exc())
        finally:
            data = {'id': self.video_id,
                    'timestamp': 'finished',
                    'mode': 'finished',
                    'batch_size': 'finished',
                    'image_width': 'finished',
                    'image_height': 'finished'
                    }
            self.__redis.rpush(name=self.video_info_key, data=json.dumps(data))
            logger.info("Video {} finished reading!".format(self.video_id))
            # print("[INFO]: Finished reading!")
            self.release()

    def release(self):
        try:
            if self.cap is not None:
                self.cap.release()
                logger.info("Cap release!")
                # print("[INFO]: Cap release!")
            if self.__redis is not None:
                self.__redis.close()
                logger.info("Redis release!")
                # print("[INFO]: Redis release!")
        except:
            logger.error(traceback.format_exc())

