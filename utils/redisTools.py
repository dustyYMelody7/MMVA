import json
import time
import redis
import logging
import traceback

import numpy as np
from typing import Union
# from helpers import base64_decode_image, base64_encode_image
# logger = get_log_hold(os.path.basename(__file__).split('.')[0])


class RedisHold:

    def __init__(self, host: str = 'localhost',
                 port: int = 6379,
                 db_name: int = 0,
                 passwd: str = '123456',
                 maximun_memory: int = 30 * (1024 ** 3),
                 logger: Union[logging.Logger, None] = None):

        assert isinstance(host, str), "[ERROR]: Argument of 'host' must be str!"
        assert isinstance(port, int), "[ERROR]: Argument of 'port' must be int!"
        assert logger is not None, "[ERROR]: You need init logger!"

        if passwd == '':
            raise ValueError("[ERROR]:  Argument of 'passwd' is empty!")

        self.db = redis.Redis(host=host, port=port, db=db_name, password=passwd)
        self.maximun_memory = maximun_memory
        self.logger = logger

    def pop_image(self, name: str, batch_size: int, image_height: int, image_width: int) -> np.ndarray:

        try:
            # 这样操作必须保证每次都是从队尾取元素
            if self.db.llen(name=name) == 0:
                return None
            else:
                images = self.db.lpop(name=name)
            if images is None:
                return None
            images = np.fromstring(images, dtype=np.uint8).reshape((batch_size, image_height, image_width, 3))
            return images
        except Exception as e:
            self.logger.error(traceback.format_exc())
            # print(traceback.format_exc())
        return None

    def pop_audio(self, name: str) -> str:

        try:
            # 这样操作必须保证每次都是从队尾取元素
            if self.db.llen(name=name) == 0:
                return None
            else:
                audio_file = self.db.lpop(name=name)
            if audio_file is None:
                return None
            audio_file = str(audio_file, 'utf-8')
            return audio_file
        except Exception as e:
            self.logger.error(traceback.format_exc())
            # print(traceback.format_exc())
        return None

    def pop_info(self, name: str = 'video_filter_info') -> tuple:
        """
        pop a batch from redis.
        :param name: the key name.
        :return: images information.
        """
        try:
            # 这里舍弃lpop方法在于lpop方法会产生饥饿丢失的问题，lpop每次只能取到当前位置的视频信息可能会一直忽略以前没有读的信息
            # 一定得从队尾取元素
            # print(name)
            if self.db.llen(name=name) == 0:
                return None
            else:
                queue = self.db.lpop(name=name)
            # 避免读出空值
            if queue is None:
                return None
            queue = json.loads(queue)
            if len(queue) > 1:
                return queue['id'], queue['timestamp'], queue['batch_size'], queue['image_width'], queue['image_height'], queue['mode']
            else:
                return queue

        except Exception as e:
            self.logger.error(traceback.format_exc())
            # print(traceback.format_exc())
        return None, None, None, None, None, None

    def pop(self, mode, **kwargs):
        if mode == 'info':
            return self.pop_info(**kwargs)
        elif mode == 'image':
            return self.pop_image(**kwargs)
        elif mode == 'audio':
            return self.pop_audio(**kwargs)
        else:
            self.logger.error("Argument of 'mode' must in ['info', 'image', 'audio'], but {}".format(mode))
            raise ValueError("[ERROR]: Argument of 'mode' must in ['info', 'image', 'audio'], but {}".format(mode))

    def rpush(self, name: str, data) -> bool:

        # images = base64_encode_image(images.tostring())
        # data = json.dumps(data)
        # print(len(data))
        try:
            # print(images)
            # print("images", type(images))
            # print(len(images))
            # rpush是往队尾插入元素, lpush是往对头插入元素
            if not self.is_lock():
                self.db.rpush(name, data)
                # self.logger.info("insert! name: {}".format(name))
                return True
            else:
                self.logger.warning("Memory usage is too high.")
                return False
            # print("[INFO]: insert! name:", name)
        except:
            self.logger.error(traceback.format_exc())
            # print("[ERROR]:", traceback.format_exc())
            return False

    def is_lock(self):
        used_memory = int(self.db.info()["used_memory"])
        # print(used_memory)
        if used_memory >= self.maximun_memory:
            return True
        return False

    def get_keys(self):
        return self.db.keys()

    def get(self, name):
        return self.db.get(name=name)

    def set(self, name, value):
        self.db.set(name, value)

    def close(self):
        self.db.close()

    def delete(self, *name):
        self.logger.info("Redis delete db: {}".format(*name))
        # print(*name)
        self.db.delete(*name)


if __name__ == '__main__':

    import os
    import cv2
    import argparse

    parse = argparse.ArgumentParser()

    parse.add_argument('--batch-size', type=int, default=8)
    parse.add_argument('--skip-num', type=int, default=5)

    args = parse.parse_args()

    def save_image(batch: np.ndarray):
        print("save")
        if batch is None:
            return

        for i, item in enumerate(batch):
            r, g, b = cv2.split(item)
            item = cv2.merge([b, g, r])
            cv2.imwrite(os.path.join('result', str(i) + '.jpg'), item)

    sql = RedisHold(db_name=0, passwd='lmc.T4.server')

    batch = []
    timestamp = []
    cap = cv2.VideoCapture('image/test.mp4')
    if not os.path.exists('result'):
        os.mkdir('result')
    num = 0
    while cap.isOpened():
        _, frame = cap.read()
        num += 1
        if not num % args.skip_num == 0:
            continue
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[2] // 2))
        # print(type(image))
        batch.append(frame)
        timestamp.append(num)
        if len(batch) == args.batch_size:
            #print(batch.shape)
            ###############################
            ##############################
            #############################
            # 这里的dtype记得加类型，不然会出错的！！！！
            #############################
            ##############################
            ###############################
            batch = np.asarray(batch, dtype=np.int)
            print("batch_shape", batch.shape)
            data = {'id': 'test',
                    'timestamp': timestamp,
                    'mode': 'video',
                    'batch_size': batch.shape[0],
                    'image_width': frame.shape[1],
                    'image_height': frame.shape[2]
            }
            if not sql.rpush(name='video_filter_info', data=json.dumps(data)):
                break
            data = batch.tobytes()
            print("len of data", len(data))
            if not sql.rpush(name='test', data=data):
                break
            del batch
            del timestamp
            batch = []
            timestamp = []
            start_time = time.time()
            video_id, timestamp_, batch_size, image_width, image_height, mode = sql.pop_info(name='video_filter_info')
            print(video_id, timestamp_, mode)
            images = sql.pop_image(name=video_id, batch_size=batch_size, image_width=image_width, image_height=image_height)
            print("Get data using:", time.time() - start_time)
            if images is None:
                break
            save_image(images)




