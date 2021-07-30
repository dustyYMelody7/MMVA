import os
import cv2
import configparser
import numpy as np
import tensorflow as tf


class NSFWModule:
    # 默认必须使用gpu
    def __init__(self, conf: configparser.ConfigParser, gpu_id: int):

        model_path = conf.get("nsfw", "model_path")
        max_memory = int(conf.get("nsfw", "max_memory"))
        gpus = tf.config.list_physical_devices('GPU')
        gpu_used = None
        for item in gpus:
            # print(item)
            if str(gpu_id) in item.name:
                gpu_used = item
        if gpu_used is None:
            raise ValueError("Your machine mayn't have %d gpu, please check your 'gpu_id'!" % (gpu_id))
        tf.config.experimental.set_virtual_device_configuration(
            gpu_used,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=max_memory)])
        self.gpu = '/device:GPU:' + str(gpu_id)
        with tf.device(self.gpu):
            self.model = self.__load_model(model_path)

    def __load_model(self, model_path: str):
        if model_path is None or not os.path.exists(model_path):
            raise ValueError("saved_model_path must be the valid directory of a saved model to load.")

        model = tf.keras.models.load_model(model_path)  # , custom_objects={'KerasLayer': hub.KerasLayer})
        return model

    def detect(self, batch_image: np.ndarray):
        batch_image = np.asarray([cv2.resize(cv2.cvtColor(item, cv2.COLOR_BGR2RGB), (224, 224))
                                  for item in batch_image], dtype=np.float32) / 255
        result = []
        with tf.device(self.gpu):
            model_preds = self.model.predict(batch_image)
            categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
            for item in model_preds.tolist():
                class_ = item.index(max(item))
                result.append(categories[class_])
        result = [item if item != "neutral" else None for item in result]
        return result

