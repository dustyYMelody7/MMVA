import cv2
import numpy as np
import configparser
from recognition.nsfw import NSFWModule

if __name__ == "__main__":
    config_file = "../config/config.cfg"
    conf = configparser.ConfigParser()
    conf.read(config_file, encoding='utf-8')
    text_module = NSFWModule(conf=conf, gpu_id=0)
    text_module.detect(np.expand_dims(cv2.imread("../data/test_image/nsfw_test.jpg"), axis=0))