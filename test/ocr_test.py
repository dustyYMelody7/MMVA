import cv2
import numpy as np
import configparser
from recognition.ocr import ocrModule

if __name__ == "__main__":
    config_file = "../config/config.cfg"
    conf = configparser.ConfigParser()
    conf.read(config_file)
    text_module = ocrModule.OCRModule(conf=conf, name_list=["范冰冰"], gpu_id=0)
    text_module.detect(np.expand_dims(cv2.imread("../data/test_image/ocr_test.jpg"), axis=0))
