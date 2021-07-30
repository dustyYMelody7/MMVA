import os
import cv2
import random
import argparse
import configparser

import numpy as np

from model import *
from PIL import Image
from utils.mysqlTools import MySqlHold
from recognition.face.mtcnn.mtcnn import MTCNN


def update_face_feature(data_dir, save_dir, mtcnn, model, mysql):

    features = []
    names = ['Unknow']
    model.eval()
    for item in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, item)):
            # print(item)
            continue
        # print(item)
        # 随机取一张
        file_list = [os.path.join(data_dir, item, val) for val in os.listdir(os.path.join(data_dir, item))]
        file_ = file_list[random.randint(0, len(file_list) - 1)]
        try:
            print(file_)
            # img = Image.open(os.path.join(file_))
            img = cv2.imread(os.path.join(file_))
        except:
            # print(traceback.format_exc())
            continue
        bboxes, score, face = mtcnn.detect(img, return_face=True)
        if bboxes is not None and bboxes.shape[0] > 1:
            raise RuntimeError("[ERROR]: {} picture has more than one face!".format(file_))
        if bboxes is None:
            continue
        # face = mtcnn.extract(img, bboxes, point)
        # print("face shape", face.shape)
        feature = model(face)
        features.append(feature)
        names.append(item)
        feature_copy = feature.cpu().detach().numpy()
        print(feature_copy.shape)
        feature_copy = np.squeeze(feature_copy).tolist()
        feature_copy = [str(item) for item in feature_copy]
        print(feature_copy)
        feature_copy = ','.join(feature_copy)
        print(len(feature_copy))
        insert_command = f'insert into face_feature (name, feature) value(\'{item}\', \'{feature_copy}\')'
        mysql.execute_command(insert_command)
        print("[INFO]: Insert succeed!")

    # print(len(features))
    features = torch.cat(features)
    print("features:", features.shape)
    names = np.asarray(names)
    torch.save(features, os.path.join(save_dir, 'facebank.pth'))
    np.save(os.path.join(save_dir, 'names.npy'), names)


def main(image_path):

    conf = configparser.ConfigParser()
    conf.read('config/config.cfg', encoding='utf-8')
    # print(conf.get('face', 'model_path'))
    # print(conf.get('general', 'resize'))
    model = torch.load(conf.get('face', 'model_path')).cuda()
    model.eval()
    mtcnn = MTCNN(image_size=(int(conf.get('face', 'resize')),
                              int(conf.get('face', 'resize'))),
                  post_process=False, device='cuda:0')
    mysql = MySqlHold(host=conf.get('mysql', 'host'),
                      user=conf.get('mysql', 'user'),
                      password=conf.get('mysql', 'passwd'),
                      database=conf.get('mysql', 'db'),
                      port=int(conf.get('mysql', 'port')))

    update_face_feature(image_path, 'image/', mtcnn, model, mysql)
    mysql.close()

if __name__ == '__main__':
    """
    图片名命格式:
    dir:
        name1:
            picture1.jpg
            picture2.jpg
            picture3.jpg
            ...
        name2:
            picture1.jpg
            picture2.jpg
            picture3.jpg
            ...
        name3:
            picture1.jpg
            picture2.jpg
            picture3.jpg
            ...
    """
    parse = argparse.ArgumentParser()
    parse.add_argument('--image-dir', type=str, default='image/face')
    args = parse.parse_args()
    main(args.image_dir)




