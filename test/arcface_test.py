
import os
import cv2
import torch
import time
import argparse


import numpy as np

from PIL import Image
from recognition.face.mtcnn.mtcnn import MTCNN
from recognition.face.faceModule import face_infer
# from recognition.face.model import *
import traceback
# from model import MobileFaceNet
from torchvision import transforms

"""
新问题：2021 4-5 15:37
如果一张图片上不止一个人脸，那么将每个batch与mtcnn的结果对应起来就会成为问题即这是一个1 -> n的问题
    暂时能想到的解决方案有两种：
        1、采用字典数据保存结果，即{1: [mtcnn1, mtcnn2 ...]}
        2、松耦合方式，采用两段存储，一个list存计数
"""

def get_config():

    parse = argparse.ArgumentParser()

    parse.add_argument('--batch-size', type=int, default=8)
    parse.add_argument('--skip-num', type=int, default=5)
    parse.add_argument('--device', type=str, default='cuda:0')
    parse.add_argument('--name-file', type=str, default='image/test/names.npy')
    parse.add_argument('--face-bank', type=str, default='image/test/facebank.pth')
    parse.add_argument('--data-dir', type=str, default='image/test/images')
    parse.add_argument('--save-dir', type=str, default='image/test/')
    parse.add_argument('--update', action='store_true', default=False)
    parse.add_argument('--model-path', type=str, default='recognition/face/model/mobilenetfacenet.pth')


    args = parse.parse_args()
    return args

def update_face_feature(args, mtcnn, model):

    data_dir = args.data_dir

    features = []
    names = ['Unknow']
    for item in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, item)):
            # print(item)
            continue
        # print(item)
        for file_ in os.listdir(os.path.join(data_dir, item)):
            try:
                print(file_)
                img = Image.open(os.path.join(data_dir, item, file_))
            except:
                # print(traceback.format_exc())
                continue
            bboxes, score = mtcnn.detect(img)
            if bboxes is None:
                continue
            face = mtcnn.extract(img, bboxes)
            # print("face shape", face.shape)
            feature = model(face)
            features.append(feature)
        names.append(item)

    # print(len(features))
    features = torch.cat(features)
    print("features:", features.shape)
    names = np.asarray(names)
    torch.save(features, os.path.join(args.save_dir, 'facebank.pth'))
    np.save(os.path.join(args.save_dir, 'names.npy'), names)


def main():

    args = get_config()
    name_list = np.load(args.name_file)
    embedding_feature = torch.load(args.face_bank)
    print("embedding_feature:", embedding_feature.shape)
    # currentpath = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join('image/result')
    print(name_list)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    model_mobile = torch.load(args.model_path)
    model_mobile = model_mobile.to(args.device)
    mtcnn = MTCNN(image_size=(112, 112), post_process=False, device=args.device)
    model_mobile.eval()
    if args.update is True:
        update_face_feature(args, mtcnn, model_mobile)
    video_path = os.path.join('image/test/test.mp4')

    cap = cv2.VideoCapture(video_path)

    batch = []
    count = 0
    start_time_ = time.time()
    while cap.isOpened():
        succeed, frame = cap.read()
        # print(frame.shape)
        if not succeed:
            break
        count += 1
        if count % args.skip_num == 0:
            # b, g, r = cv2.split(frame)
            # image = cv2.merge([r, g, b])
            # print(frame.shape)
            frame = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))
            print('frame:', frame.shape)
            batch.append(frame)

        # print("len batch", len(batch))
        if len(batch) == args.batch_size:
            start_time = time.time()
            batch_bboxes, probs, batch_faces = mtcnn.detect(np.asarray(batch), return_face=True)
            batch.clear()
            # print(probs)
            if batch_bboxes is None or len(batch_bboxes) <= 0:
                continue
            # batch_faces = mtcnn.extract(batch, batch_bboxes, landmarks)
            face_num = [len(item) if not item is None else 0 for item in batch_bboxes]
            print("-" * 50)
            # print("face num", face_num)
            # print("bboxes:", batch_bboxes[0].shape if batch_bboxes[0] is not None else 0)
            # batch_faces = [item for item in batch_faces if item is not None]
            # batch_faces = torch.Tensor(batch_faces).to(args.device)
            # print("faces:", type(batch_faces[0]))
            if batch_faces is None:
                continue
            # print("face shape:", batch_faces.shape)
            features = model_mobile(batch_faces)
            names, score = face_infer(features, embedding_feature, name_list)
            # print("features:", type(features), features.shape)
            print("Using:", time.time() - start_time)
            print("names:", names)
            print("score:", score)
            print("-" * 50)
            # for i, face in enumerate(batch_faces):
            #     if face is None:
            #         continue
            #     face = face.cpu().detach().numpy()
            #     face = np.transpose(face, (1, 2, 0))
            #     #print(face.shape)
            #     cv2.imwrite(os.path.join(result_dir, str(num) + '_' + str(i) + '.jpg'), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            # for i, bboxes in enumerate(batch_bboxes):
            #     if bboxes is None:
            #         continue
            #     # bbox = bbox[0]
            #     rectangle = batch[i]
            #     flag = 0
            #     for bbox in bboxes:
            #         if flag == 0:
            #             r, g, b = cv2.split(rectangle)
            #             rectangle = cv2.merge([b, g, r])
            #             flag = 1
            #         rectangle = cv2.rectangle(rectangle, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
            #
            #     cv2.imwrite(os.path.join(result_dir, str(num) + 'rec' + str(i) + '.jpg'), rectangle)



    print("USING:", time.time() - start_time_)


if __name__ == '__main__':
    main()
