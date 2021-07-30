import subprocess
import os
import time
import numpy as np
import cv2
import torch
from .pse import pse_cpp

# BASE_DIR = os.path.dirname(os.path.realpath(__file__))
#
# if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
#     raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))

def pse_warpper(kernals, min_area=5):
    '''
    reference https://github.com/liuheng92/tensorflow_PSENet/blob/feature_dev/pse
    :param kernals:
    :param min_area:
    :return:
    '''
    # start = time.time()
    # print(kernals.shape)
    kernal_num = len(kernals)
    if not kernal_num:
        return np.array([]), []
    kernals = np.array(kernals)

    label_num, label = cv2.connectedComponents(kernals[0].astype(np.uint8), connectivity=4)
    # print("[INFO]: connectedComponents using:", time.time() - start)
    # start = time.time()
    label_values = []
    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0
            continue
        label_values.append(label_idx)
    # print("[INFO]: for using:", time.time() - start)

    # start = time.time()
    pred = pse_cpp(label, kernals, c=kernal_num)
    # print("[INFO]: pse_cpp using:", time.time() - start)

    return np.array(pred), label_values


def decode(preds, scale=1.0,
           threshold=0.7311,
           # threshold=0.7
           no_sigmode = False
           ):
    """
    在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
    :param preds: 网络输出
    :param scale: 网络的scale
    :param threshold: sigmoid的阈值
    :return: 最后的输出图和文本框
    """
    # start = time.time()
    if not no_sigmode:
        preds = torch.sigmoid(preds)
        preds = preds.detach().cpu().numpy()
    # print("[INFO]: detach using:", time.time() - start)
    # start = time.time()

    # print("detach: ", preds.shape)
    score = preds[-1].astype(np.float32)
    # print("score", score.shape)

    preds = preds > threshold
    # print("threshold: ", preds.shape)
    # preds = preds * preds[-1] # 使用最大的kernel作为其他小图的mask,不使用的话效果更好
    pred, label_values = pse_warpper(preds, 5)
    # print("[INFO]: pse_warpper using:", time.time() - start)


    bbox_list = []

    for label_value in label_values:
        # print("for: ", pred.shape)
        # print("label_value: ", label_value)
        points = np.array(np.where(pred == label_value)).transpose((1, 0))[:, ::-1]

        if points.shape[0] < 800 / (scale * scale):
            continue

        score_i = np.mean(score[pred == label_value])
        if score_i < 0.93:
            continue

        rect = cv2.minAreaRect(points)

        bbox = cv2.boxPoints(rect)

        bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])

    return np.array(bbox_list)


def decode_batch(batch_preds, scale=1.0, threshold=0.7311, no_sigmode=False):
    """
    在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
    :param preds: 网络输出
    :param scale: 网络的scale
    :param threshold: sigmoid的阈值
    :return: 最后的输出图和文本框
    """
    # start = time.time()
    if not no_sigmode:
        batch_preds = torch.sigmoid(batch_preds)
        batch_preds = batch_preds.detach().cpu().numpy()
    # print("[INFO]: detach using:", time.time() - start)
    # start = time.time()
    # print("detach: ", batch_preds.shape)

    score = batch_preds[:,-1].astype(np.float32)
    # print("score", score.shape)

    batch_preds = batch_preds > threshold
    # print("threshold: ", batch_preds.shape)
    # preds = preds * preds[-1] # 使用最大的kernel作为其他小图的mask,不使用的话效果更好
    results = list(map(pse_warpper, batch_preds))
    # print("[INFO]: pse_wapper using:", time.time() - start)
    batch_boxes = []
    # print(results)
    for i, result in enumerate(results):
        pred, label_values = result
        bbox_list = []
        for label_value in label_values:
            # print("for: ", pred.shape)
            # print("label_value: ", label_value)
            points = np.array(np.where(pred == label_value)).transpose((1, 0))[:, ::-1]
            if points.shape[0] < 800 / (scale * scale):
                continue

            score_i = np.mean(score[i][pred == label_value])
            if score_i < 0.93:
                continue

            rect = cv2.minAreaRect(points)

            bbox = cv2.boxPoints(rect)
            bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])
        batch_boxes.append(bbox_list)

    return np.array(batch_boxes)
