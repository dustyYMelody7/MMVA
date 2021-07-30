import logging
import configparser

import numpy as np

from typing import Union
from utils.mysqlTools import MySqlHold
from utils.utils import get_feature
from .mtcnn.mtcnn import MTCNN
# 保证MobileNet的模型能加载进来
from .model import *
#
# logger = get_log_hold(os.path.basename(__file__).split('.')[0])


def face_infer(batch_feature: torch.Tensor, embedding_feature: torch.Tensor, name_list: list, threshold: float = 1.5):

    diff = batch_feature.unsqueeze(-1) - embedding_feature.transpose(1, 0).unsqueeze(0)
    dist = torch.sum(torch.pow(diff, 2), dim=1)
    minimum, min_idx = torch.min(dist, dim=1)
    min_idx[minimum > threshold] = -1  # if no match, set idx to -1

    names = [name_list[min_idx[idx] + 1] for idx in range(len(batch_feature))]

    return names, minimum


class FaceModule:

    def __init__(self, conf: configparser.ConfigParser,
                 mysql: MySqlHold,
                 gpu_id: int = 0,
                 logger: Union[logging.Logger, None] = None):

        if gpu_id is not None and isinstance(gpu_id, int) and torch.cuda.is_available():
            device = torch.device("cuda:{}".format(gpu_id))
        else:
            device = torch.device("cpu")
        self.__model = torch.load(conf.get('face', 'model_path'))
        self.__model.to(device)
        self.__model.eval()
        self.__mtcnn = MTCNN(image_size=(int(conf.get('face', 'resize')), int(conf.get('face', 'resize'))),
                             post_process=False, device=device)
        self.__mtcnn.eval()
        self.threshold = float(conf.get('face', 'threshold'))
        self.embeding_face_feature, self.embeding_name_list = get_feature(
            mysql, conf.get('mysql', 'face_feature_table'))
        self.embeding_face_feature = self.embeding_face_feature.to(device)
        if logger is not None:
            logger.info("Face Module init finished!")
        # print("[INFO]: Face Module init finished!")

    def detect(self, batch_image: np.ndarray) -> tuple:
        # print("[INFO]: Face detect!")
        # logger.info("Face detect!")
        batch_bboxes, _, batch_faces_mtcnn = self.__mtcnn.detect(batch_image, return_face=True)
        if batch_faces_mtcnn is None:
            return None, None
        # 感觉没用，这个逻辑删除
        # 原本是打算用这个逻辑解决人脸数目多时gpu使用占比高的情况，但是cat操作也会增加gpu占用，因此没效果。
        # if len(batch_faces_mtcnn) > 8:
        #     batch_faces_feature = None
        #     for i in range(0, len(batch_faces_mtcnn), 8):
        #         if batch_faces_feature is None:
        #             batch_faces_feature = self.__model(batch_faces_mtcnn[i:i+8])
        #         else:
        #             batch_faces_feature = torch.cat([batch_faces_feature, self.__model(batch_faces_mtcnn[i:i+8])])
        # else:
        #     batch_faces_feature = self.__model(batch_faces_mtcnn)
        batch_faces_feature = self.__model(batch_faces_mtcnn)
        num_faces_record = [len(item) if item is not None else 0 for item in batch_bboxes]
        name_list, _ = face_infer(batch_feature=batch_faces_feature,
                                  embedding_feature=self.embeding_face_feature,
                                  name_list=self.embeding_name_list,
                                  threshold=self.threshold)
        del batch_bboxes, batch_faces_feature, batch_faces_mtcnn
        return name_list, num_faces_record

