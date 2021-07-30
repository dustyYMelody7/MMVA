import cv2
import torch
import ahocorasick
import configparser

import numpy as np

from PIL import Image
from torchvision import transforms

from .crnn import CRNNHandler
from .psenet import PSENetHandler
from .angle import AngleHandler, LABEL_MAP_DICT, ROTAE_MAP_DICT

from .utils import sorted_boxes, get_rotate_crop_image


class OCRModule:

    def __init__(self, conf: configparser.ConfigParser, name_list: list, gpu_id: int = 0):

        # device_id = int(conf.get('ocr', 'device'))
        self.__text_handler = CRNNHandler(conf.get('ocr', 'text_model_file'), gpu_id)
        self.__text_handler_vertical = CRNNHandler(conf.get('ocr', 'text_model_vertical_file'), gpu_id)
        self.__box_handler = PSENetHandler(conf.get('ocr', 'box_model_file'),
                                           int(conf.get('ocr', 'pse_scale')), gpu_id)
        self.__angle_handler = AngleHandler(conf.get('ocr', 'angle_model_file'), gpu_id)
        self.__name_list = name_list

    def detect(self, batch_image: np.ndarray) -> list:

        trans = transforms.Compose([
            # transforms.Resize((int(48 / 1.0), int(196 / 0.875))),
            # transforms.CenterCrop((48, 196)),
            #
            transforms.Resize((48, 196)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # start = time.time()
        batch_image = np.asarray([cv2.cvtColor(item, cv2.COLOR_BGR2RGB) for item in batch_image], dtype=np.uint8)
        batch_boxes_list, num_boxes_list = self.__box_handler.predict(batch_image=batch_image)
        # print("[INFO]: Boxes using:", time.time() - start)
        # start_ = time.time()
        text_list = []
        # print(batch_boxes_list)
        batch_boxes_list = [sorted_boxes(np.array(item)) for item in batch_boxes_list]
        for index, boxes in enumerate(batch_boxes_list):
            # rect = cv2.minAreaRect(box)
            # degree, w, h, cx, cy = rect
            # box = sorted_boxes(box)
            # tmp_boxes = copy.deepcopy(boxes)
            if len(boxes) == 0:
                continue
            # print("[INFO]: Copy using:", time.time() - start_)
            # start = time.time()
            # 这里与原版本不同，采用优先固定size的方式可以使用asarray方法转为可读的batch
            # print(boxes)
            # batch_partImg_array = np.asarray(
            #     [cv2.resize(get_rotate_crop_image(batch_image[index], tmp_box.astype(np.float32)),
            #                (48, 196)) for tmp_box in boxes], dtype=np.uint8)
            # batch_partImg_array = torch.tensor([trans(get_rotate_crop_image(batch_image[index], tmp_box.astype(np.float32)))
            #                        for tmp_box in boxes])
            batch_partImg_array = [trans(Image.fromarray(
                get_rotate_crop_image(batch_image[index], tmp_box.astype(np.float32))).convert("RGB")).detach().numpy()
                                   for tmp_box in boxes]
            # print(len(batch_partImg_array))
            batch_partImg_array = torch.tensor(batch_partImg_array)
            # print(batch_partImg_array.shape)

            angel_index = self.__angle_handler.predict(batch_partImg_array)

            angel_class = [LABEL_MAP_DICT[item] for item in angel_index]
            rotate_angle = [ROTAE_MAP_DICT[item] for item in angel_class]

            # print("[INFO]: Angle using:", time.time() - start_)
            # start = time.time()
            if rotate_angle != 0:
                batch_partImg_array = np.asarray(
                    [np.rot90(batch_partImg_array[i], rotate_angle[i] // 90) for i in range(len(rotate_angle))], dtype=np.uint8)

            # batch_partImg = [Image.fromarray(item).convert("RGB") for item in batch_partImg_array]
            batch_partImg = [Image.fromarray(
                get_rotate_crop_image(batch_image[index], tmp_box.astype(np.float32))).convert("RGB") for tmp_box in boxes]
            #
            # partImg.save("./debug_im/{}.jpg".format(index))

            batch_partImg_ = [item.convert('L') for item in batch_partImg]
            text_ahocor = None
            try:
                # 这里可读性比较差，解读下，就是这里需要处理batch，但是需要判断角度，所以采取简单的方式完成任务
                simPreds = [self.__text_handler_vertical.predict(item) if angel_class[i] in ["shudao", "shuzhen"] else
                            self.__text_handler.predict(item) for i, item in enumerate(batch_partImg_)]
                # print("[INFO]: Angle using:", time.time() - start_)
                # # 消除空字符串
                # simPreds = [item if item.strip() != u'' else None for item in simPreds]
                # text_ahocor = ahocorasick.Automaton()
                # print(simPreds)
                # for index, word in enumerate(simPreds):
                #     text_ahocor.add_word(word, (index, word))
            except:
                continue
            # print(text_ahocor)
            text_list.append(simPreds)

        # print("[INFO]: Text using:", time.time() - start_)
        # start = time.time()
        # print(text_list)
        # print(self.__name_list)
        name_list = []
        for i, item in enumerate(text_list):
            flag = False
            # print(item)
            if item is None:
                continue
            for name in self.__name_list:
                if flag:
                    break
                for sentence in item:
                    # print(sentence)
                    if flag:
                        break
                    if len(name) < 3:
                        if name in sentence:
                            if num_boxes_list[i]:
                                name_list.append(name)
                            else:
                                while len(name_list) <= i - 1:
                                    name_list.append(None)
                                name_list.append(name)
                            flag = True
                    else:
                        # 这里使按三个子的名字的逻辑写的
                        if name[0:2] in sentence or name[1:] in sentence or name[0::2] in sentence:
                            if num_boxes_list[i]:
                                name_list.append(name)
                            else:
                                while len(name_list) <= i - 1:
                                    name_list.append(None)
                                name_list.append(name)
                            flag = True
        # print("[INFO]: Analysis using:", time.time() - start)
        if list(set(name_list)) == [None]:
            return None
        return name_list


if __name__ == "__main__":
    config_file = "../../config/config.cfg"
    conf = configparser.ConfigParser()
    conf.read(config_file)
    text_module = OCRModule(conf=conf, name_list=["冯新柱"], gpu_id=0)




