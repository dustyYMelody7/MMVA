import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from .pse import decode_batch as pse_decode
# from .pse import decode as pse_decode
from .model import PSENet


def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton


class SingletonType(type):
    def __init__(cls, *args, **kwargs):
        super(SingletonType, cls).__init__(*args, **kwargs)

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        cls.__init__(obj, *args, **kwargs)
        return obj


class PSENetHandel(metaclass=SingletonType):
    def __init__(self, model_path: str, scale: int, gpu_id=None):
        """
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param net: 网络计算图，如果在model_path中指定的是参数的保存路径，则需要给出网络的计算图
        :param img_channel: 图像的通道数: 1,3
        :param gpu_id: 在哪一块gpu上运行
        """
        if gpu_id is not None and isinstance(gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(gpu_id))
        else:
            self.device = torch.device("cpu")
        ########################
        # 这里写死了用mobilenetv2
        ########################
        net = PSENet(backbone="mobilenetv2", pretrained=False, result_num=6, scale=1)
        self.net = torch.load(model_path, map_location=self.device)['state_dict']
        print('device:', self.device)

        net = net.to(self.device)
        net.scale = scale

        try:
            sk = {}
            for k in self.net:
                sk[k[7:]] = self.net[k]
            net.load_state_dict(sk)
        except:

            net.load_state_dict(self.net)

        self.net = net
        print('load model')
        self.net.eval()

    #
    def predict(self, batch_image: np.ndarray, long_size: int = 960):

        # start = time.time()
        if len(batch_image.shape) >= 4:
            b = batch_image.shape[0]
            h, w, c = batch_image[0].shape
        else:
            b = 1
            h, w, c = batch_image.shape[:2]
        # print(h, w)
        if h > w:
            scale_h = long_size / h
            tar_w = w * scale_h
            tar_w = tar_w - tar_w % 32
            tar_w = max(32, tar_w)
            scale_w = tar_w / w
        else:
            scale_w = long_size / w
            tar_h = h * scale_w
            tar_h = tar_h - tar_h % 32
            tar_h = max(32, tar_h)
            scale_h = tar_h / h
        # print(b, c, h, w)
        # batch_image = F.interpolate(batch_image, size=(int(h * scale_h), int(w * scale_w)), mode="bilinear", align_corners=True)
        # print(batch_image.shape)
        # batch_image = np.resize(batch_image, new_shape=(b, int(scale_h * h), int(scale_w * w), c))
        # print(batch_image.shape)
        # batch_image = np.asarray([cv2.resize(image, None, fx=scale_w, fy=scale_h) for image in batch_image])
        batch_image = batch_image.astype(np.float32)
        batch_image /= 255.0
        batch_image -= np.array((0.485, 0.456, 0.406))
        batch_image /= np.array((0.229, 0.224, 0.225))
        tensor = torch.cat([transforms.ToTensor()(item).unsqueeze_(0) for item in batch_image])
        # print(tensor.shape)
        # tensor = F.interpolate(tensor, size=(int(h * scale_h), int(w * scale_w)), mode="bilinear",
        #                            align_corners=True)
        # print(tensor.shape)
        # print("batch_image:", tensor.shape)
        # tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(self.device)
        # print("[INFO]: Data using:", time.time() - start)
        # start = time.time()
        # batch_boxes_list = []
        with torch.no_grad():
            # torch.cuda.synchronize()
            preds = self.net(tensor)
            # print("[INFO]: Batch using:", time.time() - start)
            # start = time.time()
            batch_boxes_list = pse_decode(preds) # list(map(pse_decode, preds))#
            # print("[INFO]: pse_decode using:", time.time() - start)
            num_boxes_list = [len(item) for item in batch_boxes_list]

            # for pred in preds:
            #     boxes_list = pse_decode(pred, self.scale)
            #     print("[INFO]: pse decode using:", time.time() - start)
            #     # start = time.time()
            #
            #     scale = (pred.shape[1] / w, pred.shape[0] / h)
            #     # print(scale)
            #     # preds, boxes_list = decode(preds,num_pred=-1)
            #     if len(boxes_list):
            #         boxes_list = boxes_list / scale
            #         batch_boxes_list.append(boxes_list)
            # print("[INFO]: Analysis using:", time.time() - start)
            # torch.cuda.synchronize()
        return np.asarray(batch_boxes_list), num_boxes_list

if __name__ == "__main__":
    import cv2
    def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
        if isinstance(img_path, str):
            img_path = cv2.imread(img_path)
            # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
        img_path = img_path.copy()
        for point in result:
            point = point.astype(int)

            cv2.polylines(img_path, [point], True, color, thickness)
        return img_path

    text_handle = PSENetHandel("../model/box/psenet_lite_mbv2.pth", 1, gpu_id=0)
    # img = cv2.imread("1.jpg")
    # print(np.expand_dims(img, axis=0).shape)
    img = np.append(np.expand_dims(img, axis=0), np.expand_dims(img, axis=0), axis=0)
    print(img.shape)
    box_list, score_list, num_list = text_handle.predict(img)
    # img = draw_bbox(img, box_list)
    # cv2.imwrite("test.jpg", img)
