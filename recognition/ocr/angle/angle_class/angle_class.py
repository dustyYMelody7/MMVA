import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image


class AangleClassHandle:
    def __init__(self, model_path: str, net, gpu_id=None):
        """
           初始化pytorch模型
           :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
           :param net: 网络计算图，如果在model_path中指定的是参数的保存路径，则需要给出网络的计算图

           :param gpu_id: 在哪一块gpu上运行
           """
        
        if gpu_id is not None and isinstance(gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(gpu_id))
        else:
            self.device = torch.device("cpu")
        self.net = torch.load(model_path, map_location=self.device)
        print('device:', self.device)
        
        self.trans = transforms.Compose([
            # transforms.Resize((int(48 / 1.0), int(196 / 0.875))),
            # transforms.CenterCrop((48, 196)),
            #
            transforms.Resize((48, 196)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        if net is not None:
            # 如果网络计算图和参数是分开保存的，就执行参数加载
            net = net.to(self.device)
            
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
    
    def predict(self, batch_image: torch.Tensor) -> np.ndarray:
        """
        预测
        """
        # 处理batch或单独的image
        if len(batch_image.shape) >= 4:
            # batch_image = [Image.fromarray(item).convert("RGB") for item in batch_image]
            # batch_image = torch.cat([self.trans(item).unsqueeze_(0) for item in batch_image]).to(self.device)
            batch_image = batch_image.to(self.device)
        else:
            # batch_image = Image.fromarray(batch_image).convert("RGB")
            # batch_image = self.trans(batch_image)
            batch_image = batch_image.to(self.device)
            batch_image = batch_image.view(1, *batch_image.size())
        batch_image = Variable(batch_image)
        preds = self.net(batch_image)
        preds = torch.softmax(preds, 1)
        preds = preds.cpu().detach().numpy()
        # 以batch的方式处理
        preds = np.argmax(preds, axis=1)
        return preds


"""
今日问题：
    如何改这个predict函数为batch，一次检测到所有的偏转角预测值，如果不行直接舍弃偏转角检测。
"""
