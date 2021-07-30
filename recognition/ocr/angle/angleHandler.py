from .angle_class import AangleClassHandle, shufflenet_v2_x0_5

LABEL_MAP_DICT = {0: "hengdao", 1: "hengzhen", 2: "shudao", 3: "shuzhen"}
ROTAE_MAP_DICT = {"hengdao": 180, "hengzhen": 0, "shudao": 180, "shuzhen": 0}

class AngleHandler(AangleClassHandle):

    def __init__(self, model_path: str, gpu_id: int):

        super(AngleHandler, self).__init__(model_path,
                                           shufflenet_v2_x0_5(num_classes=len(LABEL_MAP_DICT), pretrained=False),
                                           gpu_id=gpu_id)