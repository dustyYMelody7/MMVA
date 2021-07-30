import os
import configparser

from typing import Union
from utils.common import ReBase
from utils.logging import get_log_hold
from utils.utils import get_feature
from ocr import OCRModule


class OR(ReBase):

    def __init__(self, config_files: Union[str, list, tuple], gpu_id: int = 0):
        super(OR, self).__init__(config_files, gpu_id)
        self.mode = 'ocr'
        self._file = os.path.dirname(__file__)

    # @configurable
    def _run_init(self):
        self.logger = get_log_hold(os.path.basename(__file__).split('.')[0])
        super(OR, self)._run_init()
        assert self._mysql is not None and self._redis_read is not None, self.logger.error("Mysql Or Redis init error!")
        conf = configparser.ConfigParser()
        conf.read(self.config_files, encoding='utf-8')
        ##################################
        # 各模块的初始化位置在这里有些细微的差别
        _, name_list = get_feature(self._mysql, self._mysql_dict['face_feature_table'])
        self._model = OCRModule(conf=conf, name_list=name_list, gpu_id=self._gpu_id)
        self.logger.info("Ocr module init!")
        del conf
