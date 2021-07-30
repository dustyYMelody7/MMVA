import os
import configparser

from typing import Union
from utils.common import ReBase
from utils.logging import get_log_hold
from nsfw import NSFWModule


class NR(ReBase):

    def __init__(self, config_files: Union[str, list, tuple], gpu_id: int = 0):
        super(NR, self).__init__(config_files, gpu_id)
        self.mode = 'nsfw'
        self._file = os.path.dirname(__file__)

    # @configurable
    def _run_init(self):
        self.logger = get_log_hold(os.path.basename(__file__).split('.')[0])
        super(NR, self)._run_init()
        assert self._mysql is not None and self._redis_read is not None, self.logger.error("Mysql Or Redis init error!")
        conf = configparser.ConfigParser()
        conf.read(self.config_files, encoding='utf-8')
        ##################################
        # 各模块的初始化位置在这里有些细微的差别
        self._model = NSFWModule(conf=conf, gpu_id=self._gpu_id)
        self.logger.info("Nsfw module init!")
        del conf
