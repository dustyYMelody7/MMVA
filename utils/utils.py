import numpy as np
import torchvision.transforms as trans

from .mysqlTools import MySqlHold


def get_feature(mysql: MySqlHold, table_name: str = 'face_feature') -> tuple:
    command = f'select * from {table_name}'
    mysql.execute_command(command)
    result = mysql.fetchall()
    features = []
    names = ['Unknown']
    for i, name, item in result:
        item = np.asarray([float(val) for val in item.split(',')], dtype=np.float64)
        features.append(item)
        names.append(name)
    features = np.asarray(features, dtype=np.float64)
    features = trans.ToTensor()(features).squeeze(0)
    # logger.info("Get video_filter.{} table!".format(table_name))
    return features, names
