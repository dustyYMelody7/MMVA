
import torch
import configparser

import numpy as np

from utils.mysqlTools import MySqlHold

conf = configparser.ConfigParser()
conf.read('config/config.cfg')

mysql = MySqlHold(host=conf.get('mysql', 'host'),
                      user=conf.get('mysql', 'user'),
                      password=conf.get('mysql', 'passwd'),
                      database=conf.get('mysql', 'db'),
                      port=int(conf.get('mysql', 'port')))

face_banck = 'image/facebank.pth'
name_file = 'image/names.npy'

face_feature = torch.load(face_banck).cpu().detach().numpy()
name_list = np.load(name_file).tolist()[1:]
print(face_feature.shape)
print(name_list)

for i, item in enumerate(name_list):
    feature = face_feature[i].tolist()
    feature = [str(item) for item in feature]
    feature = ','.join(feature)
    insert_command = f'insert into face_feature (name, feature) value(\'{item}\', \'{feature}\')'
    mysql.execute_command(insert_command)
    print("[INFO]: Insert succeed!")

mysql.close()


