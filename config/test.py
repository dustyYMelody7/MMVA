import configparser

conf = configparser.ConfigParser()
conf.read('config.cfg', encoding='utf-8')
conf.get('ocr', 'lable_map_dict')
print(conf.get('ocr', 'lable_map_dict'))

