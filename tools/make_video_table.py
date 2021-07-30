import os
import uuid
import argparse
import configparser


from utils.mysqlTools import MySqlHold

conf = configparser.ConfigParser()
conf.read('config/config.cfg')

mysql = MySqlHold(host=conf.get('mysql', 'host'),
                 user=conf.get('mysql', 'user'),
                 password=conf.get('mysql', 'passwd'),
                 database=conf.get('mysql', 'db'),
                 port=int(conf.get('mysql', 'port')))
table1 = 'video_docker'
table2 = 'video'

mysql.execute_command(f"truncate {table1}")
mysql.execute_command(f"truncate {table2}")

def insert_id(video_abs_path: str, video_save_path=None):
    video_list = os.listdir(video_abs_path)

    for item in video_list:
        if 'back' in item or 'videocut' in item:
            continue
        video_path = os.path.join(video_abs_path, item)
        # video_save_path = os.path.join('/video', item) if video_save_path is None else os.path.join(video_save_path, item)
        # print(video_path)
        if os.path.isfile(video_path):
            # video_save_path = None
            if not video_path.endswith('mp4'):
                continue
            video_save_path = os.path.join('/video', video_path.split('/')[-2], item)
            video_id = uuid.uuid1()
            insert_command = f"insert into {table1} (video_id, video_path) value(\"{video_id}\", \"{video_save_path}\")"
            mysql.execute_command(insert_command)
            insert_command = f"insert into {table2} (video_id, video_path) value(\"{video_id}\", \"{video_path}\")"
            print(insert_command)
            mysql.execute_command(insert_command)
            print("[INFO]: Insert, ", item)
            # video_save_path = None
        else:
            insert_id(video_path, video_save_path)
    return None

if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument("--video_dir", type=str, default='./videos', help='video dir')

    args = parse.parse_args()

    insert_id(args.video_dir)

