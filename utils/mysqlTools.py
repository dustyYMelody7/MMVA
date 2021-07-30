import pymysql


class MySqlHold:

    def __init__(self, host: str, user: str, password: str, database: str, port=3306):
        self.db = pymysql.connect(host=host, user=user, port=port, database=database, password=password)
        self.cursor = self.db.cursor()

    def execute_command(self, command: str):
        self.cursor.execute(command)
        self.cursor.connection.commit()

    def fetchall(self):
        result = self.cursor.fetchall()
        return result

    def search(self, table: str, column_name: str, value: str):
        command = f"select * from {table} where {column_name} like \'{value}\'"
        self.execute_command(command)
        return self.fetchall()

    def close(self):
        self.cursor.close()
        self.db.close()


def get_one_item(conf, table: str, column_name: str, value: str):
    mysql = MySqlHold(host=conf.get('mysql', 'host'),
                      user=conf.get('mysql', 'user'),
                      password=conf.get('mysql', 'passwd'),
                      database=conf.get('mysql', 'db'),
                      port=int(conf.get('mysql', 'port')))
    result = mysql.search(table=table, column_name=column_name, value=value)
    mysql.close()
    return result


def get_video_info(conf) -> dict:

    mysql = MySqlHold(host=conf.get('mysql', 'host'),
                      user=conf.get('mysql', 'user'),
                      password=conf.get('mysql', 'passwd'),
                      database=conf.get('mysql', 'db'),
                      port=int(conf.get('mysql', 'port')))
    command = f"select * from {conf.get('mysql', 'video_table')}"
    mysql.execute_command(command)
    video_list = mysql.fetchall()

    result = {}
    for i, id_, path in video_list:

        result.update({id_: path})
    mysql.close()

    return result

