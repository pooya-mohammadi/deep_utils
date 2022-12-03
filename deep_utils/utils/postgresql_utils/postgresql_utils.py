from typing import Dict


class PostgresqlUtils:
    @staticmethod
    def connect(connection=None, username=None, password=None, host=None, port=None):
        import psycopg2
        if connection is None:
            connection = psycopg2.connect(
                database="postgres", user=username, password=password, host=host, port=port)
        return connection

    @staticmethod
    def create_db(db_name, connection=None, username=None, password=None, host=None, port=None):
        from psycopg2.errors import DuplicateDatabase
        connection = PostgresqlUtils.connect(connection, username, password, host, port)
        connection.autocommit = True
        # Creating a cursor object using the cursor() method
        cursor = connection.cursor()
        sql = f'''CREATE database {db_name}'''
        try:
            cursor.execute(sql)
            print("Database created successfully........")
        except DuplicateDatabase:
            print("Database is already created!")
        connection.close()

    @staticmethod
    def check_connection(connection=None, username=None, password=None, host=None, port=None,
                         status_key="POSTGRES_STATUS") -> Dict[str, str]:
        connection = PostgresqlUtils.connect(connection, username, password, host, port)
        # check postgres
        output = {}
        try:
            postgres_status = connection.status
            output[status_key] = "Alive" if postgres_status else "Down"
        except:
            output[status_key] = "Down"
        connection.close()
        return output



