import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return conn


# https://www.sqlitetutorial.net/sqlite-python/insert/

def create_table(conn):
    create_table_sql = \
        """ CREATE TABLE IF NOT EXISTS prices (
                id PRIMARY KEY,
                date text,
                time text,
                price float
                ); """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def insert_price(conn, price):
    sql = """ INSERT INTO prices(date,time,price)
              VALUES(?,?,?) """
    cur = conn.cursor()
    cur.execute(sql, price)
    conn.commit()


if __name__ == '__main__':
    conn = create_connection("pythonsqlite.db")
    if conn is not None:
        create_table(conn)
        price = ('a', '1', 101)
        insert_price(conn, price)
    else:
        print("Error! cannot create the database connection.")
