# -*- coding: utf-8 -*- 
import pymysql
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('mysql+pymysql://DAN:dudeks7052@localhost/AI',convert_unicode=True)
conn = engine.connect()

data = pd.read_sql_table('after_prepro', conn) #db데이터를 csv파일처럼 읽어온다.

print(data[:5]) # 상위 5개 출력

print('a')
data = data.drop_duplicates(['Content'], keep=False) # 중복되는 모든 값 제거
print('b')

conn.commit()
conn.close()