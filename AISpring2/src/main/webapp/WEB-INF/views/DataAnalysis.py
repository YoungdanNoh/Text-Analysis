#-*-coding: utf-8-*-

import csv
import pymysql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import string
from sqlalchemy import create_engine
from sklearn.neighbors import KNeighborsRegressor

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
 
### 매개변수 받기
### 업로드한 파일 이름 (장소는 임의로 지정)
parser = argparse.ArgumentParser()
parser.add_argument('X', type=str,
            help="fileName")
 
args = parser.parse_args()
     
X = args.X # upload file name


# skip row 1 so pandas can parse the data properly.
uploadFilePath='/Users/noyeongdan/Downloads/Spring4/AISpring/src/main/webapp/WEB-INF/views/'+X
    
#'/Users/hyunjin/Documents/dataAnalysis/airbnb/airbnb_listings.csv'  
 
connDB = pymysql.connect(host='127.0.0.1', user='root', password='dbwjd7052',db='AI', charset='utf8')
 
connDB.query("set character_set_connection=utf8;")
connDB.query("set character_set_server=utf8;")
connDB.query("set character_set_client=utf8;")
connDB.query("set character_set_results=utf8;")
connDB.query("set character_set_database=utf8;")

curs = connDB.cursor()

sql="delete from airbnb_result" #예측결과가 담긴 테이블
print("0") 
curs.execute(sql)
connDB.commit()

sql="delete from airbnb_test" #사용자가 올린 csv데이터 정규화 함. 
print("0.6")  
curs.execute(sql)
connDB.commit()

print("0.9") 
### 정규화 시작      
loans_2007 = pd.read_csv(uploadFilePath,header=0,thousands = ',', low_memory=False)
half_count = len(loans_2007) / 2
loans_2007 = loans_2007.dropna(thresh=half_count,axis=1) # Drop any column with more than 50% missing values
    
print(loans_2007.shape)
    
loans_2007.head()
    
null_counts = loans_2007.isnull().sum()
#print("Number of null values in each column:\n{}".format(null_counts))    
loans_2007_filtered=loans_2007.fillna(1)
    
#print(loans_2007_filtered.isnull().sum())
loans_2007_filtered.shape
    
object_columns_df = loans_2007_filtered.select_dtypes(include=['object'])
#print(object_columns_df.iloc[0])
    
#loans_2007_filtered['price'] = loans_2007_filtered['price'].str.lstrip('$').str.replace(',', '').astype('float')
    
int_columns_df = loans_2007_filtered.select_dtypes(include=['int'])
#print(int_columns_df.iloc[0])
    
loans_2007_filtered= loans_2007_filtered.astype('float')
#print("Data types and their frequency\n{}".format(loans_2007_filtered.dtypes.value_counts()))
#print(loans_2007_filtered.dtypes)

engine = create_engine("mysql+pymysql://root:"+"dbwjd7052"+"@localhost:3306/AI",encoding="utf8")

new_dataframe=pd.DataFrame(loans_2007_filtered)
#new_dataframe.to_csv(uploadFilePath,index=False) #정규화한 파일 저장

print("끝!!!!!!!!!!!!!1")

conn = engine.connect()   
new_df=new_dataframe.set_index("id") #index해줘야 자동으로 만든 index 삭제함
new_df.to_sql(name='airbnb_test', con=engine, if_exists='append')

print("끝!!!!!!!!!!!!!2")

conn.close()

### 정규화 완료

### 에어비앤비 가격 예측 시작
SQL = "select * from airbnb_assets" # 기본 data + 업로드 data

normalized_listings=pd.read_sql(SQL,connDB)

curs.close()
connDB.close()

print("***************")
print(normalized_listings.shape)
normalized_listings=normalized_listings.sample(frac=1,random_state=0) 

print("0")
train_num=round(normalized_listings.shape[0]*0.7)

print("1")
norm_train_df=normalized_listings.ix[:train_num]
print(normalized_listings.ix[:train_num])

print("2")
norm_test_df=normalized_listings.ix[train_num:]
print(normalized_listings.ix[train_num:])

print("***********88")

cols=['accommodates','bedrooms','bathrooms','beds']
# 
print("2.5")
knn=KNeighborsRegressor(algorithm='brute')
# 
print("3")
knn.fit(norm_train_df[cols], norm_train_df['price']) #모델제작

four_features_predictions=knn.predict(norm_test_df[cols]) #예측
# 
print("4")
four_features_mse =(four_features_predictions-norm_test_df['price'])**2
norm_test_df['predicted_price']=four_features_predictions #예측값 저장
norm_test_df['squared_error']=four_features_mse #오류값 저장

four_features_rmse=four_features_mse ** (1/2)

print('four_features_rmse',four_features_rmse)


chart_df=norm_test_df.sample(n=1000)

# engine = create_engine("mysql+pymysql://root:"+"shguswls12"+"@localhost:3306/SpringTest",encoding="utf8")
# print("이번엔 모델 예측값 저장")
# conn = engine.connect()   
# print("4.2")
# result_df=chart_df.set_index("id") #index해줘야 자동으로 만든 index 삭제함
# print("4.4")
# 
# result_df.to_sql(name='airbnb_chart', con=engine, if_exists='append')
# print("성공")
# 
# conn.close()

test_df=new_dataframe #예측할 데이터

test_four_features_predictions=knn.predict(test_df[cols])

print("5")

test_df['predicted_price']=test_four_features_predictions #예측값 저장

print("6")
print(test_df['predicted_price']) # 예측값
print(list(test_df))

engine = create_engine("mysql+pymysql://root:"+"dbwjd7052"+"@localhost:3306/AI",encoding="utf8")
print("이제 인서트하면 끝")
conn = engine.connect()   
print("7")
result_df=test_df.set_index("id") #index해줘야 자동으로 만든 index 삭제함
print("8")

result_df.to_sql(name='airbnb_result', con=engine, if_exists='append')
print("성공")

conn.close()

