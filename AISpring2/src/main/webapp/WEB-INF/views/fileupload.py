# -*- coding: utf-8 -*- 
import scipy.io
import csv
import pymysql
import pandas as pd


def fileUpload():
    conn = pymysql.connect(host='127.0.0.1', user='DAN', password='dudeks7052', db='AI', charset='utf8')
    curs = conn.cursor()
    conn.commit()
#     data = pd.read_csv('./after_preprocessing_data/after_prepro.csv')
    f = open('/Users/noyeongdan/Downloads/Server_AI/AISpring2/src/main/webapp/WEB-INF/views/uploadFile.csv','r')
    csvReader = csv.reader(f)
    sql = """delete from uploadfile"""
    curs.execute(sql)
    
    for row in csvReader:
        Category = (row[0])
        Title = (row[1])
        Content = (row[2])
        
        if Category != 'Category':
            sql = """insert into uploadfile (Category,Title,Content) values (%s,%s,%s)"""
            curs.execute(sql, (Category,Title,Content))
    
    #db의 변화 저장
    conn.commit()
    f.close()
    conn.close()
    
    
def fileUpload6():
    conn = pymysql.connect(host='127.0.0.1', user='DAN', password='dudeks7052', db='AI', charset='utf8')
    curs = conn.cursor()
    conn.commit()
#     data = pd.read_csv('./after_preprocessing_data/after_prepro.csv')
    f = open('/Users/noyeongdan/Downloads/Spring4/AISpring/src/main/webapp/WEB-INF/views/uploadFile.csv','r')
    csvReader = csv.reader(f)
    sql = """delete from uploadfile"""
    curs.execute(sql)
    
    for row in csvReader:
        Category = (row[0])
        Title = (row[1])
        Content = (row[2])
        
        if Category != 'Category':
            sql = """insert into uploadfile (Category,Title,Content) values (%s,%s,%s)"""
            curs.execute(sql, (Category,Title,Content))
    
    #db의 변화 저장
    conn.commit()
    f.close()
    conn.close()
def DBInsert6():
    conn = pymysql.connect(host='127.0.0.1', user='DAN', password='dudeks7052', db='AI', charset='utf8')
    curs = conn.cursor()
    conn.commit()
    
#     data = pd.read_csv('./after_preprocessing_data/after_prepro.csv')
    f = open('./after_preprocessing_data/after_prepro.csv','r')
    csvReader = csv.reader(f)
    
    sql = """delete from uploadfile"""
    curs.execute(sql)
#      
    for row in csvReader:
        Date = (row[0])
        Content = (row[1])
        Category = (row[2])
        
        if Date != 'Date':
            sql = """insert into uploadfile (Date,Content,Category) values (%s, %s,%s)"""
            curs.execute(sql, (Date,Content,Category))
    
    #db의 변화 저장
    conn.commit()
    f.close()
    conn.close()
    
    
def fileUpload4():
    conn = pymysql.connect(host='127.0.0.1', user='DAN', password='dudeks7052', db='AI', charset='utf8')
    curs = conn.cursor()
    conn.commit()
#     data = pd.read_csv('./after_preprocessing_data/after_prepro.csv')
    f = open('/Users/noyeongdan/Downloads/Server_AI/AISpring2/src/main/webapp/WEB-INF/views/4Category3.csv','r')
    csvReader = csv.reader(f)
    sql = """delete from uploadfile"""
    curs.execute(sql)
    
    for row in csvReader:
        Category = (row[0])
        Title = (row[1])
        Content = (row[2])
        
        if Category != 'Category':
            sql = """insert into uploadfile (Category,Title,Content) values (%s,%s,%s)"""
            curs.execute(sql, (Category,Title,Content))
    
    #db의 변화 저장
    conn.commit()
    f.close()
    conn.close()
    

