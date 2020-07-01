# -*- coding: utf-8 -*- 
import MeCab, csv, os
import glob, pandas as pd, numpy as np
from sqlalchemy import create_engine
import pymysql
m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ko-dic')
print("여기1")
#os.chdir("/Users/noyeongdan/data")

category_, date_, content_ = [], [], []
#file_name = ['Article_경제','Article_사회','Article_생활문화','Article_세계','Article_정치','Article_IT과학']

'''def register_dic(f_name):
    files = glob.glob(root_dir+"/"+f_name+"/*.csv", recursive=True)
    for i in files:
        file_to_ids(f_name)'''


def file_to_ids():
    #tags for tokenizer
    '''tag_classes = ['NNG', 'NNP','VA', 'VV+EC', 'XSV+EP', 'XSV+EF', 'XSV+EC', 'VV+ETM', 
                   'MAG', 'MAJ', 'NP', 'NNBC', 'IC', 'XR', 'VA+EC']'''
                    #일반명사, 고유명사, 형용사, 동사+연결어미, 동사파생접미사+선어말어미, 동사파생접미사+종결어미, 동사파생접미사+연결어미,
                    #동사+관형형전성어미, 일반부사, 접속부사, 대명사, 단위를 나타내는 명사, 감탄사, 어근, 형용사+연결어미
    
    tag_classes = ['NNG', 'NNP','VA', 'VV+EC', 'VV+ETM', 'MAJ', 'XR', 'VA+EC']
    
    #데이터 읽어오고.
    engine = create_engine('mysql+pymysql://DAN:dudeks7052@localhost/AI',convert_unicode=True)
    conn = engine.connect()
    data = pd.read_sql_table('uploadfile', conn)
    
    #각각 분류
    title = data.iloc[:,1].values
    #date = data.iloc[:, 0].values
    content = data.iloc[:, 2].values
    category = data.iloc[:,0].values
    for cnt, value in enumerate(title):
        result = ''
#         print(str(title[cnt].encode('utf8')).strip())
#         print(str(content[cnt]).strip())
        
        #파이썬은 기본적으로 unicode를 사용한다. 따라서 한글을 사용하려면 unicode를 utf-8로 바꿔주어야 사용 가능하다.
        value = m.parseToNode(str(title[cnt].encode('utf8')).strip() + str(content[cnt].encode('utf8')).strip())
        while value:
            tag = value.feature.split(",")[0]
            word = value.feature.split(",")[3]
            if tag in tag_classes:
                if word == "*": value = value.next
                result += word.strip()+" "
            value = value.next
        content_.append(result)
        #date_.append(date[cnt])
        #category
        if category[cnt].encode('utf8')=='생활문화' : category_.append("0")
        if category[cnt].encode('utf8')=='세계' : category_.append("1")
        if category[cnt].encode('utf8')=='IT과학' : category_.append("2")
#file_to_ids()
def save():
    conn = pymysql.connect(host='127.0.0.1', user='DAN', password='dudeks7052', db='AI', charset='utf8')
    curs = conn.cursor()
    conn.commit()
    
    sql = """delete from after_prepro"""
    curs.execute(sql)
        
    for cnt, i in enumerate(content_):
        content__ = content_[cnt]
        category__ = category_[cnt]

        sql = """insert into after_prepro (Content,Category) values (%s,%s)"""
        curs.execute(sql, (content__,category__))
    
    #db의 변화 저장
    conn.commit()
    conn.close()
    
    
def file_to_ids6():
    #name = fname.split(".csv")[0]
    #name_array = name.split('/')
    #f_name = name_array[3] #파일이름
    
    #tags for tokenizer
    tag_classes = ['NNG', 'NNP','VA', 'VV+EC', 'XSV+EP', 'XSV+EF', 'XSV+EC', 'VV+ETM', 'MAG', 'MAJ', 'NP', 'NNBC', 'IC', 'XR', 'VA+EC']
    #데이터 읽어오고.
    engine = create_engine('mysql+pymysql://DAN:dudeks7052@localhost/AI',convert_unicode=True)
    conn = engine.connect()
    data = pd.read_sql_table('uploadfile', conn)
    #data = pd.read_csv('uploadFile.csv')
    
    #각각 분류
    title = data.iloc[:,1].values
    #date = data.iloc[:, 0].values
    content = data.iloc[:, 2].values
    category = data.iloc[:,0].values
    for cnt, value in enumerate(title):
        result = ''
#         print(str(title[cnt].encode('utf8')).strip())
#         print(str(content[cnt]).strip())
        
        #파이썬은 기본적으로 unicode를 사용한다. 따라서 한글을 사용하려면 unicode를 utf-8로 바꿔주어야 사용 가능하다.
        value = m.parseToNode(str(title[cnt].encode('utf8')).strip() + str(content[cnt].encode('utf8')).strip())
        while value:
            tag = value.feature.split(",")[0]
            word = value.feature.split(",")[3]
            if tag in tag_classes:
                if word == "*": value = value.next
                result += word.strip()+" "
            value = value.next
        content_.append(result)
        #date_.append(date[cnt])
        #category
        if category[cnt].encode('utf8')=='경제' : category_.append("0")
        if category[cnt].encode('utf8')=='사회' : category_.append("1")
        if category[cnt].encode('utf8')=='생활문화' : category_.append("2")
        if category[cnt].encode('utf8')=='세계' : category_.append("3")
        if category[cnt].encode('utf8')=='정치' : category_.append("4")
        if category[cnt].encode('utf8')=='IT과학' : category_.append("5")
#         if '경제' in fname : category_.append("0")
#         if '사회' in fname : category_.append("1")
#         if '생활문화' in fname : category_.append("2")
#         if '세계' in fname : category_.append("3")
#         if '정치' in fname : category_.append("4")
#         if 'IT과학' in fname : category_.append("5")

def save6():
    conn = pymysql.connect(host='127.0.0.1', user='DAN', password='dudeks7052', db='AI', charset='utf8')
    curs = conn.cursor()
    conn.commit()
    
    sql = """delete from after_prepro"""
    curs.execute(sql)
     
    print("여기1")
    for cnt, i in enumerate(content_):
        content__ = content_[cnt]
        category__ = category_[cnt]

        sql = """insert into after_prepro (Content,Category) values (%s,%s)"""
        curs.execute(sql, (content__,category__))
    
    #db의 변화 저장
    conn.commit()
    conn.close()
    
    
#     if not os.path.exists(file_path):
#         os.mkdir(file_path)
#     with open(file_path+"/"+'after_prepro.csv', 'w') as f:
#         writer = csv.writer(f)
#     
#         
#         for cnt, i in enumerate(content_):
#             if cnt==0:
#                 date__ = "Date"
#                 content__ = "Content"
#                 category__ = "Category"
#             else:
#                 date__ = date_[cnt]
#                 content__ = content_[cnt]
#                 category__ = category_[cnt]
#             writer.writerow((date__, content__, category__))
            
            
            
            
#             if f_name=='Article_경제' and category_[cnt]=='0':
#                 date__ = date_[cnt]
#                 content__ = content_[cnt]
#                 category__ = category_[cnt]
#                 writer.writerow((date__, content__, category__))
#             elif f_name=='Article_사회' and category_[cnt]=='1':
#                 date__ = date_[cnt]
#                 content__ = content_[cnt]
#                 category__ = category_[cnt]
#                 writer.writerow((date__, content__, category__))
#             elif f_name=='Article_생활문화' and category_[cnt]=='2':
#                 date__ = date_[cnt]
#                 content__ = content_[cnt]
#                 category__ = category_[cnt]
#                 writer.writerow((date__, content__, category__))
#             elif f_name=='Article_세계' and category_[cnt]=='3':
#                 date__ = date_[cnt]
#                 content__ = content_[cnt]
#                 category__ = category_[cnt]
#                 writer.writerow((date__, content__, category__))
#             elif f_name=='Article_정치' and category_[cnt]=='4':
#                 date__ = date_[cnt]
#                 content__ = content_[cnt]
#                 category__ = category_[cnt]
#                 writer.writerow((date__, content__, category__))
#             elif f_name=='Article_IT과학' and category_[cnt]=='5':
#                 date__ = date_[cnt]
#                 content__ = content_[cnt]
#                 category__ = category_[cnt]
#                 writer.writerow((date__, content__, category__))
         #for cnt, i in enumerate(content_):
         #   print(category_[cnt])
         #   if category_[cnt]==0:
         #       print('int')
         #   elif category_[cnt]=='0':
         #       print('string')



def combine():
    os.chdir("./data_after_preprocessing_data") # Csv가 있는 파일 위치 수정
    category = ['경제','사회','생활문화','세계','정치','IT과학']
    
    file_unity = open('after_prepro.csv', 'w')
    wcsv = csv.writer(file_unity)
    
    for category_element in category:
        file = open('Article_'+category_element+'_after_prepro.csv', 'r')
        line = csv.reader(file)
        print(file)
        try:
            for line_text in line:
                wcsv.writerow([line_text[0], line_text[1], line_text[2]])
        except:
            pass
    print("파일 합치기 완료!")

    
def file_to_ids4():
    #tags for tokenizer
    tag_classes = ['NNG', 'NNP','VA', 'VV+EC', 'XSV+EP', 'XSV+EF', 'XSV+EC', 'VV+ETM', 
                   'MAG', 'MAJ', 'NP', 'NNBC', 'IC', 'XR', 'VA+EC']
                    #일반명사, 고유명사, 형용사, 동사+연결어미, 동사파생접미사+선어말어미, 동사파생접미사+종결어미, 동사파생접미사+연결어미,
                    #동사+관형형전성어미, 일반부사, 접속부사, 대명사, 단위를 나타내는 명사, 감탄사, 어근, 형용사+연결어미
    #데이터 읽어오고.
    engine = create_engine('mysql+pymysql://DAN:dudeks7052@localhost/AI',convert_unicode=True)
    conn = engine.connect()
    data = pd.read_sql_table('uploadfile', conn)
    
    #각각 분류
    title = data.iloc[:,1].values
    #date = data.iloc[:, 0].values
    content = data.iloc[:, 2].values
    category = data.iloc[:,0].values
    for cnt, value in enumerate(title):
        result = ''
#         print(str(title[cnt].encode('utf8')).strip())
#         print(str(content[cnt]).strip())
        
        #파이썬은 기본적으로 unicode를 사용한다. 따라서 한글을 사용하려면 unicode를 utf-8로 바꿔주어야 사용 가능하다.
        value = m.parseToNode(str(title[cnt].encode('utf8')).strip() + str(content[cnt].encode('utf8')).strip())
        while value:
            tag = value.feature.split(",")[0]
            word = value.feature.split(",")[3]
            if tag in tag_classes:
                if word == "*": value = value.next
                result += word.strip()+" "
            value = value.next
        content_.append(result)
        #date_.append(date[cnt])
        #category
        if category[cnt].encode('utf8')=='경제' : category_.append("0")
        if category[cnt].encode('utf8')=='생활문화' : category_.append("1")
        if category[cnt].encode('utf8')=='세계' : category_.append("2")
        if category[cnt].encode('utf8')=='IT과학' : category_.append("3")

def save4():
    conn = pymysql.connect(host='127.0.0.1', user='DAN', password='dudeks7052', db='AI', charset='utf8')
    curs = conn.cursor()
    conn.commit()
    
    sql = """delete from after_prepro"""
    curs.execute(sql)
     
    print("여기1")
    for cnt, i in enumerate(content_):
        content__ = content_[cnt]
        category__ = category_[cnt]

        sql = """insert into after_prepro (Content,Category) values (%s,%s)"""
        curs.execute(sql, (content__,category__))
    
    #db의 변화 저장
    conn.commit()
    conn.close()