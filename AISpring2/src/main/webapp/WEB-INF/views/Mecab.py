# -*- coding: utf-8 -*- 
import MeCab, csv, os
import glob, pandas as pd, numpy as np
m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ko-dic')
#os.chdir("/Users/noyeongdan/data")

category_, date_, content_ = [], [], []
file_name = ['Article_경제','Article_사회','Article_생활문화','Article_세계','Article_정치','Article_IT과학']

'''def register_dic(f_name):
    files = glob.glob(root_dir+"/"+f_name+"/*.csv", recursive=True)
    for i in files:
        file_to_ids(f_name)'''


def file_to_ids(fname):
    #name = fname.split(".csv")[0]
    #name_array = name.split('/')
    #f_name = name_array[3] #파일이름
    
    #tags for tokenizer
    tag_classes = ['NNG', 'NNP','VA', 'VV+EC', 'XSV+EP', 'XSV+EF', 'XSV+EC', 'VV+ETM', 'MAG', 'MAJ', 'NP', 'NNBC', 'IC', 'XR', 'VA+EC']
    #데이터 읽어오고.
    data = pd.read_csv(fname+'.csv')
    #각각 분류
    title = data.iloc[:,3].values
    date = data.iloc[:, 0].values
    content = data.iloc[:, 4].values

    for cnt, value in enumerate(title):
        result = ''
        value = m.parseToNode(str(title[cnt]).strip() + str(content[cnt]).strip())
        while value:
            tag = value.feature.split(",")[0]
            word = value.feature.split(",")[3]
            if tag in tag_classes:
                if word == "*": value = value.next
                result += word.strip()+" "
            value = value.next
        content_.append(result)
        date_.append(date[cnt])
        #category
        if '경제' in fname : category_.append("0")
        if '사회' in fname : category_.append("1")
        if '생활문화' in fname : category_.append("2")
        if '세계' in fname : category_.append("3")
        if '정치' in fname : category_.append("4")
        if 'IT과학' in fname : category_.append("5")

def save(month, file_path, f_name):
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    with open(file_path+"/"+f_name+'_after_prepro.csv', 'a') as f:
        writer = csv.writer(f)
    
        for cnt, i in enumerate(content_):
            if f_name=='Article_경제' and category_[cnt]=='0':
                date__ = date_[cnt]
                content__ = content_[cnt]
                category__ = category_[cnt]
                writer.writerow((date__, content__, category__))
            elif f_name=='Article_사회' and category_[cnt]=='1':
                date__ = date_[cnt]
                content__ = content_[cnt]
                category__ = category_[cnt]
                writer.writerow((date__, content__, category__))
            elif f_name=='Article_생활문화' and category_[cnt]=='2':
                date__ = date_[cnt]
                content__ = content_[cnt]
                category__ = category_[cnt]
                writer.writerow((date__, content__, category__))
            elif f_name=='Article_세계' and category_[cnt]=='3':
                date__ = date_[cnt]
                content__ = content_[cnt]
                category__ = category_[cnt]
                writer.writerow((date__, content__, category__))
            elif f_name=='Article_정치' and category_[cnt]=='4':
                date__ = date_[cnt]
                content__ = content_[cnt]
                category__ = category_[cnt]
                writer.writerow((date__, content__, category__))
            elif f_name=='Article_IT과학' and category_[cnt]=='5':
                date__ = date_[cnt]
                content__ = content_[cnt]
                category__ = category_[cnt]
                writer.writerow((date__, content__, category__))
         #for cnt, i in enumerate(content_):
         #   print(category_[cnt])
         #   if category_[cnt]==0:
         #       print('int')
         #   elif category_[cnt]=='0':
         #       print('string')

for name in file_name:
    file_to_ids(name)
    #register_dic(name)
    save(name, './data_after_preprocessing_data', name)
print("형태소 분석완료!")

import os
import csv
 
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