# -*- coding: utf-8 -*- 

import Preprocessing
import Make_model
import fileupload

#file_name = ['Article_경제','Article_사회','Article_생활문화','Article_세계','Article_정치','Article_IT과학']
# file_name = 'uploadFile'
  
#for name in file_name:
print("file upload")
fileupload.fileUpload()
print("업로드파일 db에 insert완료!")
 
#형태소 분석
Preprocessing.file_to_ids()
#Preprocessing.file_to_ids()
#형태소 분석 후 DB에 저장
Preprocessing.save()
print("형태소 분석완료!")

#   
# fileupload.DBInsert()
# print("DB INSERT 완료!")
#   
#  #Preprocessing.combine()
#   

#훈련시작!
Make_model.Model()

# fileupload.DBInsert()
# print("DB INSERT 완료!")