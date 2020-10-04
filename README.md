Spring과 Python을 사용한 텍스트 분석
===============================

* Mecab을 사용해 형태소분석을 하고 LSTM을 사용해 훈련을 하며 모델을 생성합니다.
* 훈련 결과를 Spring을 통해 시각적으로 제공합니다.

Version of this program
------------------------

* tf.__version__

  1.13.1
  
* import numpy
* numpy.version.version

  1.16.1
  
* keras.__version__

  2.3.1

* pip install tensorflow-gpu==2.0

주의사항
------
> BiLSTM층 추가 시 
>> model.add(Bidirectional(LSTM(60, return_sequences=True)))

> RNN층 추가 시
>> model.add(SimpleRNN(60, return_sequences=True))
