import streamlit as st

st.title("텍스트 데이터 기반 문서 분류 프로젝트")
st.markdown('---')

st.header("프로젝트 목표")
st.write("한국어 원문 데이터(법원 판결문)의 요약문을 카테고리('일반행정', '세무', '특허', '형사', '민사', '가사')별로 분류하는 프로젝트 수행")

st.header("데이터 출처")
st.write("https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=580")


st.header("프로젝트 개요")
st.write("이번 프로젝트에서는 LSTM 기술을 활용하여 법원 판결문을 분류하는 프로젝트를 수행합니다.") 
st.write("법원 판결문 데이터를 형태소 분석기를 활용하여 한국어 텍스트를 전처리하는 방법과 이를 학습하여 분류 성능을 확인합니다.")

# --- ch1 ---
st.subheader("1. 데이터 읽기")
st.write("pandas를 사용하여 `project_data_all3.json` 데이터를 읽고 dataframe 형태로 저장하겠습니다.")

code = '''
import pandas as pd
import streamlit as st

fp = './project_data_all3.json'
df = pd.read_json(fp)

# streamlit을 이용한 데이터프레임 출력
st.dataframe(df)
'''
st.code(code, language='python')


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf

fp = './project_data_all3.json'
df = pd.read_json(fp)
st.dataframe(df)

st.write("`category` 종류 확인 결과")
st.bar_chart(df['category'].value_counts())

df['category'] = df['category'].replace({'가사': 0, '형사': 1, '특허': 2, '민사': 3, '일반행정': 4, '세무': 5})


st.write("정답 레이블이 되는 `category`데이터를 `target`변수에 저장합니다.")
st.write("학습할 요약문 데이터 `abstractive`를 `data`변수에 저장합니다.")

code = '''
target = df['category'].values
data = df['abstractive'].values
'''
st.code(code, language='python')

target = df['category'].values
data = df['abstractive'].values

st.write("target 데이터의 개수를 확인 : " + str(len(target)))
st.write("abstractive 데이터의 개수를 확인 : " + str(len(data)))

# ---ch2---
st.subheader("2. 형태소 분석하기")
st.write("KoNLPy(\'코엔엘파이\'라고 읽습니다)는 한국어 정보처리를 위한 파이썬 패키지입니다.")
st.write("이번 프로젝트에서는 Okt(Twitter) 클래스를 활용하겠습니다.")

code = '''
# 판결요약문을 KoNLPy 의 Okt 클래스로 형태소 분석
from konlpy.tag import Okt

# Okt 객체 선언
okt = Okt()

# stemming기반 형태소 분석
# 먼저 요약문 1개만 품사 태깅을 해보겠습니다.
pos_results = okt.pos(data[0][0], norm=True, stem=True)

# 품사를 태깅한다는 것은 주어진 텍스트를 형태소 단위로 나누고 명사, 조사, 동사 등의 형태소를 배열 형태로 만다는 과정입니다.
print(pos_results)
'''
st.code(code, language='python')

from konlpy.tag import Okt

okt = Okt()

pos_results = okt.pos(data[0][0], norm=True, stem=True)

st.write("(▾를 누르면 결과를 축소할 수 있습니다)")
st.write(pos_results)


# 판결요약문 데이터를 형태소 분석 결과로 저장 
data_tokenized = []

# 학습데이터로 명사만 사용
for text in data:
    data_tokenized.append(okt.nouns(text[0]))

# 행태소 분석된 결과를 확인
st.write("판결요약문 데이터를 형태소 분석 결과로 저장하여 형태소 분석을 통해 명사만 추출 (개수 : "+str(len(data_tokenized))+")")
st.write("원본 문장 : " + str(data[0]))
st.write("(▾를 누르면 결과를 축소할 수 있습니다)")
st.write(data_tokenized[0])

st.write("`data_tokenized` 변수의 각 배열마다 몇개의 명사가 들어있는지 히스토그램으로 확인하면 대부분의 요약문이 20~60개의 명사를 가지고 있다는 것을 확인할 수 있습니다.")

from bokeh.plotting import figure

p = figure(
    x_axis_label = "length of samples",
    y_axis_label = "number of samples"
)
a = [len(s) for s in data_tokenized]

np_test = np.array(a)
unique, count = np.unique(np_test, return_counts=True)

p.quad(
    top=count[:-1], 
    bottom=0, 
    left=unique[:-1],
    right=unique[1:],
    fill_color="navy", 
    line_color="white", 
    alpha=0.5
)

st.bokeh_chart(p, use_container_width=True)

# ---ch3---
st.subheader("3. Keras 텍스트 전처리")
st.write("형태소 분석된 결과를 학습하기 위해서 Keras를 활용하겠습니다.")

code = '''
# Keras의 텍스트 전처리기를 이용하여 정수 인코딩
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

# fit_on_texts()안에 형태소 분석된 데이터를 입력으로 넣으면 빈도수를 기준으로 단어 집합을 생성
tokenizer.fit_on_texts(data_tokenized) 
'''

st.code(code, language='python')

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

# fit_on_texts()안에 형태소 분석된 데이터를 입력으로 넣으면 빈도수를 기준으로 단어 집합을 생성
tokenizer.fit_on_texts(data_tokenized) 

st.write("결과")
st.write("(▾를 누르면 결과를 축소할 수 있습니다)")
st.write(tokenizer.word_index)

st.write("실제로 단어의 빈도수를 확인하려면 `word_counts`를 보면 되고, '경마', '의향' 단어는 1번씩 사용된걸 확인할 수 있습니다.")
st.write("(▾를 누르면 결과를 축소할 수 있습니다)")

for key, val in tokenizer.word_counts.items() :
    if key == '경마' or key == '의향' :
        st.write(str(key) + " (빈도수 : " + str(val) + ")")

st.write('''
케라스 토크나이저에서는 숫자를 지정해서 빈도수가 높은 단어를 몇개까지 사용할지를 결정할 수 있습니다.
이번 프로젝트에서는 빈도수 상위 1000개의 단어를 사용한다고 토크나이저를 재정의하겠습니다.
''')
         
vocab_size = 1000
tokenizer = Tokenizer(num_words = vocab_size) 
tokenizer.fit_on_texts(data_tokenized)

code = '''
# 상위 1000개 단어만 학습에 사용

vocab_size = 1000
tokenizer = Tokenizer(num_words = vocab_size) 
tokenizer.fit_on_texts(data_tokenized)

data_index = tokenizer.texts_to_sequences(data_tokenized)
'''

data_index = tokenizer.texts_to_sequences(data_tokenized)

st.code(code, language='python')
st.write("(▾를 누르면 결과를 축소할 수 있습니다)")

st.write(data_index[0])

# ---ch3---
st.subheader("4. LSTM으로 판결 요약문 분류하기")

st.write('''
텍스트 분류를 LSTM을 통해서 수행하겠습니다.
먼저 `data_index`의 학습할 데이터를 학습데이터 80%, 테스트데이터 20% 비율로 나눠주겠습니다. 그리고 앞서 설명한바와 같이 각 카테고리의 비율을 유지하기 위하여 `stratify` 에 파라미터에 정답 레이블 데이터를 설정해줍니다.
''')

code = '''
# LSTM으로 판결요약문 분류하기
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

# class 비율(train:validation)에 유지하기 위해 stratify 옵션을 target으로 지정
X_train, X_test, y_train, y_test = train_test_split(data_index, target, test_size=0.2, stratify=target, random_state=100)
'''

st.code(code, language='python')