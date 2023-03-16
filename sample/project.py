import streamlit as st

st.title("텍스트 데이터 기반 문서 분류 프로젝트")


st.header("프로젝트 목표")

st.text("한국어 원문 데이터(법원 판결문)의 요약문을 카테고리('일반행정', '세무', '특허', '형사', '민사', '가사')별로 분류하는 프로젝트 수행")

st.header("데이터 출처")
st.text("https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=580")


st.header("프로젝트 개요")
st.text("이번 프로젝트에서는 LSTM 기술을 활용하여 법원 판결문을 분류하는 프로젝트를 수행합니다.") 
st.text("법원 판결문 데이터를 형태소 분석기를 활용하여 한국어 텍스트를 전처리하는 방법과 이를 학습하여 분류 성능을 확인합니다.")

st.subheader("1. 데이터 읽기")
st.text("pandas를 사용하여 `project_data_all3.json` 데이터를 읽고 dataframe 형태로 저장해 봅시다.")


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 엘리스 환경에서 한글 폰트를 사용하기 위한 코드입니다.
font_dirs = ['/usr/share/fonts/truetype/nanum', ]
font_files = fm.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    fm.fontManager.addfont(font_file)
    
plt.rc('font', family='NanumBarunGothic') 

fp = './project_data_all3.json'


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.random.set_seed(0)


# json 파일 읽기
df = pd.read_json(fp)
st.dataframe(df)


st.text("먼저 카테고리 종류를 확인")
st.text(df.category.unique())


st.text("형사와 민사가 가장 많고 특허나 가사 요약문은 상대적으로 적다는 것을 확인할 수 있습니다. 그렇기 때문에 데이터를 학습하기 위하여 테스트셋을 분류할 때 비율을 유지해주는 것이 좋습니다. train_test_split 메서드에서 stratify 옵션으로 비율을 유지할 수 있습니다.\n카테고리를 정답 레이블로 활용하기 위하여 숫자데이터로 치환합니다.")


df['category'] = df['category'].replace({'가사': 0, '형사': 1, '특허': 2, '민사': 3, '일반행정': 4, '세무': 5})


st.text("정답 레이블이 되는 `category`데이터를 `target`변수에 저장합니다.")


target = df['category'].values

st.text("target 데이터의 개수를 확인 :" + str(len(target)))


data = df['abstractive'].values

st.text("abstractive 데이터의 개수를 확인 :" + str(len(data)))

# ## 2. 형태소 분석하기

# KoNLPy("코엔엘파이"라고 읽습니다)는 한국어 정보처리를 위한 파이썬 패키지입니다.
# 
# KoNLPy에는 형태소를 분석하고 품사를 태깅할 수 있는 여러개의 패키지를 제공합니다. 여러가지 품사 태거들의 비교는 https://konlpy.org/ko/latest/morph/ 에서 확인할 수 있습니다.
# 
# 이번 프로젝트에서는 Okt(Twitter) 클래스를 활용하겠습니다.
# 
# Stemming(어간 추출)은 어형이 변형된 단어로부터 접사 등을 제거하고 그 `단어의 어간을 분리`하는 것을 말합니다.[위키백과]
# 
# 형태소는 두 가지 종류가 있습니다. 각각 어간(stem)과 접사(affix)입니다.
# 
# 1) 어간(stem)
# : 단어의 의미를 담고 있는 단어의 핵심 부분.
# 
# 2) 접사(affix)
# : 단어에 추가적인 의미를 주는 부분.
# 
# stemming은 정해진 규칙만 보고 단어의 의미를 구분하기 때문에, 어간 추출 후에 나오는 결과 단어는 `사전에 존재하지 않는 단어`일 수도 있습니다.

# 판결요약문을 KoNLPy 의 Okt 클래스로 형태소 분석
from konlpy.tag import Okt

# Okt 객체 선언
okt = Okt()

# stemming기반 형태소 분석
# 먼저 요약문 1개만 품사 태깅을 해보겠습니다.
pos_results = okt.pos(data[0][0], norm=True, stem=True)

# 품사를 태깅한다는 것은 주어진 텍스트를 형태소 단위로 나누고 명사, 조사, 동사 등의 형태소를 배열 형태로 만다는 과정입니다.
print(pos_results)


# `data_tokenized` 변수에 모든 요약문을 형태소 분석하여 저장하겠습니다.
# 
# 형태소를 분석하는 메서드는 아래와 같습니다.
# 
# 1. okt.morphs()
# 
# - 텍스트를 형태소 단위로 나눈다. 옵션으로는 norm과 stem이 있다 
# 
# - norm은 normalize의 약자로 문장을 정규화하는 역할
# 
# - stem은 각 단어에서 어간을 추출하는 기능
# 
#  
# 
# 2. okt.nouns()
# 
# - 텍스트에서 명사만 뽑아낸다.
# 
#  
# 
# 3. okt.phrases()
# 
# - 텍스트에서 어절을 뽑아낸다.
# 
#  
# 
# 4. okt.pos()
# 
# - 각 품사를 태깅하는 역할. 
# 
# - 품사를 태깅한다는 것은 주어진 텍스트를 형태소 단위로 나누고, 나눠진 각 형태소를 그에 해당하는 품사와 함께 리스트화 하는 것을 의미한다. 
# 
#                                                                 ※ 출처(https://soyoung-new-challenge.tistory.com/31)
# 
# 이번 프로젝트에서는 명사만 추출하는 방식으로 진행하겠습니다.

# In[11]:


# 판결요약문 데이터를 형태소 분석 결과로 저장 
data_tokenized = []

# 학습데이터로 명사만 사용
for text in data:
    data_tokenized.append(okt.nouns(text[0]))

# 행태소 분석된 결과를 확인
print(data_tokenized[0])


# 위 결과를 보면 okt.pos의 결과에서 'Noun'으로 된 명사만 남아있는 것을 확인할 수 있습니다.

# In[12]:


# 형태소 분석 결과의 개수를 확인. 이전에 확인한 개수와 동일한 것을 확인할 수 있습니다.
print(len(data_tokenized))


# 각 문장마다 분리된 형태소, 즉 명사가 몇개씩인지 확인해보겠습니다.
# 
# `data_tokenized` 변수의 각 배열마다 몇개의 명사가 들어있는지 히스토그램으로 확인하면 대부분의 요약문이 20~60개의 명사를 가지고 있다는 것을 확인할 수 있습니다.

# In[13]:


print('판결 요약문의 최대 길이 :{}'.format(max(len(l) for l in data_tokenized)))
print('판결 요약문의 평균 길이 :{}'.format(sum(map(len, data_tokenized))/len(data_tokenized)))

plt.hist([len(s) for s in data_tokenized], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()


# ---

# ## 3. 케라스(Keras)의 텍스트 전처리

# 형태소 분석된 결과를 학습하기 위해서 Keras를 활용하겠습니다.
# 
# Keras는 기본적인 전처리 도구들을 제공하는데, `정수 인코딩`을 위해서 Keras의 토크나이저를 사용하겠습니다.
# 
# 여기서 `정수 인코딩`이란 컴퓨터가 텍스트보다는 숫자를 더 잘 처리할 수 있기 때문에, 자연어 처리 과정에서 텍스트를 숫자로 바꾸는 기법중에 하나입니다. 그리고 그러한 기법들을 본격적으로 적용시키기 위한 첫 단계로 각 단어를 고유한 정수에 맵핑(mapping)시키는 전처리 작업이 필요할 때가 있습니다.
# 
# 예를 들어 갖고 있는 텍스트에 단어가 5,000개가 있다면, 5,000개의 단어들 각각에 1번부터 5,000번까지 단어와 맵핑되는 고유한 정수, 다른 표현으로는 인덱스를 부여합니다. 가령, book은 150번, dog는 171번, love는 192번, books는 212번과 같이 숫자가 부여됩니다. 인덱스를 부여하는 방법은 여러 가지가 있을 수 있는데 랜덤으로 부여하기도 하지만, 보통은 전처리 또는 빈도수가 높은 단어들만 사용하기 위해서 단어에 대한 빈도수를 기준으로 정렬한 뒤에 부여합니다. [위키독스]

# In[14]:


# Keras의 텍스트 전처리기를 이용하여 정수 인코딩
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

# fit_on_texts()안에 형태소 분석된 데이터를 입력으로 넣으면 빈도수를 기준으로 단어 집합을 생성
tokenizer.fit_on_texts(data_tokenized) 


# fit_on_texts는 입력한 텍스트로부터 단어 `빈도수가 높은 순`으로 정수 인덱스를 부여합니다.
# 
# 각 단어에 인덱스가 어떻게 부여되었는지를 보려면, `word_index`를 확인하면 됩니다.

# In[16]:


# 각 단어에 부여된 인덱스 확인
print(tokenizer.word_index)


# 출력 결과를 보면, '경마', '의향'이란 단어의 인덱스가 가장 크기 때문에 가정 적은 빈도수를 가졌다고 생각할 수 있습니다.
# 
# 실제로 단어의 빈도수를 확인하려면 `word_counts`를 보면 되고, '경마', '의향' 단어는 1번씩 사용된걸 확인할 수 있습니다.

# In[17]:


# 각 단어의 사용 빈도수 확인
print(tokenizer.word_counts)


# 케라스 토크나이저에서는 숫자를 지정해서 빈도수가 높은 단어를 몇개까지 사용할지를 결정할 수 있습니다.
# 
# 이번 프로젝트에서는 빈도수 상위 1000개의 단어를 사용한다고 토크나이저를 재정의하겠습니다.

# In[18]:


# 상위 1000개 단어만 학습에 사용

vocab_size = 1000
tokenizer = Tokenizer(num_words = vocab_size) 
tokenizer.fit_on_texts(data_tokenized)


# 위에서 사용한 방법과 같이 `print(tokenizer.word_index)`로 정수 인덱스를 확인해보면 똑같이 7509개의 인덱스가 보입니다. 
# 
# 실제로 1000개의 단어가 적용되는건 `texts_to_sequences`를 사용할 때 적용됩니다.
# 
# `print(data_index[0])`의 결과를 보면 정수 인덱스가 1000을 넘지 않습니다.

# In[19]:


# 판결요약문 데이터를 인덱스로 변환
data_index = tokenizer.texts_to_sequences(data_tokenized)
print(data_index[0])


# ---

# ## 4. LSTM으로 판결 요약문 분류하기

# 텍스트 분류를 LSTM을 통해서 수행하겠습니다.
# 
# 먼저 `data_index`의 학습할 데이터를 학습데이터 80%, 테스트데이터 20% 비율로 나눠주겠습니다. 그리고 앞서 설명한바와 같이 각 카테고리의 비율을 유지하기 위하여 `stratify` 에 파라미터에 정답 레이블 데이터를 설정해줍니다.

# In[ ]:


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


# 학습할 데이터(판결 요약문)에 들어있는 단어의 개수는 모두 다릅니다. 
# 
# 앞서 히스토그램으로 확인해봤을때 각 단어의 평균이 43이기 때문에 단어의 개수를 40개로 동일하게 패딩하겠습니다.

# In[ ]:


# 훈련용 판결요약문과 테스트용 판결요약문의 길이, 즉 단어수를 40으로 일치
# 단어수가 40개보다 많으면 나머지는 제거하고 모자르면 0으로 채워짐
max_len = 40

X_train = pad_sequences(X_train, maxlen=max_len) # 훈련용 판결요약문 패딩
X_test = pad_sequences(X_test, maxlen=max_len) # 테스트용 판결요약문 패딩


# 훈련용 데이터와 테스트용 데이터를 `원-핫 인코딩` 하겠습니다.
# 
# `원-핫 인코딩`은 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식입니다.[위키독스]
# 
# 이번 실습에서는 카테고리('일반행정', '세무', '특허', '형사', '민사', '가사')의 개수가 6개이므로 벡터의 크기는 6이 됩니다.

# In[21]:


# 훈련용, 테스트용 판결요약문 데이터의 레이블을 원-핫 인코딩

y_train = to_categorical(y_train) # 훈련용 판결요약문 레이블의 원-핫 인코딩
y_test = to_categorical(y_test) # 테스트용 판결요약문 레이블의 원-핫 인코딩


# `Embedding()`은 최소 두 개의 인자를 받습니다. 
# 
# 첫번째 인자는 단어 집합의 크기, 즉 총 단어의 개수입니다.
# 
# 두번째 인자는 임베딩 벡터의 출력 차원, 즉 결과로서 나오는 임베딩 벡터의 크기입니다.
# 
# 결과적으로 아래의 코드는 120차원을 가지는 임베딩 벡터 1,000개를 생성합니다. 
# 
# 마지막으로 6개의 카테고리를 분류해야하므로, 출력층에서는 6개의 뉴런을 사용합니다. 활성화 함수로는 소프트맥스를 사용하여 6개의 확률분포를 만듭니다. 

# In[22]:


model = Sequential()
model.add(Embedding(1000, 120))
model.add(LSTM(120))
model.add(Dense(6, activation='softmax'))


# 검증 데이터 손실(val_loss)이 증가하면, 과적합 징후므로 검증 데이터 손실이 5회 증가하면 학습을 조기 종료(Early Stopping) 하겠습니다.
# 
# ModelCheckpoint를 사용하여 검증 데이터의 정확도(val_acc)가 이전보다 좋아질 경우에만 모델을 저장하겠습니다.
# 

# In[23]:


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


# 다중 클래스 분류(Multi-Class Classification) 문제이므로 손실 함수로는 categorical_crossentropy를 사용합니다.
# 
# categorical_crossentropy는 모델의 예측값과 실제값에 대해서 두 확률 분포 사이의 거리를 최소화하도록 훈련합니다.
# 

# In[24]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


# 이제 학습을 진행합니다. 

# In[25]:


history = model.fit(X_train, y_train, batch_size=128, epochs=30, callbacks=[es, mc], validation_data=(X_test, y_test))


# 마지막으로 검증 데이터에 대한 정확도가 가장 높았을 때 저장된 모델인 `best_model.h5`를 로드하여 성능을 평가합니다.

# In[26]:


loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))


# epoch마다 변화하는 훈련데이터와 검증데이터(테스트 데이터)의 손실을 시각화하겠습니다.
# 
# 검증데이터의 loss값을 확인하면 작아지다가 다시 증가지는게 보입니다. 이는 과적합이 발생했다고 유추할 수 있습니다.[위키독스]

# In[27]:


epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ---
