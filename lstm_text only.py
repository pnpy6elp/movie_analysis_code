from keras.preprocessing.text import Tokenizer
import codecs
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk
from konlpy.tag import Okt
from tqdm import tqdm

# train set
# definition 1
name = "1"
filename1 = './new_data/definition' + name + '_spynorth_scaling_trust.txt'
filename2 = './new_data/definition' + name + '_spynorth_scaling_untrust.txt'
filename3 = './new_data/definition' + name + '_intistranger_scaling_trust.txt'
filename4 = './new_data/definition' + name + '_intistranger_scaling_untrust.txt'
filename5 = './new_data/definition' + name + '_assassin_scaling_trust.txt'
filename6 = './new_data/definition' + name + '_assassin_scaling_untrust.txt'
filename7 = './new_data/definition' + name + '_1987_scaling_trust.txt'
filename8 = './new_data/definition' + name + '_1987_scaling_untrust.txt'
filename9 = './new_data/definition' + name + '_taxi_scaling_trust.txt'
filename10 = './new_data/definition' + name + '_taxi_scaling_untrust.txt'

with codecs.open(filename1, 'r', encoding='utf-8-sig') as f:
    lines1 = f.readlines()
with codecs.open(filename2, 'r', encoding='utf-8-sig') as f:
    lines2 = f.readlines()
with codecs.open(filename3, 'r', encoding='utf-8-sig') as f:
    lines3 = f.readlines()
with codecs.open(filename4, 'r', encoding='utf-8-sig') as f:
    lines4 = f.readlines()
with codecs.open(filename5, 'r', encoding='utf-8-sig') as f:
    lines5 = f.readlines()
with codecs.open(filename6, 'r', encoding='utf-8-sig') as f:
    lines6 = f.readlines()
with codecs.open(filename7, 'r', encoding='utf-8-sig') as f:
    lines7 = f.readlines()
with codecs.open(filename8, 'r', encoding='utf-8-sig') as f:
    lines8 = f.readlines()
with codecs.open(filename9, 'r', encoding='utf-8-sig') as f:
    lines9 = f.readlines()
with codecs.open(filename10, 'r', encoding='utf-8-sig') as f:
    lines10 = f.readlines()

# test set


with codecs.open('./new_data/definition' + name + '_spynorth_test_t.txt', 'r', 'utf-8-sig') as f:
    test1 = f.readlines()
with codecs.open('./new_data/definition' + name + '_spynorth_test_ut.txt', 'r', 'utf-8-sig') as f:
    test2 = f.readlines()
with codecs.open('./new_data/definition' + name + '_intistranger_test_t.txt', 'r', encoding='utf-8-sig') as f:
    test3 = f.readlines()
with codecs.open('./new_data/definition' + name + '_intistranger_test_ut.txt', 'r', encoding='utf-8-sig') as f:
    test4 = f.readlines()
with codecs.open('./new_data/definition' + name + '_assassin_test_t.txt', 'r', encoding='utf-8-sig') as f:
    test5 = f.readlines()
with codecs.open('./new_data/definition' + name + '_assassin_test_ut.txt', 'r', encoding='utf-8-sig') as f:
    test6 = f.readlines()
with codecs.open('./new_data/definition' + name + '_1987_test_t.txt', 'r', encoding='utf-8-sig') as f:
    test7 = f.readlines()
with codecs.open('./new_data/definition' + name + '_1987_test_ut.txt', 'r', encoding='utf-8-sig') as f:
    test8 = f.readlines()
with codecs.open('./new_data/definition' + name + '_taxi_test_t.txt', 'r', encoding='utf-8-sig') as f:
    test9 = f.readlines()
with codecs.open('./new_data/definition' + name + '_taxi_test_ut.txt', 'r', encoding='utf-8-sig') as f:
    test10 = f.readlines()

from keras.layers import Input, Dense
from keras.models import Model

# The spy gone north
lines_t = lines1[:2560]
lines_ut = lines2[:2560]
test_t = test1[:640]
test_ut = test2[:640]

lines_ = []
test_lines_ = []

lines1_ = []
lines2_ = []
lines3_ = []
lines4_ = []
feature1 = []
feature2 = []
rating = []
sentiment = []
correlation = []

for line in lines_t:
    text = line.split(",")[3]
    lines_.append(text.strip())
    a = line.split(",")[:3]
    a = list(map(float, a))
    feature1.append(a)

for line in lines_ut:
    text = line.split(",")[3]
    lines_.append(text.strip())
    a = line.split(",")[:3]
    a = list(map(float, a))
    feature1.append(a)

for line in test_t:
    text = line.split(",")[3]
    test_lines_.append(text.strip())
    a = line.split(",")[:3]
    a = list(map(float, a))
    feature2.append(a)

for line in test_ut:
    text = line.split(",")[3]
    test_lines_.append(text.strip())
    a = line.split(",")[:3]
    a = list(map(float, a))
    feature2.append(a)

train_labels = []  # train 데이터 label
test_labels = []  # test 데이터 label
for i in range(len(lines_t)):
    train_labels.append(1)
for j in range(len(lines_ut)):
    train_labels.append(0)
for i in range(len(test_t)):
    test_labels.append(1)
for j in range(len(test_ut)):
    test_labels.append(0)

train_labels = np.asarray(train_labels)
test_labels = np.asarray(test_labels)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

okt = Okt()
x_train = []
x_test = []
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

# train data
for sentence in tqdm(lines_):
    tokenized_sentence = okt.morphs(sentence, stem=True)  # tokenize
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]  # remove stopwords
    x_train.append(stopwords_removed_sentence)

# test data
for sentence in tqdm(test_lines_):
    tokenized_sentence = okt.morphs(sentence, stem=True)  # tokenize
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]  # remove stopwords
    x_test.append(stopwords_removed_sentence)

maxlen = 1000
training_samples = 200  # the number of train sample
validation_samples = 10000  # the number of validation sample
max_words = 10000  # 데이터셋에서 가장 빈도 높은 10,000개의 단어만 사용합니다
# train 토큰화
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)

word_index = tokenizer.word_index
train = pad_sequences(sequences, maxlen=maxlen)

# test 토큰화
tokenizer2 = Tokenizer(num_words=max_words)
tokenizer2.fit_on_texts(x_test)
sequences2 = tokenizer2.texts_to_sequences(x_test)

word_index2 = tokenizer2.word_index
test = pad_sequences(sequences2, maxlen=maxlen)

from keras.preprocessing import sequence


max_features = 10000  # 특성으로 사용할 단어의 수

batch_size = 32
input_train = sequence.pad_sequences(train, maxlen=maxlen)
input_test = sequence.pad_sequences(test, maxlen=maxlen)

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(input_train, train_labels, test_size=0.2, shuffle=True,
                                                      stratify=train_labels, random_state=34)
# LSTM
from keras.layers import LSTM

from keras.models import Sequential
from keras.layers import Embedding

from keras.layers import Dense

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
# 'binary_crossentropy
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                    epochs=10,
                    batch_size=128
                    )

model.evaluate(input_test, test_labels)
