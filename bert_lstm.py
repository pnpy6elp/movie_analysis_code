# historical three feature
from keras.preprocessing.text import Tokenizer
import codecs
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import nltk
from konlpy.tag import Okt
from tqdm import tqdm

# train set
name = "1" # definition number
filename1 = './new_data/definition'+name+'_spynorth_scaling_trust.txt'
filename2 = './new_data/definition'+name+'_spynorth_scaling_untrust.txt'
filename3 = './new_data/definition'+name+'_intistranger_scaling_trust.txt'
filename4 = './new_data/definition'+name+'_intistranger_scaling_untrust.txt'
filename5 = './new_data/definition'+name+'_assassin_scaling_trust.txt'
filename6 = './new_data/definition'+name+'_assassin_scaling_untrust.txt'
filename7 = './new_data/definition'+name+'_1987_scaling_trust.txt'
filename8 = './new_data/definition'+name+'_1987_scaling_untrust.txt'
filename9 = './new_data/definition'+name+'_taxi_scaling_trust.txt'
filename10 = './new_data/definition'+name+'_taxi_scaling_untrust.txt'
filename11 = './new_data/definition'+name+'_gongjo_scaling_trust.txt'
filename12 = './new_data/definition'+name+'_gongjo_scaling_untrust.txt'


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
with codecs.open(filename9, 'r', encoding='utf-8-sig') as f:
    lines11 = f.readlines()
with codecs.open(filename10, 'r', encoding='utf-8-sig') as f:
    lines12 = f.readlines()



# test set


with codecs.open('./new_data/definition'+name+'_spynorth_test_t.txt', 'r', 'utf-8-sig') as f:
    test1 = f.readlines()
with codecs.open('./new_data/definition'+name+'_spynorth_test_ut.txt', 'r', 'utf-8-sig') as f:
    test2 = f.readlines()
with codecs.open('./new_data/definition'+name+'_intistranger_test_t.txt', 'r', encoding='utf-8-sig') as f:
    test3 = f.readlines()
with codecs.open('./new_data/definition'+name+'_intistranger_test_ut.txt', 'r', encoding='utf-8-sig') as f:
    test4 = f.readlines()
with codecs.open('./new_data/definition'+name+'_assassin_test_t.txt', 'r', encoding='utf-8-sig') as f:
    test5 = f.readlines()
with codecs.open('./new_data/definition'+name+'_assassin_test_ut.txt', 'r', encoding='utf-8-sig') as f:
    test6 = f.readlines()
with codecs.open('./new_data/definition'+name+'_1987_test_t.txt', 'r', encoding='utf-8-sig') as f:
    test7 = f.readlines()
with codecs.open('./new_data/definition'+name+'_1987_test_ut.txt', 'r', encoding='utf-8-sig') as f:
    test8 = f.readlines()
with codecs.open('./new_data/definition'+name+'_taxi_test_t.txt', 'r', encoding='utf-8-sig') as f:
    test9 = f.readlines()
with codecs.open('./new_data/definition'+name+'_taxi_test_ut.txt', 'r', encoding='utf-8-sig') as f:
    test10 = f.readlines()
with codecs.open('./new_data/definition'+name+'_gongjo_test_t.txt', 'r', encoding='utf-8-sig') as f:
    test11 = f.readlines()
with codecs.open('./new_data/definition'+name+'_gongjo_test_ut.txt', 'r', encoding='utf-8-sig') as f:
    test12 = f.readlines()
    

# intimate stranger

from keras.layers import Input, Dense
from keras.models import Model
lines_t = lines3[:2560]
lines_ut = lines4[:2560]
test_t = test3[:640]
test_ut = test4[:640]



lines_ = []
test_lines_ = []

lines1_ = []
lines2_ = []
lines3_ = []
lines4_ = []
# current three feature
feature1 = [] # train
feature2 = [] # test
# historical three feature
feature3 = [] # train
feature4 = [] # test
# current text
current_text1 = [] # train
current_text2= [] # test
# historical text
history_text1 = [] # train
history_text2= [] # test

rating = []
sentiment = []
correlation = []
## current three feature
for line in lines_t:
    
    try:
        # line = line.replace("\n"," ")
        a = line.split(",")[:3]
        a = list(map(float, a))
        feature1.append(a) # current feature
    except:
        print(lines_t.index(line))

for line in lines_ut:
    line = line.replace("\n"," ")
    a = line.split(",")[:3]
    a = list(map(float, a))
    feature1.append(a)

# historical three feature

for line in lines_t:
    line = line.replace("\n"," ")
    a = line.split(",")[3:10]
    a = list(map(float, a))
    ravg = []
    savg = []
    cavg = []
    ravg.append(a[0])
    ravg.append(a[3])
    savg.append(a[1])
    savg.append(a[4])
    cavg.append(a[5])
    cavg.append(a[6])
    ll = []
    ll.append(ravg)
    ll.append(savg)
    ll.append(cavg)
    feature3.append(ll) # current feature

for line in lines_ut:
    line = line.replace("\n"," ")
    a = line.split(",")[3:10]
    a = list(map(float, a))
    ravg = []
    savg = []
    cavg = []
    ravg.append(a[0])
    ravg.append(a[3])
    savg.append(a[1])
    savg.append(a[4])
    cavg.append(a[5])
    cavg.append(a[6])
    ll = []
    ll.append(ravg)
    ll.append(savg)
    ll.append(cavg)
    feature3.append(ll)

# current text

for line in lines_t:
    a = line.split(",")[10]
    current_text1.append(a.strip()) # current feature

for line in lines_ut:
    a = line.split(",")[10]
    current_text1.append(a.strip())
    
# historical text

for line in lines_t:
    #print(lines_t.index(line))
    a = line.split(",")[11]
    history_text1.append(a.strip()) # current feature

for line in lines_ut:
    a = line.split(",")[11]
    history_text1.append(a.strip())
    
    
# test data
# current three
for line in test_t:
    a = line.split(",")[:3]
    a = list(map(float, a))
    feature2.append(a)
    
for line in test_ut:
    a = line.split(",")[:3]
    a = list(map(float, a))
    feature2.append(a)

    
# historical three
for line in test_t:
    a = line.split(",")[3:10]
    a = list(map(float, a))
    ravg = []
    savg = []
    cavg = []
    ravg.append(a[0])
    ravg.append(a[3])
    savg.append(a[1])
    savg.append(a[4])
    cavg.append(a[5])
    cavg.append(a[6])
    ll = []
    ll.append(ravg)
    ll.append(savg)
    ll.append(cavg)
    feature4.append(ll)
    
for line in test_ut:
    a = line.split(",")[3:10]
    a = line.split(",")[3:10]
    a = list(map(float, a))
    ravg = []
    savg = []
    cavg = []
    ravg.append(a[0])
    ravg.append(a[3])
    savg.append(a[1])
    savg.append(a[4])
    cavg.append(a[5])
    cavg.append(a[6])
    ll = []
    ll.append(ravg)
    ll.append(savg)
    ll.append(cavg)
    feature4.append(ll) 
    
# current text

for line in test_t:
    a = line.split(",")[10]
    current_text2.append(a.strip()) # current feature

for line in test_ut:
    a = line.split(",")[10]
    current_text2.append(a.strip())

# history text
for line in test_t:
    
    a = line.split(",")[11]
    history_text2.append(a.strip()) # current feature

for line in test_ut:
    a = line.split(",")[11]
    history_text2.append(a.strip())

train_labels = [] # train 데이터 label
test_labels = [] # test 데이터 label
for i in range(len(lines_t)):
    train_labels.append(0)
for j in range(len(lines_ut)):
    train_labels.append(1)
for i in range(len(test_t)):
    test_labels.append(0)
for j in range(len(test_ut)):
    test_labels.append(1)

train_feature = np.asarray(feature3) # feature를 벡터화
test_feature = np.asarray(feature4)
train_labels = np.asarray(train_labels)
test_labels = np.asarray(test_labels)   
train_labels = np.asarray(train_labels).astype('int32').reshape((-1,1)) # input이 두개라서 dimension을 바꿔줘야 함
test_labels = np.asarray(test_labels).astype('int32').reshape((-1,1))
import pandas as pd
train_df = pd.DataFrame(train, columns = ['text', 'label'])
test_df = pd.DataFrame(test, columns = ['text', 'label'])

X_train = train_df['text']
X_test = test_df['text']
y_train = train_df['label']
y_test = test_df["label"]

from sklearn.model_selection import train_test_split


x_train, x_val,x_feature,x_val_feature, y_train, y_val = train_test_split(X_train,train_feature, train_labels, test_size=0.1, 
                                                 random_state=1, shuffle=False)


#from keras import layers
#from keras.models import Model
# lstm model
#from keras import layers
#from keras.models import Model
# lstm model
with tf.device('/cpu:0'):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)
    #l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['sequence_output'])
    l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
    # l = tf.keras.layers.LSTM(32, name="lstm")(l)

    # feature 은 wide model..?
    # feature_input = tf.keras.layers.Input(shape=(1,3,),) # tf.keras.layer랑 keras.layer 섞어 쓰면 안됨..
    feature_input = tf.keras.layers.Input(shape=(3,2,),)
    #feature_output = tf.keras.layers.LSTM(32, return_sequences=True)(feature_input)
    #feature_output = tf.keras.layers.LSTM(32)(feature_output)
    #feature_output = tf.keras.layers.Dense(3,activation="relu")(feature_output)
    feature_output = tf.keras.layers.Dense(3,activation="relu")(feature_input)
    feature_output = tf.keras.layers.Flatten()(feature_output)
    

    concatenated = tf.keras.layers.concatenate([l, feature_output])
    concat_reshape = tf.keras.layers.Reshape((1,777))(concatenated) # reshape 2d to 3d
    concat_out = tf.keras.layers.LSTM(32, return_sequences=True)(concat_reshape)
    concat_out = tf.keras.layers.LSTM(32, return_sequences=True)(concat_out)
    concat_out = tf.keras.layers.LSTM(32)(concat_out)
    concat_out = tf.keras.layers.Dense(1, activation='sigmoid')(concat_out)

    model = tf.keras.models.Model([text_input, feature_input], concat_out)

    METRICS = [
          tf.keras.metrics.BinaryAccuracy(name='accuracy'),
          tf.keras.metrics.Precision(name='precision'),
          tf.keras.metrics.Recall(name='recall')
        ]

    model.compile(optimizer='adam',
     loss='binary_crossentropy',
     metrics=METRICS)
    model.fit([x_train,x_feature], y_train, epochs=10,batch_size=128, validation_data=([x_val,x_val_feature], y_val))
    # multi input 이면 꼭!!! validation도 []로 묶어주는 거 잊지 말기..
result2 = model.evaluate([X_test,test_feature], test_labels) # definition 1
print(result2)