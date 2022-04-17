from keras.preprocessing.text import Tokenizer
import codecs
from keras.preprocessing.sequence import pad_sequences
import numpy as np


# train set
name = "1" # definition 1
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
    a = line.split(",")[:3]
    a = list(map(float, a))
    feature1.append(a)

for line in lines_ut:
    a = line.split(",")[:3]
    a = list(map(float, a))
    feature1.append(a)

for line in test_t:
    a = line.split(",")[:3]
    a = list(map(float, a))
    feature2.append(a)

for line in test_ut:
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
input_train = np.asarray(feature1)
input_test = np.asarray(feature2)

# SVM

from sklearn.svm import SVC

svc = SVC(C=10, gamma=0.1)
svc.fit(input_train, train_labels)
print(svc.score(input_test, test_labels))
