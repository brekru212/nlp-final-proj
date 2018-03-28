# - composed of things i found on github/stackoverflow need to clean up
import os
import codecs
import json
import csv

import re
import numpy as np
import pandas as pd
import scipy
import glob


path = 'test_price_history/'
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    df['ticker'] = file_
    list_.append(df)
frame = pd.concat(list_)
frame['ticker'] = frame['ticker'].apply(lambda x: x.replace('price_history/', "").replace(".csv", ""))
print(frame.head())
frame = frame.sort_values(['ticker', 'Date'], ascending=[1, 1])
frame["return"] = (frame["Adj Close"].diff(1)/frame["Adj Close"].shift(1))*float(100)
frame["Before Market"] = float(100)*(frame.Open - frame.Close.shift(1))/frame.Close.shift(1)
frame["During Market"] = float(100)*(frame.Close - frame.Close.shift(1))/frame.Close.shift(1)
frame["After Market"] = float(100)*(frame.Open.shift(-1) - frame.Close)/frame.Close
frame['Date'] = pd.to_datetime(frame['Date'])
print(frame.head())

path = 'test_parsed-8K-gz/'
allFiles = glob.glob(path + "/*.csv")
data = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    df = df.rename(columns = {'date':'Date'})
    df['ticker'] = file_
    list_.append(df)
data = pd.concat(list_)

import datetime
test=data
test['timestamp'] = pd.to_datetime(test['timestamp'], format='%Y-%m-%d', errors='ignore')
test['Date'] = pd.to_datetime(test['Date'])

test['release_time_type'] = test['timestamp']
test['release_time_type'] = test.release_time_type.apply(lambda x: 1 if x < \
                                                         datetime.datetime(x.year, x.month, x.day,9,30,0) else x)
test['release_time_type'] = test.release_time_type.apply(lambda x: 2 if x not in [1] \
                                                         and x > datetime.datetime(x.year, x.month, x.day,9,30,0) \
                                                         and x <= datetime.datetime(x.year, x.month, x.day,16,0,0) \
                                                         else x)
test['release_time_type'] = test.release_time_type.apply(lambda x: 3 if x not in [1,2] \
                                                         and x > datetime.datetime(x.year, x.month, x.day,16,0,0) else x)

test['ticker']=test['ticker'].apply(lambda x: x.replace('parsed-8K-gz/', "").replace(".csv", ""))

print(test.head(3))

combined = pd.merge(test, frame, how='left', on=['ticker','Date'])

combined['stock_performance'] = 0
combined['stock_performance'] = \
  combined.release_time_type.apply(lambda x: 1 if x ==1 else 0)*combined['Before Market']\
+ combined.release_time_type.apply(lambda x: 1 if x ==2 else 0)*combined['During Market']\
+ combined.release_time_type.apply(lambda x: 1 if x ==3 else 0)*combined['After Market']

combined.dropna(axis=0, how='any')
print(combined.head(5))

# path = 'parsed-EPS/'
# allFiles = glob.glob(path + "/*.csv")
# print len(allFiles)
# frame = pd.DataFrame()
# list_ = []
# for file_ in allFiles:
#     df = pd.read_csv(file_,index_col=None, header=0)
#     list_.append(df)
# EPS = pd.concat(list_)
#
# #Rename symbol to ticker
# EPS = EPS.rename(columns = {'Symbol':'ticker'})
#
# EPS["Date"]=pd.to_datetime(EPS["Date"])
#
# combined_w_eps = pd.merge(EPS, combined, how='inner', on=['ticker','Date'])
# print(combined_w_eps.head(3))

# up = 0
# down = 1
# stay = 2
combined['label'] = combined.stock_performance
combined['label'] = combined.label.apply(lambda x: 'UP' if x >1 else x)
combined['label'] = combined.label.apply(lambda x: 'DOWN' if (isinstance(x, str) == 0 and x <-1) else x)
combined['label'] = combined.label.apply(lambda x: x if isinstance(x, str) else 'STAY')
combined.dropna(axis=0, how='any')
print(combined.head(5))

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score,f1_score

tfidf_vectorizer = TfidfVectorizer()

tfidf_vectorizer.fit(combined['text'])
X_train = np.array(tfidf_vectorizer.transform(combined['text']).todense())
y_train = combined['label']

kf = KFold(n_splits=10, shuffle=False)
# clf = MLPClassifier(hidden_layer_sizes=[5, 5])
clf = MultinomialNB()
pred = []
correct = []
fold = 0
for train_indices, test_indices in kf.split(X_train):
    fold += 1
    print fold
    # print(X_train[train_indices])
    # print 'x_trian\n'
    # print(y_train[train_indices])
    # print 'y_train\n'
    clf.fit(X_train[train_indices], y_train[train_indices])
    y_pred = clf.predict(X_train[test_indices])
    # pred += y_pred
    for yp in y_pred:
        pred.append(yp)
    for p in y_train[test_indices]:
        correct.append(p)
    # correct += y_train[test_indices]
    # print f1_score(y_train[test_indices], y_pred)

tp = 0
for i in range(0,len(pred)):
    if pred[i] == correct[i]:
        tp +=1
    else:
        print pred[i]
        print correct[i]
        print i
        print '\n'

print tp
print len(pred)


