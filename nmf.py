# - composed of things i found on github/stackoverflow need to clean up
import os
import codecs
import json
import csv
import datetime
import calendar
import re
import numpy as np
import pandas as pd
import scipy
import glob
import json

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout,Activation, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
# - composed of things i found on github/stackoverflow need to clean up
import os
import codecs
import json
import csv
import datetime
import calendar
import re
import numpy as np
import pandas as pd
import scipy
import glob
import json

from collections import Counter

companies = {}
companies_100 = []
with open('CompanyIndustry.json') as json_file:
    data = json.load(json_file)
    for item in data:
        if len(companies_100) < 101:
            companies_100.append(item[u'ticker'])
        companies[item[u'ticker']] = item[u'industry']

articles = {}
with open('ArticleForIndustry.json') as json_file:
    data = json.load(json_file)
    for item in data:
        # if item[u'indDateKey'][0:3] == 'ALL':
        #     print item[u'indDateKey']
        #     print item[u'articles']
        # else:
        #     print item[u'indDateKey'][0:3]
        articles[item[u'indDateKey']] = item[u'articles']

allFiles = []
path = 'price_history/'
for c in companies_100:
    allFiles.append('price_history/' + str(c) + '.csv')
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
unigram = {}
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    df['ticker'] = file_

    list_.append(df)

# path = 'test_price_history/'
# allFiles = glob.glob(path + "/*.csv")
# frame = pd.DataFrame()
# list_ = []
# for file_ in allFiles:
#     df = pd.read_csv(file_,index_col=None, header=0)
#     df['ticker'] = file_
#     list_.append(df)
frame = pd.concat(list_)
frame['ticker'] = frame['ticker'].apply(lambda x: x.replace('price_history/', "").replace(".csv", ""))
# print(frame.head())
frame = frame.sort_values(['ticker', 'Date'], ascending=[1, 1])
frame["return"] = (frame["Adj Close"].diff(1)/frame["Adj Close"].shift(1))*float(100)
frame["Before Market"] = float(100)*(frame.Open - frame.Close.shift(1))/frame.Close.shift(1)
frame["During Market"] = float(100)*(frame.Close - frame.Close.shift(1))/frame.Close.shift(1)
frame["After Market"] = float(100)*(frame.Open.shift(-1) - frame.Close)/frame.Close
frame['Date'] = pd.to_datetime(frame['Date'])
# print(frame.head())

# path = 'test_parsed-8K-gz/'
# allFiles = glob.glob(path + "/*.csv")
# data = pd.DataFrame()
# list_ = []
# for file_ in allFiles:
#     df = pd.read_csv(file_,index_col=None, header=0)
#     df = df.rename(columns = {'date':'Date'})
#     df['ticker'] = file_
#     list_.append(df)
# data = pd.concat(list_)

path = 'parsed-8K-gz/'
allFiles = []
for c in companies_100:
    allFiles.append('parsed-8K-gz/' + str(c) + '.csv')
data = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    df = df.rename(columns = {'date':'Date'})
    # for index, row in df.iterrows():
    #     bow = row['bow']
    #     bow_dict = dict(eval(bow))
    #     for k,v in bow_dict.iteritems():
    #         unigram.setdefault(k, 0)
    #         unigram[k] += v
            # print k
            # print v
    df['ticker'] = file_
    list_.append(df)
# print len(unigram)
print len(list_)
data = pd.concat(list_)

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

combined['Change_Date'] = combined.Date.shift(1)

print 'Combined made'

print(combined.head(3))

def add_day(sourcedate,days):
    month = sourcedate.day + days
    year = sourcedate.year + month / 12
    month = month % 12 + 1
    day = min(sourcedate.day,calendar.monthrange(year,month)[1])
    return datetime.date(year,month,day)

first_date = datetime.datetime(2006,10,20)
end_date = datetime.datetime(2013,02,01)

def add_sent(ticker, date, Change_Date):
    # ticker = ticker.split(' copy')[0]
    # ticker = ticker[5:]
    # print type(date)
    date = date.to_pydatetime()
    Change_Date = Change_Date.to_pydatetime()
    if Change_Date < first_date or date < first_date:
        return 0
    if date > end_date or Change_Date > end_date:
        return 0
    # print ticker
    # print date
    diff = date - Change_Date
    # 11
    total_score = 0.0
    total_arts = 1.0
    while diff.days > 0:
        indus = companies[ticker]
        key = indus + Change_Date.strftime("%Y-%m-%d")
        # allkey = 'ALL' + Change_Date.strftime("%Y-%m-%d")
        Change_Date = Change_Date + datetime.timedelta(days=1)
        diff = date - Change_Date
        try:
        # print articles[allkey]
            arts = articles[key]
            total_arts += len(arts)
            for i in range(len(arts)):
                cur_art = arts[i]
                total_score += cur_art[u'score']
        except:
            total_score += 0
            pass

        #
    # "Information Technology2006-10-20"
    #
    # indus = companies[ticker]
    # end_date = date + datetime.timedelta(days=1)
    # print indus
    #
    # print key
    # print end_date
    # try:
    #     print articles[key]
    # except:
    #     return 0
    #
    # return 0
    # print diff.days

    # print Change_Date.strftime("%Y-%m-%d")
    # print total_score
    sent_score = (total_score * 1.0) / total_arts
    # print sent_score
    return sent_score




combined['sent'] = combined.apply(lambda row: add_sent(row['ticker'], row['Date'], row['Change_Date']), axis=1)
print 'sent'
combined['stock_performance'] = 0
combined['stock_performance'] = \
  combined.release_time_type.apply(lambda x: 1 if x ==1 else 0)*combined['Before Market']\
+ combined.release_time_type.apply(lambda x: 1 if x ==2 else 0)*combined['During Market']\
+ combined.release_time_type.apply(lambda x: 1 if x ==3 else 0)*combined['After Market']

combined.dropna(axis=0, how='any')
print 'stock performance'
# print(combined.head(5))

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
combined['label'] = combined.label.apply(lambda x: 1.0 if x >1 else x)
combined['label'] = combined.label.apply(lambda x: 2.0 if (isinstance(x, str) == 0 and x <-1) else x)
combined['label'] = combined.label.apply(lambda x: x if isinstance(x, str) else 0.0)
combined.dropna(axis=0, how='any')
# print(combined.head(5))
print 'labeled'
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score,f1_score

train_data = combined.loc[combined.Date < '2010-12-31', :]
# dev_data = combined.loc[(combined.Date >= '2009-01-01') & (combined.Date <= '2010-12-31'), :]
test_data = combined.loc[combined.Date >= '2011-01-01', :]

tfidf_vectorizer = TfidfVectorizer()

tfidf_vectorizer.fit(combined['text'])
combined = ''
train_vecs = np.array(tfidf_vectorizer.transform(train_data['text']).todense())
print train_vecs.shape
train_sent = train_data.sent.values
train_sent = train_sent.reshape(13412,1)
print train_sent.shape
X_train = np.hstack((train_vecs, train_sent))
y_train = train_data['label']
print 'done with the training'
test_vecs = np.array(tfidf_vectorizer.transform(test_data['text']).todense())
test_sent = test_data.sent.values
test_sent = test_sent.reshape(test_sent.shape[0],1)
print test_sent.shape
X_test= np.hstack((test_vecs, test_sent))
y_test = test_data['label']

# X_train = np.array(tfidf_vectorizer.transform(train_data['text']).todense())
# y_train = train_data['label']
#
# X_test= np.array(tfidf_vectorizer.transform(test_data['text']).todense())
# y_test = test_data['label']

kf = KFold(n_splits=10, shuffle=False)
# clf = MLPClassifier(hidden_layer_sizes=[5, 5])
# clf = MultinomialNB()
# clf = RandomForestClassifier(n_estimators=100)

pred = []
correct = []
fold = 0
print (X_train.shape)
print (X_test.shape)
# clf.fit(X_train,y_train)
# print(clf.score(X_test,y_test))

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model = Sequential()
model.add(Dense(64,input_shape=(91091,)))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(X_train, y_train,
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_split=0.2,
          callbacks=[history])

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



