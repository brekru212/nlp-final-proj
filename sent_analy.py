import pandas as pd
import numpy as np
import json

df = pd.read_csv('labelled_articles.csv')
data = json.load(open('finance.json'))

def sent_analys(text):
    total = 0
    # print text
    words = text.split(' ')
    for i in words:
        # print i
        if i in data:
            total += data[i]
    # print total
    return total

df = df.dropna(axis=0, how='any')

print pd.isnull(df).any(axis=1)
df['sent_score'] = df['full_text'].apply(sent_analys)

df.to_csv('sent_score_articles.csv', index=None, encoding='utf-8')
