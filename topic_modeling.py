"""
https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)
"""

import pandas as pd

import pickle
import numpy as np
import os
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import Normalizer, normalize
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, Birch, MeanShift
from sklearn.utils.extmath import randomized_svd
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, GridSearchCV

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.tag import StanfordNERTagger
from string import punctuation
from gensim.models.phrases import Phrases, Phraser
from gensim import corpora, models, similarities, matutils, models
import spacy

import seaborn as sns
# import matplotlib.pyplot as plt
#
st = StanfordNERTagger('stanford-ner-2018-02-27/classifiers/english.muc.7class.distsim.crf.ser.gz',
                       'stanford-ner-2018-02-27/stanford-ner.jar')

stops = stopwords.words('english')


def clean_text(raw_text, stop=True):
    raw_text = raw_text.replace('U.S', 'US')
    letters_only = re.sub('[^a-zA-Z]', ' ', raw_text)
    letters_only = ' '.join(letters_only.split())
    #     underscored = underscore_entities(letters_only)
    #     words = underscored[0].split()
    words = letters_only.split()

    if stop == True:
        meaningful_words = [w for w in words if not w in stops]
        return (" ".join(meaningful_words))
    else:
        return (" ".join(words))
df = pd.read_csv('parsed-reuters-financial-news-dataset-master.csv')
df['cleaned_article_text'] = df['text'].astype(str).apply(clean_text)
print 'done parsing'
# df.to_csv('news_articles_cleaned.csv', index=None, encoding='utf-8')
count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer.fit(df['cleaned_article_text'])
counts = count_vectorizer.transform(df['cleaned_article_text']).transpose()
corpus = matutils.Sparse2Corpus(counts)
id2word = dict((v, k) for k, v in count_vectorizer.vocabulary_.items())
print 'bout to lda'
# models.LDA
lda = models.LdaModel(corpus=corpus, num_topics=10, id2word=id2word)
# lda = models.LdaMulticore(corpus=corpus, num_topics=10, id2word=id2word, passes=1,workers=1)

lda.print_topics()


def stem_lem_text(s, type='Lancaster'):
    words = s.split()

    if type == 'Porter':
        choice = PorterStemmer()
        reformed = [choice.stem(word) for word in words]
    elif type == 'Snowball':
        choice = SnowballStemmer('english')
        reformed = [choice.stem(word) for word in words]
    elif type == 'Lemmatize':
        choice = WordNetLemmatizer()
        reformed = [choice.lemmatize(word) for word in words]
    else:
        choice = LancasterStemmer()
        reformed = [choice.stem(word) for word in words]

    reformed = " ".join(reformed)
    return reformed
print 'making new texts'
# df['text_lancaster'] = df['cleaned_article_text'].apply(stem_lem_text, type='Lancaster')
# df['text_porter'] = df['cleaned_article_text'].apply(stem_lem_text, type='Porter')
df['text_snowball'] = df['cleaned_article_text'].apply(stem_lem_text, type='Snowball')
# df['text_lemmatize'] = df['cleaned_article_text'].apply(stem_lem_text, type='Lemmatize')
print 'snowball: we are not them '
snowball_df = df.drop(['text','cleaned_article_text',], axis=1)
snowball = snowball_df['text_snowball'].tolist()
tf_idf = TfidfVectorizer(stop_words='english', ngram_range=(1, 4), min_df=25, max_df=0.98)
tf_idf_vecs = tf_idf.fit_transform(snowball)

print 'lsa'
lsa = TruncatedSVD(400, algorithm='arpack')
lsa_vecs = lsa.fit_transform(tf_idf_vecs)
lsa_vecs = Normalizer(copy=False).fit_transform(lsa_vecs)
feature_names = tf_idf.get_feature_names()
lsa_df = pd.DataFrame(lsa.components_.round(5), columns=feature_names)
print 'lsa done'
np.save('tf_idf_vecs', tf_idf_vecs)
np.save('lsa_vecs', lsa_vecs)
np.save('feature_names', feature_names)
lsa_df.to_pickle('lsa_df.pkl')
print 'kmeans time'
km = KMeans(n_clusters=150, init='k-means++')
km.fit(lsa_vecs)
clusters = km.predict(lsa_vecs)
km.cluster_centers_.shape
original_space_centroids = lsa.inverse_transform(km.cluster_centers_)
original_space_centroids.shape
order_centroids = original_space_centroids.argsort()[:, ::-1]
order_centroids.shape
for cluster in range(150):
    features = order_centroids[cluster,0:10]
    print('Cluster {}\n'.format(cluster))
    for feature in features:
        print(feature_names[feature])
    print('\n')

np.save('ordered_centroids', order_centroids)
km_150 = pd.DataFrame({'new_km_150':clusters})
df_km_150 = pd.concat([snowball_df, km_150], axis=1)
df_km_150.to_csv('km.csv', encoding='utf-8')
df_km_150.to_pickle('km.pkl')
km_150.to_csv('km_150_clusters.csv', encoding='utf-8')
km_150.to_pickle('km_150_clusters.pkl')