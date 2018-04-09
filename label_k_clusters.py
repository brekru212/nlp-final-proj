import pandas as pd
import numpy as np
import json
import re


feature_names = np.load('feature_names.npy')
feature_names.shape
centroids = np.load('ordered_centroids.npy')
centroids.shape
features = centroids[0][:10]

df = pd.read_csv('km.csv')
cluster_labels = {}

cluster_labels[0] = 'Financials'
cluster_labels[1] = 'Financials'
cluster_labels[2] = 'Financials'
cluster_labels[3] = 'Energy'
cluster_labels[4] = 'Financials'
cluster_labels[5] = 'Consumer Discretionary'
cluster_labels[6] = 'Information Technology'
cluster_labels[7] = 'Financials'
cluster_labels[8] = 'Financials'
cluster_labels[9] = 'Financials'
cluster_labels[10] = 'Financials'
cluster_labels[11] = 'Financials'
cluster_labels[12] = 'Industrials'
cluster_labels[13] = 'Energy'
cluster_labels[14] = 'Financials'
cluster_labels[15] = 'Financials'
cluster_labels[16] = 'Consumer Discretionary'
cluster_labels[17] = 'Consumer Discretionary'
cluster_labels[18] = 'Consumer Staples'
cluster_labels[19] = 'Financials'
cluster_labels[20] = 'Financials'
cluster_labels[21] = 'Financials'
cluster_labels[22] = 'All'
cluster_labels[23] = 'Financials'
cluster_labels[24] = 'Information Technology'
cluster_labels[25] = 'Financials'
cluster_labels[26] = 'Financials'
cluster_labels[27] = 'Financials'
cluster_labels[28] = 'Consumer Discretionary'
cluster_labels[29] = 'Financials'
cluster_labels[30] = 'Consumer Discretionary'
cluster_labels[31] = 'Telecommunication Services'
cluster_labels[32] = 'Materials'
cluster_labels[33] = 'Consumer Staples'
cluster_labels[34] = 'Information Technology'
cluster_labels[35] = 'Consumer Discretionary'
cluster_labels[36] = 'Financials'
cluster_labels[37] = 'Consumer Discretionary'
cluster_labels[38] = 'Financials'
cluster_labels[39] = 'Energy'
cluster_labels[40] = 'Financials'
cluster_labels[41] = 'Consumer Discretionary'
cluster_labels[42] = 'Financials'
cluster_labels[43] = 'Financials'
cluster_labels[44] = 'Financials'
cluster_labels[45] = 'Industrials'
cluster_labels[46] = 'Financials'
cluster_labels[47] = 'Financials'
cluster_labels[48] = 'Financials'
cluster_labels[49] = 'Consumer Discretionary'
cluster_labels[50] = 'Financials'
cluster_labels[51] = 'Financials'
cluster_labels[52] = 'Financials'
cluster_labels[53] = 'Financials'
cluster_labels[54] = 'Financials'
cluster_labels[55] = 'Energy'
cluster_labels[56] = 'Utilities'
cluster_labels[57] = 'Financials'
cluster_labels[58] = 'Information Technology'
cluster_labels[59] = 'Information Technology'
cluster_labels[60] = 'Industrials'
cluster_labels[61] = 'Financials'
cluster_labels[62] = 'Financials'
cluster_labels[63] = 'Energy'
cluster_labels[64] = 'ALL'
cluster_labels[65] = 'Financials'
cluster_labels[66] = 'Information Technology'
cluster_labels[67] = 'Financials'
cluster_labels[68] = 'ALL'
cluster_labels[69] = 'ALL'
cluster_labels[70] = 'Financials'
cluster_labels[71] = 'ALL'
cluster_labels[72] = 'Consumer Discretionary'
cluster_labels[73] = 'Financials'
cluster_labels[74] = 'Information Technology'
cluster_labels[75] = 'Consumer Discretionary'
cluster_labels[76] = 'ALL'
cluster_labels[77] = 'Consumer Discretionary'
cluster_labels[78] = 'Financials'
cluster_labels[79] = 'ALL'
cluster_labels[80] = 'Telecommunication Services'
cluster_labels[81] = 'ALL'
cluster_labels[82] = 'Financials'
cluster_labels[83] = 'Information Technology'
cluster_labels[84] = 'Financials'
cluster_labels[85] = 'Financials'
cluster_labels[86] = 'Information Technology'
cluster_labels[87] = 'Industrials'
cluster_labels[88] = 'Energy'
cluster_labels[89] = 'Financials'
cluster_labels[90] = 'Consumer Discretionary'
cluster_labels[91] = 'Financials'
cluster_labels[92] = 'ALL'
cluster_labels[93] = 'Information Technology'
cluster_labels[94] = 'Financials'
cluster_labels[95] = 'Financials'
cluster_labels[96] = 'Financials'
cluster_labels[97] = 'Consumer Discretionary'
cluster_labels[98] = 'ALL'
cluster_labels[99] = 'Industrials'
cluster_labels[100] = 'Consumer Staples'
cluster_labels[101] = 'Consumer Staples'
cluster_labels[102] = 'Healthcare'
cluster_labels[103] = 'Financials'
cluster_labels[104] = 'Financials'
cluster_labels[105] = 'Consumer Discretional'
cluster_labels[106] = 'Financials'
cluster_labels[107] = 'ALL'
cluster_labels[108] = 'Financials'
cluster_labels[109] = 'Consumer Discretional'
cluster_labels[110] = 'Financials'
cluster_labels[111] = 'Financials'
cluster_labels[112] = 'ALL'
cluster_labels[113] = 'Financials'
cluster_labels[114] = 'ALL'
cluster_labels[115] = 'Financials'
cluster_labels[116] = 'Industrials'
cluster_labels[117] = 'Industrials'
cluster_labels[118] = 'ALL'
cluster_labels[119] = 'ALL'
cluster_labels[120] = 'Consumer Discretionary'
cluster_labels[121] = 'Consumer Discretionary'
cluster_labels[122] = 'Information Technology'
cluster_labels[123] = 'Materials'
cluster_labels[124] = 'ALL'
cluster_labels[125] = 'Financials'
cluster_labels[126] = 'ALL'
cluster_labels[127] = 'Financials'
cluster_labels[128] = 'Consumer Staples'
cluster_labels[129] = 'Financials'
cluster_labels[130] = 'Financials'
cluster_labels[131] = 'Telecommunication Services'
cluster_labels[132] = 'Industrials'
cluster_labels[133] = 'Consumer Discretionary'
cluster_labels[134] = 'Financials'
cluster_labels[135] = 'Telecommunication Services'
cluster_labels[136] = 'Financials'
cluster_labels[137] = 'Financials'
cluster_labels[138] = 'ALL'
cluster_labels[139] = 'Information Technology'
cluster_labels[140] = 'ALL'
cluster_labels[141] = 'ALL'
cluster_labels[142] = 'Financials'
cluster_labels[143] = 'Financials'
cluster_labels[144] = 'ALL'
cluster_labels[145] = 'Materials'
cluster_labels[146] = 'Industrials'
cluster_labels[147] = 'ALL'
cluster_labels[148] = 'ALL'
cluster_labels[149] = 'Financials'

from nltk.corpus import stopwords

stops = stopwords.words('english')


def add_labels(val):
    return cluster_labels[val]

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
print 'getting news'
full_articles = pd.read_csv('parsed-reuters-financial-news-dataset-master.csv')
print full_articles.shape
print df.shape
print 'done parsing'

# def sent_analys(text):
#     for
# combined = pd.merge(df, full_articles, on='title', how='inner')
df['full_text'] = full_articles['text']
df.full_text = df.full_text.astype(str)
print df.shape
df['label'] = df['new_km_150'].apply(add_labels)
print df.shape
# data = json.load(open('fiance.json'))
#
# df['sent']

df.to_csv('labelled_articles.csv', index=None, encoding='utf-8')







