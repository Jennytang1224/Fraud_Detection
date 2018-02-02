from bs4 import BeautifulSoup
import unicodedata
import re
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

def getDescription(data):
    data['description'].fillna(0, inplace=True)
    desc = []
    for description in data['description']:
        soup = BeautifulSoup(description, 'html.parser')
        text = soup.get_text().strip()
        text = unicodedata.normalize('NFKD', text)
        desc.append(text)
    corpus = des
    labels = df['fraud'].tolist()
    return corpus, labels

def vectorizer(corpus, labels):
    vectorizer = TfidfVectorizer(min_df=0.1, max_df = 0.9, sublinear_tf=True, use_idf=True,stop_words='english')
    corpus_tf_idf = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names()
    Xtr = corpus_tf_idf
    y = labels
    dfs = top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25)
    plot_tfidf_classfeats_h(dfs)
    return dfs

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()
    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs):
    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    fig = plt.figure(figsize=(5, 5), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()

def predict(list_of_words, df_norm, df_fraud):
    score_norm = df_norm[df_norm['feature'].isin(list_of_words)].sum()
    score_fraud = df_fraud[df_fraud['feature'].isin(list_of_words)].sum()
    if score_norm > score_fraud:
        return 'Normal event.'
    else:
        return 'Fraud.'

if __name__ == '__main__':
    dfs = top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25)
    plot_tfidf_classfeats_h(dfs)
