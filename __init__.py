import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pymongo 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re

def get_list_pages(col_name, ip='34.209.242.27'):
    cli = pymongo.MongoClient(ip, 27016)
    wikidb = cli.wikipedia
    col_pages = wikidb.get_collection(col_name)
    cursor = col_pages.find()
    text_list = []
    for entry in list(cursor):
        text_list.append(entry)
        
    return text_list

ml_df = pd.DataFrame(get_list_pages('ml_col'))
bizsoft_df = pd.DataFrame(get_list_pages('bussof'))

bizy = lambda x: "Business_Software"
bizsoft_df['Category'] = bizsoft_df['Category'].map(bizy)

wiki_df = pd.concat([ml_df, bizsoft_df], axis=0)

def cleaner(text):
    text = re.sub('&#39;','',text).lower()
    text = re.sub('<br />','',text)
    text = re.sub('<.*>.*</.*>','', text)
    text = re.sub('\\ufeff', '', text)
    text = re.sub('[\d]','',text)
    text = re.sub('[^a-z ]','',text)
    #text = ' '.join(text.split())
    return text

wiki_df['text'] = wiki_df['text'].map(str)
wiki_df['text'] = wiki_df['text'].apply(cleaner)

wiki_df.set_index('page_id', inplace=True)

tfidf_vector = TfidfVectorizer(min_df=5, stop_words="english")

wiki_pages_matrix_spare = tfidf_vector.fit_transform(wiki_df['text'])
wiki_pages_df_tfd = pd.DataFrame(wiki_pages_matrix_spare.toarray(),
                                index=wiki_df.index,
                                columns=tfidf_vector.get_feature_names())
full_wiki_text_tfd_df = pd.concat([wiki_df['text'], wiki_pages_df_tfd], axis=1)

n_components = 100
SVD = TruncatedSVD(n_components)
component_names = ["component_"+str(i+1) for i in range(n_components)]

wiki_svd_matrix = SVD.fit_transform(wiki_pages_df_tfd)