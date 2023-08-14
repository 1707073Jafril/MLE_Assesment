#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 04:52:41 2023

@author: shbmsk
"""

import numpy as np
import pandas as pd
pd.set_option("display.precision", 2)
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

# importing and reading the .csv file

df = pd.read_csv('/home/shbmsk/JobAssesment/ResumeDataSet.csv')
print("The number of rows are", df.shape[0],"and the number of columns are", df.shape[1])
df.head()


length = df["Resume"].shape
length
# Checking the information of the dataframe(i.e the dataset)

df.info()

# Checking all the different unique values

df.nunique()

# Plotting the distribution of Categories as a Count Plot

plt.figure(figsize = (15,15))
sns.countplot(y = "Category", data = df)
df["Category"].value_counts()

# Plotting the distribution of Categories as a Pie Plot

plt.figure(figsize = (18,18))
Category = df['Category'].value_counts().reset_index()['Category']
Labels = df['Category'].value_counts().reset_index()['index']
plt.title("Categorywise Distribution", fontsize=20)
plt.pie(Category, labels = Labels, autopct = '%1.2f%%', shadow = True)
df["Category"].value_counts()*100/df.shape[0]

df["res_new"] = df["Resume"]
eval_res = df["res_new"].copy(deep=True)

import string
def rem_punc(s):
    punc = string.punctuation
    return [i for i in s if i not in punc]

def rem_sw(s):
    sw = set(STOPWORDS)
    return [i for i in s if i not in sw]

#eval_res = Resumes
j=0
i=0
l=[]
for i in range(length[0]):
    try:
        eval_res[i] = eval(eval_res[i]).decode()
    except:
        l.append(i)
        pass

df["res_new"] = eval_res
#df = df.drop(l,axis=0)
#print(df[30:40])
df = df.reset_index(drop = True)

df = df[["Category","res_new","Resume"]]
df['res_new'].replace('', np.nan, inplace=True)
df.dropna(subset=['res_new'], inplace=True)
df = df.reset_index(drop = True)
df.shape
df.head()

length = df["res_new"].shape
eval_res = df["res_new"].copy(deep=True)
df.shape

import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt')
import matplotlib.pyplot as plt
import collections
from wordcloud import WordCloud,STOPWORDS
for i in range(length[0]):
    eval_res[i] = " ".join(eval_res[i].split("\n"))
    token = rem_sw(nltk.word_tokenize(eval_res[i])) #Removing punctaution later since we need punctaution for sentence tokenization
    eval_res[i] = " ".join(token).lower()
eval_res_backup  = eval_res.copy(deep = True)

for i in range(length[0]):
    eval_res[i] = (eval_res[i].encode("ASCII","ignore")).decode() #encoding the text to ascii.
eval_res.shape

df["res_new"] = eval_res

df_cols = ["Category","res_new","Resume"]
df = df[df_cols]
df.head()

REGEX_SPACE = re.compile("[ ][ ]+")
REGEX_JUNK = re.compile("[^A-WX-wyz][xX][^A-WX-wyz]+[ ]*|[.\-\_][.\-\_]+")
REGEX_EMAIL = re.compile("[Xx]+[._]?[Xx]+.@.[Xx]+\.?[Xx]+")
REGEX_PNO = re.compile("[(][xX][xX][xX][)][xX][xX][xX][xX][xX][xX][xX]|[xX][xX][xX][xX][xX][xX][xX][xX][xX][xX]|[xX][xX][xX][\-][xX][xX][xX]+[-][xX][xX][xX]+")

df["newer_res"] = df["res_new"]
for i,j in enumerate(df.itertuples()):
    strin = re.sub(REGEX_PNO,"",j[3])
    strin = re.sub(REGEX_EMAIL,"",strin)
    strin = re.sub(REGEX_SPACE,"",strin)
    strin  =re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', strin)
    strin = re.sub(REGEX_JUNK, "" ,strin)
    df["newer_res"][i] = strin

df = df[["Category","newer_res","Resume"]]
df.head()

df = df[["Category","newer_res","Resume"]]
df['newer_res'].replace('', np.nan, inplace=True)
df.dropna(subset=['newer_res'], inplace=True)
df = df.reset_index(drop = True)
df.shape

df = df.drop(['Resume'],axis=1)
df.rename(columns={'newer_res':'Resume'},inplace=True)
resume_punc = df["Resume"].copy(deep  = True)
df.head()

import string
def rem_punc(s):
    punc = string.punctuation
    return [i for i in s if i not in punc]

#Remove punctaution for further processing
for ind,i in enumerate(df.itertuples()):
    token = nltk.word_tokenize(i[2])
    #print(token)
    df["Resume"][ind] = " ".join(rem_punc(token))

import string
from wordcloud import STOPWORDS
def rem_punc(s):
    punc = string.punctuation
    return [i for i in s if i not in punc]

def rem_sw(s):
    sw = set(STOPWORDS)
    return [i for i in s if i not in sw]

def preprocess(eval_res):
    try:
        eval_res = eval(eval_res).decode()
    except:
        pass
    eval_res = eval_res.encode("ASCII","ignore").decode()
    length = len(eval_res)
    eval_res = " ".join(eval_res.split("\n"))
    token = rem_sw(nltk.word_tokenize(eval_res)) #Removing punctaution later since we need punctaution for sentence tokenization
    eval_res = " ".join(token).lower()
    return eval_res

df.head()

from io import StringIO
col = ['Category', 'Resume']
df = df[col]
df = df[pd.notnull(df['Resume'])]
df.columns = ['Category', 'Resume']
df['category_id'] = df['Category'].factorize()[0]
category_id_df = df[['Category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Category']].values)

df.head()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1,2), stop_words='english')
features = tfidf.fit_transform(df.Resume).toarray()
labels = df.category_id
features.shape

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

x_train, x_test, y_train, y_test = train_test_split(df['Resume'], df['Category'],test_size = 0.3, random_state = 0)

print(x_test)

count_vect = CountVectorizer() # bag-of-ngrams model , based on frequency count
x_train_counts = count_vect.fit_transform(x_train)
#x_test_counts = count_vect.fit_transform(x_test)

 #passing the word:word count
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)




models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    DecisionTreeClassifier(random_state=0),
    XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic'),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]


CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

cv_df.groupby('model_name').accuracy.mean()
print(cv_df.groupby('model_name').accuracy.mean())


classifier = LinearSVC()
classifier.fit(x_train_tfidf, y_train)

import pickle
vec_file = 'vectorizer.pickle'
pickle.dump(count_vect, open(vec_file, 'wb'))

filename = 'finalized_model.sav'
pickle.dump(classifier, open(filename, 'wb'))
