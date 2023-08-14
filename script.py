#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 05:00:20 2023

@author: shbmsk
"""

import preprocess_model
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
count_vect = CountVectorizer()

df = pd.read_csv('/home/shbmsk/JobAssesment/ResumeDataSet.csv')
print("The number of rows are", df.shape[0],"and the number of columns are", df.shape[1])
df.head()



length = df["Resume"].shape
length

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
x_train, x_test, y_train, y_test = train_test_split(df['Resume'], df['Category'],test_size = 0.3, random_state = 0)
print(x_test)
count_vect = CountVectorizer() # bag-of-ngrams model , based on frequency count
x_train_counts = count_vect.fit_transform(x_train)


loaded_vectorizer = pickle.load(open('/home/shbmsk/JobAssesment/vectorizer.pickle', 'rb'))
loaded_model = pickle.load(open('/home/shbmsk/JobAssesment/finalized_model.sav', 'rb'))


from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

def convertPDFtoText(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    #codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)
    fp.close()
    device.close()
    string = retstr.getvalue()
    retstr.close()
    return string

path = "/home/shbmsk/JobAssesment/CV (1).pdf"
test_resume = convertPDFtoText(path)
print(test_resume)


print(loaded_model.predict(loaded_vectorizer.transform([test_resume])))

a = loaded_model.predict(loaded_vectorizer.transform([test_resume]))

import pandas as pd
data_cat = [[path , a]]
df_cat = pd.DataFrame(data_cat, columns=['File', 'Resume Categoty'])
df_cat

df_cat.to_csv('category.csv')



predicted = []
for i in x_test:
    predicted.append((loaded_model.predict(count_vect.transform([i])))[0])

res= pd.DataFrame(y_test)
res['predicted'] = predicted
res.head()

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
accuracy=accuracy_score(res.Category, res.predicted)
print("Accuracy from SVM:",accuracy)

from sklearn import metrics

print(metrics.classification_report(res.Category, res.predicted, target_names=df['Category'].unique()))