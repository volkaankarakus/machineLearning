# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 16:43:36 2021

@author: VolkanKarakuş
"""

import pandas as pd

# import twitter data
data=pd.read_csv(r'gender_classifier.csv',encoding='latin1') #encoding='latin1' : csv file'imin icinde latin harfleri var demek.
    # basına r koymamizin sebebi read'in r'si.
data=pd.concat([data.gender,data.description],axis=1) # datamda bunlari kullanicam.

infoData=data.info()
descData=data.describe()
# nan value dropping
data.dropna(axis=0,inplace=True)

#gender'lar string oldugu icin, classification yapmak icin bunu int'e ya da categorical'a cevirmem gerek.
data.gender=[1 if each=='F' else 0 for each in data.gender]
# Female: 1
# Male  : 0

#%% cleaning data
# regular expression (RE) 

# string halinde bulunan ozel textleri barindirir.bir pattern'i search etmek icin kullanilir.
import re

first_description=data.description[4]
description=re.sub('[^a-zA-Z]',' ',first_description) # kucuk ve buyuk a'dan z'ye kadar olanlari bul. ^ isareti de bulma demek.
                                    # yani burada :) isareti gibi isaretleri boslukla degistir.

# tum harfleri kucuk harfe cevirelim.
description=description.lower()

#%% stopwords (irrelavent words) gereksiz kelimeler (the-and...)
import nltk # natural language toolkit

# nltk.download('stopwords') # corpus diye bir klasore indiriliyor.
from nltk.corpus import stopwords # sonra corpus klasorunden import ediyorum.

#description=description.split() # cumleyi kelimerine ayirir ve bir listede tutar.

#split yerine tokenizer kullanabiliriz.
description=nltk.word_tokenize(description) # splite gore avantaji , shouldn't gibi kisaltmalari da ayirir.

#%% 
# gereksiz kelimeleri cikar.
description=[ word for word in description if not word in set(stopwords.words('english'))]

#%% kelime koklerini bulalim.
#lemmatization
import nltk as nlp

lemma=nlp.WordNetLemmatizer()
description=[ lemma.lemmatize(word) for word in description ]

description=' '.join(description) # herbir kelimeyi boslukla birlestir.

#%% suana kadar 4. indexteki cumle icin yapmistik. komple butun dosyayi alalim.

descriptionList=[]

for description in data.description:
    description=re.sub('[^a-zA-Z]',' ',description)
    description=description.lower()
    description=nltk.word_tokenize(description) # cumleleri kelimelerine ayir.
    # description=[ word for word in description if not word in set(stopwords.words('english'))] # gereksiz kelimeleri cikar.
    lemma=nlp.WordNetLemmatizer() # kelimelerin koklerini buldum.
    description=[ lemma.lemmatize(word) for word in description ]
    description=' '.join(description) # kok halindeki kelimeleri cumle haline getir.
    descriptionList.append(description)
    
#%% bag of words
from sklearn.feature_extraction.text import CountVectorizer # bag of words kullanmak icin 
max_features=1500
    
count_vectorizer=CountVectorizer(max_features=max_features,stop_words='english') # token parameter da alir ama yukarida yapmistik.

sparce_matrix=count_vectorizer.fit_transform(descriptionList).toarray() # png'de 1ler ve 0'lardan olusan matrix.

print('En sik kullanilan {} kelimeler {}'.format(max_features,count_vectorizer.get_feature_names()))

#%% 
# artik train edebilicegimiz bir datamiz var.
# featurelarimizin oldugu data sparce_matrix= x 
# bize ne lazim? Labellarimizin oldugu data.yani y= gender o da data[0] indexi
y=data.iloc[:,0] # male or female classes
y=y.values # bunu numpy arraye cevirdik.
x=sparce_matrix

# train test split
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42) # 10% si test olsun.

#%% naive bayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()

nb.fit(x_train,y_train)

#%%prediction
y_pred=nb.predict(x_test)

print('Accuracy is : {}'.format(nb.score(y_pred.reshape(-1,1),y_test)))
