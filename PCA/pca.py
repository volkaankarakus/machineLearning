# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 19:57:40 2021

@author: VolkanKarakuş
"""

# principal component analysis PCA
# feature sayisini azaltmak icin kullanilir.

#dataset'i bu sefer sklearn'in icindeki datasetlerden birinden alicaz.
from sklearn.datasets import load_iris
import pandas as pd

#%% 
iris=load_iris()

data=iris.data 
feature_names=iris.feature_names
y=iris.target

df=pd.DataFrame(data,columns=feature_names)
df['classes']=y
x=data

#%% PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2,whiten=True) # 4 boyutlu datami(4 feature vardi) 2 boyuta dusurmek istiyorum.
                                    # whiten=True normalize eder.
pca.fit(x) # fit ederken y yok.target'la bir isimiz yok, ML algoritmasi egitmesi ya da prediction yok.
            # 4 boyutu 2 ye dusurerek fit et.

x_pca=pca.transform(x) # 4 boyuttan 2 ye dusurecek modeli yukarida elde ettik. Burada 4'ten 2'ye dusur.
 
print('variance ratio :',pca.explained_variance_ratio_)
# variance ratio : [0.92461872 0.05306648] ilk kisim principle component, digeri de second component 

# pca yaparken amacim varyansi korumak, bilgileri korumak
print('sum :',sum(pca.explained_variance_ratio_))
#sum : 0.9776852063187949 . Yuzde 97 oraninda datanin %97'sine sahibim. 3%'lik bir data kaybi yasadik.

#%% 2D

df['p1']=x_pca[:,0]
df['p2']=x_pca[:,1]

# 3 farkli cicek turu var.
color=['red','green','blue']

import matplotlib.pyplot as plt

for each in range(3):
    plt.scatter(df.p1[df.classes==each],df.p2[df.classes==each],color=color[each],label=iris.target_names[each])
    
plt.legend()
plt.xlabel('p1')
plt.ylabel('p2')
plt.show()
