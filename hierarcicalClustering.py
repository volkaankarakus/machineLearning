# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 19:13:19 2021

@author: VolkanKarakuş
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% create dataset
# class 1
x1=np.random.normal(25,5,100) # normal demek Gaussian demek. 25 ortalamaya sahip, sigmasi 5 , 1000 tane deger uret.
y1=np.random.normal(25,5,100)

# class 2
x2=np.random.normal(55,5,100)
y2=np.random.normal(60,5,100)

# class 3
x3=np.random.normal(55,5,100)
y3=np.random.normal(15,5,100)

x=np.concatenate((x1,x2,x3),axis=0)
y=np.concatenate((y1,y2,y3),axis=0)

dictionary={'x':x,'y':y}
data=pd.DataFrame(dictionary)

dataInfo=data.info() # Data columns (total 2 columns):
                    #   Column  Non-Null Count  Dtype  
                    # ---  ------  --------------  -----  
                    #  0   x       3000 non-null   float64
                    #  1   y       3000 non-null   float64


dataDescribe=data.describe() # mean,std,min,max

plt.scatter(x1,y1,color='black')
plt.scatter(x2,y2,color='black')
plt.scatter(x3,y3,color='black')
plt.show()

#%% Dendogram
from scipy.cluster.hierarchy import linkage, dendrogram # linkage, dendogram cizdirmek icin hiearcical algoritma

merg=linkage(data,method='ward') # ward, cluster icindeki yayilimlari minimize eden algoritma.
dendrogram(merg,leaf_rotation=90)
plt.xlabel('data points')
plt.ylabel('euclidian distance')
plt.show()
#plot ettirdigimizde en uzun mesafeyi bulup, enine bir threshold cekince, en mantikli cluster secimi : 3

#%% Hierarcital Clustering
from sklearn.cluster import AgglomerativeClustering # AgglomerativeClustering: tumevarim. 

hc=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
cluster=hc.fit_predict(data) # fit, benim modelimi olusturuyordu. datami kullanarak hc'imi fit ediyor. prediction ederek 
                            # de cluster'larimi olustur.
                            
data['label']=cluster

plt.scatter(data.x[data.label==0],data.y[data.label==0],color='red') # label'a gore filtrelicem, x ve y eksenine gore cizdiricem.
plt.scatter(data.x[data.label==1],data.y[data.label==1],color='green') 
plt.scatter(data.x[data.label==2],data.y[data.label==2],color='blue') 
plt.show()

