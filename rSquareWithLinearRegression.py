# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 19:46:52 2021

@author: VolkanKarakuş
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('linear_regression_dataset.csv',sep=';')
plt.scatter(df.deneyim,df.maas)
plt.xlabel('deneyim')
plt.ylabel('maas')
plt.show()

from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()

x=df.deneyim # type(x)= pandas.core.series.Series
y=df.maas 

#bunlari pandas'tan numpy'e ceviricez.
x=df.deneyim.values # numpy array'e cevirdik.
y=df.maas.values

# x.shape = (14,) cikti.
# Aslinda bu 14'e 1. Ama sklearn bunu anlamaz. (14,1) gormek ister.
x=df.deneyim.values.reshape(-1,1)
# x.shape=(14,1) oldu.
y=df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)
y_head=linear_reg.predict(x)
plt.plot(x,y_head,color='red')

#%% 
# fit ettigimiz modeli r_square ile performansina bakalim.
from sklearn.metrics import r2_score

print('r2_score :',r2_score(y,y_head)) # r2_score : 0.9775283164949902
