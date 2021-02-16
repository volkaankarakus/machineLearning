# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 18:46:01 2021

@author: VolkanKarakuş
"""
#%%
# residual , line fit ettikten sonra residual=y-y_head
# square residual = residual^2

# SSR = sum square residual = sum ((y-y_head)^2)

#%% 
# deneyim =x, maas=y icin y_avg=12000 olsun.
# SST = sum square total = sum((y-y_avg)^2)

# r_square = 1-(SSR-SST)
# r_square degeri 1'e ne kadar yakinsa o kadar iyi demektir.

# r_square elde ettigimiz regression modelimizin performansini olcmeye yarayan bir metrik.
# regression modelimizin performansi yaptigi predictionlara bagli.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('decisionTreeRegressionDataset.csv',sep=';',header=None) # onceki csv dosyasini kullanalim.

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

#%% Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=100,random_state=42) # n_estimators kac tane tree kullanicam,2. parametre daha sonra ayrintili.
                                                            # random_state bir kez daha run ettigimde ayni randomlikta bu datayi bol demekti.
                                                            # niye datayi boluyorduk: 100 tane tree'ye sub_data olusturuyordu.
rf.fit(x,y)
print('7.8 seviyesinde fiyat :',rf.predict([[7.8]]))
y_head=rf.predict(x)

#%% r_square bir metriktir.
from sklearn.metrics import r2_score
print('r_score :',r2_score(y,y_head)) # r_score : 0.9798724794092587. 1'e yakin oldugu icin iyi.





