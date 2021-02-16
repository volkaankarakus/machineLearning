# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 02:46:07 2021

@author: VolkanKarakuş
"""

#%% RANDOM FOREST
# ensemble learning'in bir uyesi.

# ensemble learning : ayni anda pek cok algoritmayi kullanarak elde edilen bir model.
#   birden fazla ML algoritmasini kullanip ortalama alir.

# data'nin icinden n sayida sample sectim. -> sub_data elde ettim.
# sub_data,  farkli tree'lere giderek herbir tree'den farkli bir sonuc cikiyor.
# daha sonra tree'lerden cikan degerler toplanip ortalamasi alinip asilsonuc bulunuyor.

# body part classification
# stock price prediction 
# recommendation systems random forestla calisir.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('decisionTreeRegressionDataset.csv',sep=';',header=None) # onceki csv dosyasini kullanalim.

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

#%% Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=100,random_state=42) # n_estimators kac tane tree kullanicam,2. parametre daha sonra ayrintili.
                                                            # random_state 5,10 gibi birsey de olabilirdi. stabil degil.
rf.fit(x,y)

print('7.8 seviyesinde fiyat :',rf.predict([[7.8]]))

x_ = np.arange(min(x),max(x),0.001).reshape(-1,1)
y_head = rf.predict(x_)

plt.scatter(x,y,color='red')
plt.plot(x_,y_head,color='green') # 1. parametre yeni yarattigim tahmin etmek istediğim degerler,2.  tahmin sonuclarim
plt.xlabel('Trıbun level')
plt.ylabel('price')
plt.show() 
# ne kadar dogru tahmin yaptigimiza bakmiyoruz,100 tane decision tree 1 tane decision tree'den daha iyidir.
