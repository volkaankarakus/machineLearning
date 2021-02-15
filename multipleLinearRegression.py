# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 00:03:51 2021

@author: VolkanKarakuş
"""

#%% MULTIPLE LINEAR REGRESSION

# maasi sadece deneyim degil, yas da etkilesin.
# Multiple Linear Regression : y=b0+b1*x1+b2*x2
#                           maas=b0+b1*deneyim+b2*yas

# maas: dependent variable
# deneyim,yas : independent variable
 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df=pd.read_csv('multiple_linear_regression_dataset.csv',sep=';')

x=df.iloc[:,[0,2]].values
y=df.maas.values.reshape(-1,1)

multiple_linear_regression=LinearRegression()
multiple_linear_regression.fit(x,y)

print('b0: ',multiple_linear_regression.intercept_) #  b0: [10376.62747228]
print('b1, b2:',multiple_linear_regression.coef_) # b1, b2: [[1525.50072054 -416.72218625]]

# iki calisan da 35 yillik calisan ama deneyimleri farkli olsun, maaslarina bakalim.
print(multiple_linear_regression.predict(np.array([[10,35],[15,35]])))
# [[11046.35815877]
#  [18673.86176145]] biri 11k, digeri 18k aliyormus.
