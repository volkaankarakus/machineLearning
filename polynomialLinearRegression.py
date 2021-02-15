# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 00:23:13 2021

@author: VolkanKarakuş
"""

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('polynomialRegression.csv',sep=';')

x=df.araba_max_hiz.values.reshape(-1,1) # values, series'i numpy array'e ceviriyordu.
y=df.araba_fiyat.values.reshape(-1,1)
 
plt.scatter(x,y)
plt.xlabel('araba_max_hizi')
plt.ylabel('araba_fiyat')
plt.show()

#%% polynomial linear regression (polynomial olanlar x'ler, linear olanlar b'ler)
from sklearn.preprocessing import PolynomialFeatures

# polynomial linear regression : y= b0 + b1*x + b2*x^2 + .... bn*x^n
# benim elimde b0,b1,b2 gibi degerler var. x'i de araba_fiyat ya da araba_max_hiz alsam da x^2 elimde yok.
# x^2'yi tanimlamam gerekiyor.

poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(x) # araba fiyatimi 2. dereceden poly feature a cevir.
# x_poly'nin icindeki ilk sutun heryerde 1. cunku x^0=1


#%% fit
from sklearn.linear_model import LinearRegression

lr2=LinearRegression()
lr2.fit(x_poly,y)

#%%
y_head=lr2.predict(x_poly)
plt.plot(x,y_head,color='green',label='poly : n=2')
plt.legend()
plt.show()

#%%
poly_reg10=PolynomialFeatures(degree=10)
x_poly10=poly_reg10.fit_transform(x)
lr10=LinearRegression()
lr10.fit(x_poly10,y)

y_head10=lr10.predict(x_poly10)
plt.plot(x,y_head10,color='blue',label='poly : n=10')
plt.legend()
plt.show()
