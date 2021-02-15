# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 03:03:21 2021

@author: VolkanKarakuş
"""
#%% LINEAR REGRESSION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#variable explorerdan name'in uzerindeki indirme isaretine git.
#burdan excel'de hazirlanmis dosyayi sec.
# skip row'u 1 yap. daha sonra array sec. arraydan dataframe'e gecise bakalim. direk df de secebilirdik.
#col seperater'i da ; yaptik.
# df=pd.DataFrame(linear_regression_datasetcsv,columns=['deneyim','maas']) ya da
df=pd.read_csv('linear_regression_dataset.csv',sep=';')
plt.scatter(df.deneyim,df.maas)
plt.xlabel('deneyim')
plt.ylabel('maas')
plt.show()

#bu grafige line fit etmek istedigimiz zaman :
    # y=b0 + b1*x
    
    # b0:constant(bias)
    # b1:coefficient

# y=0'da 2500 degerini kesen bi line fit edersek -> b0=2500
  
#    /|          icin b1= a/b (egim)
#   / |
#  /  | a        maas= b0+b1*deneyim
# /___|          12500=2500 + b1*10 dersek ; b1= 1000,b0=2500
#   b            deneyim = 11 icin ; maas= b0 + b1*deneyim
#                                    maas= 2500+1000*11= 13500


#%% 
# gercek y degeri ile line fit dogrusundaki izdusumu(y_head) arasindaki fark : residual

# residual= y - y_head
# bu deger eksi cikabildigi icin bunun karesini alıp hepsini toplayalim.
# sum(residual^2)
# MSE=sum(residual^2)/n yapiyoruz.  n: sample sayisi          MEAN SQUARED ERROR
# buradaki bolme bir cesit scale etmek.

#%% sklearn Line Fit

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

#line fit
linear_reg.fit(x,y)
#prediction
b0=linear_reg.predict([[0]]) # ya da b0=linear_reg.intercept_ 
# b0: [[1663.89519747]]

b1=linear_reg.coef_ # b1: egim,slope
# b1: array([[1138.34819698]])

#visualize line
array=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1) # deneyim
plt.scatter(x,y)
plt.show()

y_head=linear_reg.predict(array)  #maas 
plt.plot(array,y_head,color='red')
