# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 20:17:39 2021

@author: VolkanKarakuş
"""

# ismi logisticRegression olmasina ragmen bu bir Classification algoritmasidir.
# binary classification'dir.(0 ve 1) iki farkli label. Kedi ve kopek gibi.

# logistic regression, simple neural network'tur. (simple deep learning)

# logistic regression methodunda Computation Graph uzerinden gidelim.
# computation graphs are a nice way to think about mathematical expressions.
# it is like visualization of mathematical expressions.
# c=squareroot(a^2+b^2)

# amac benim modelimi test ya da train etmek.
# herhangi bir resmin ne oldugunu anlamaya calisicam. ister train ister test.

#%% train 

# train etmek : resmi kendi modelime uydurucam. bir sutunda 4096 pixel olsun
# aslinda bu, 4096 tane ayri feature demek.
# herbir pixeli bir weight ile carpiyorum. w1,w2,...w4096 ile.
# sonra bunlari topluyorum . px1*w1 + px2*w2 + ..........
# bir de elimde bir bias degeri var (b). Bunu da toplayinca :
   
    # z= b+ px1*w1 + px2*w2 + px3*w3.....
    # artik parametrelerim : weightler ve bias.
    # weights, coefficients of each pixels.
    
# simdi bu z degerini Sigmoid Function'a sokucaz.(activation function)
# bunun outputu 0'la 1 arasindadir. cikan degere gore de bir threshold degeri belirleyip, o degerin alti 0, ustu 1.
# sigmoid function kullanilmasinin sebebi turevinini alinmasi.
# turev alinabildigi icin w0,w1 gibi degerleri guncelleyebilicez.

# peki weights ve bias icin initial parametreler nelerdir? weights=0.01 ve bias=0

#%% FORWARD PROPAGATION(ilerleme)

# z=(w.T)x+b  ; x:pixel, w:weights , b:bias , T: Transpose
# z'yi Sigmoid Functiona sokunca sonuc olarak y_head elde ediyoruz.(probability)

# 0 tahmin edip, resmim 0'sa loos'um yok. loss=0
# 0 tahmin edip, sonuc 1 'se loss var. 
# loss function'un matematiksel tanimi : -(1-y) * log(1-y_head) - y*log(y_head)

# herbir islem icin loss functionlarin toplamı : cost function

#%% BACKWARD PROPAGATION

# initial default degerler icin cost yuksek cikar. Yuksek cost'a gore islem tekrar yapilir ve yeni w ve b ler bulunur.
# bu surece de backward propagation denir. Costtan geriye dogru.

# Backward Propagation yaparken kullanilacak method : Gradient Descent

# optimum w=2 olsun. bizim ilk sectigimiz w=5 olsun. ikinci secilecek w : w-slope (w=5 icin slope'u cikarip yeni w buluyoruz.) 
    # bu surekli yapilip optimum w icin minimum cost bulunur.
    # slope : turev alip denkleme degeri yazinca bulunuyordu.
    # minimum cost degerine yaklastikca slope azalir. (x'e paralel gibi dusun)
    
#%% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read csv
data=pd.read_csv('data.csv') # bu data kanser hucresinin iyi huylu mu kotu huylu mu oldugunu gosteren data.
    # M: melignant(kotu huylu)
    # B: binal(iyi huylu)
    
info=data.info()
data.drop(['Unnamed: 32','id'],axis=1,inplace=True) # bunlara ihtiyacim yok, dropladim.

# elimde M ve B var. Bunlari classify ederken 0 ve 1 vermek yeterli ve kolay.
# diagnosisin icindeki M ve B object oldugu icin classify'da bunu kullanamam. ya categorical ya da int olmasi lazim.
# 0 ve 1 e cevirelim.

data.diagnosis=[1 if each=='M' else 0 for each in data.diagnosis]
# M : kotu huylu (1)
# B : iyi huylu (0)

y=data.diagnosis.values # values, series'ten numpy array'e cevirir.
# geri kalan butun datalar benim x eksenim.
x_data=data.drop(['diagnosis'],axis=1)

#%% normalization
# tum featurelari normalize etmem gerekiyor. bir feature bir feature a asiri baskinlik yaratmasin diye.
    # cunku verilerden bazisi 250 bazisi 0.005
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
# (x-min(x))/(max(x)-min(x))

#%% TRAINING TEST SPLIT
# datalari logistic regressiondan gecirdikten sonra elimde bir Model olucak.
# benim elimde butun datalar var. bunlarin hangisini train,hangisini test yapicam.
    # bunun icin train-test split yapilir. 80% train, 20% test.
    
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_train=x_train.T
x_test=x_test.T
y_train=y_train.T
y_test=y_test.T

print('x_train shape :',x_train.shape)
print('x_test shape :',x_test.shape)
print('y_train shape :',y_train.shape)
print('y_test shape :',y_test.shape)
# x_train shape : (30, 455)
# x_test shape : (30, 114)
# y_train shape : (455,)
# y_test shape : (114,)

#%% PARAMETERS AND SIGMOID FUNCTION
#dimension = 30
def initialize_w_and_b(dimension):
    
    w=np.full((dimension,1),0.01) # dimension'a 1'lik 0.01'lerden olusan matris. zeros ve ones gibi.
    b=0.0
    
    return w,b

#w,b=initialize_w_and_b(30)

def sigmoid(z):
    
    y_head=1/(1+np.exp(-z))
              
    return y_head

#sigmoid(0)




    
 


