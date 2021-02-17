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

#print(sigmoid(0))

#%% forward-backward propagation
def forward_backward_propagation(w,b,x_train,y_train):
    'forward propagation'
    z = np.dot(w.T,x_train)+b # transpose alinmasi carpilabilsin diye
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost=(np.sum(loss))/x_train.shape[1]  # x_train.shape[1] is for scaling(normalization)
    
    'backward propagation'
    derivative_weight=(np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias=np.sum(y_head-y_train)/x_train.shape[1]
    gradients={'derivative_weight': derivative_weight,'derivative_bias': derivative_bias}
    
    return cost,gradients

#%% Updating(learning) parameters
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion): # learning rate, how fast it learned
    cost_list = []
    cost_list2 = []
    index = []
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        
        if i % 10 == 0: #Let's show the cost value every 10 steps.for visuals only
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list
#parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate = 0.009,number_of_iterarion = 200)
 
#%% 
# prediction
def predict(w,b,x_test): # data to predict is x_test, not x_train !
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1])) # create a matrix of size y_test. x_test.shape[1] is 114.
    #The size of y_test is actually (114,1). Since the vector comparison is made (1.114) it does not cause any problems.

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction
# predict(parameters["weight"],parameters["bias"],x_test)

#%% logistic regression

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 4096
    w,b = initialize_w_and_b(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    # Print test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 3)
# Cost after iteration 0: 0.692977
# test accuracy: 78.0701754385965 %
# The graph turned out to be blank because I said store every 10 values ​​while plotting.

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 20)
# Cost after iteration 0: 0.692977
# Cost after iteration 10: 0.499667
# test accuracy: 94.73684210526316 %

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 500)
#Approaching zero at iteration 300. So let's do iteration = 300.
#%%
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 3, num_iterations = 300)
# learning_rate and num_iterations are hyperparameter. We can tune it. 3 and 300 are suitable values for it.

#%% sklearn with logistic regression 
from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(x_train.T,y_train.T) # x_train.shape : (30,455) -> 30 feature, 455 sample
# instead, I increase the number of features by transposing.

print('test accuracy {}'.format(lr.score(x_test.T,y_test.T))) # score: predict and tell directly how many percent you got right
# test accuracy 0.7271016126223551
