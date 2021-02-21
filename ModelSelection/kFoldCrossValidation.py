# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 02:41:09 2021

@author: VolkanKarakuş
"""

# test datami gercek hayat problemi olarak alip, bunu tamamen ayiriyorum.
# train data'mi k degeri icin mesela k=3 icin 3 e ayirip:
    # 1. ve 2. kismi train alip 3'u test alip accuracy buluyorum.
    # 2. ve 3. kismi train, 2. test
    # 1. test, 2 ve 3 train olarak alıp bu uc accuracy'nin ortalamasini aliyorum.
# sonra test datasinda test ediyoruz.

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

#%%
iris=load_iris()
x=iris.data
y=iris.target

#%%
# normalization
x=(x-np.min(x))/(np.max(x)-np.min(x))

#%% train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#%% KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)

#%% K Fold Cross Validation (K=10)
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=knn, X=x_train, y=y_train, cv=10) # genelde 10'dur.

print('Average accuracy :{}'.format(np.mean(accuracies)))
print('Std of accuracies : {}'.format(np.std(accuracies)))
# Average accuracy :0.9445454545454547
# Std of accuracies : 0.073231208629623

#%% 
# simdi asil ayirdigimiz y_test ile test edelim.
knn.fit(x_train,y_train)
print('Test Accuracy : {}'.format(knn.score(x_test,y_test))) # Test Accuracy : 1.0

#%% grid search cross validation
# basta knn'de k degerini 3 secmistik.(komsuluk, train datasini ayiran k degil.)
# peki optimum k degeri baska birsey olamaz miydi ?
# bunu bilebilmek icin grid search cross validation yapmamiz gerek.
from sklearn.model_selection import GridSearchCV

grid={'n_neighbors':np.arange(1,50)}
knn=KNeighborsClassifier()

knn_cv=GridSearchCV(knn,grid,cv=10)
knn_cv.fit(x,y)

#print
print('Tuned hyperparameter of K : {}'.format(knn_cv.best_params_))
print('Best accuracy ( best score ) : {} , according to K: {}'.format(knn_cv.best_score_,knn_cv.best_params_))

# Tuned hyperparameter of K : {'n_neighbors': 13}
# Best accuracy ( best score ) : 0.9800000000000001 , according to K: {'n_neighbors': 13}


#%% grid search CV ile logistic regression deneyelim. (Ornek olsun diye.)
# elimde 3 tane class vardi bunu 2 ye dusurucem.
x=x[:100,:] # ilk 100 benim ilk 2 class'im.
y=y[:100]
from sklearn.linear_model import LogisticRegression

grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}  # l1 = lasso ve l2 = ridge

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,grid,cv = 10)
logreg_cv.fit(x,y)

print("tuned hyperparameters: (best parameters): ",logreg_cv.best_params_)
print("accuracy: ",logreg_cv.best_score_)


# {'C': 0.001, 'penalty': 'l2'}
# accuracy:  1.0













