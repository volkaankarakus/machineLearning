# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:12:05 2021

@author: VolkanKarakuş
"""

#%% SVM (Support Vector Machine)
# best decision boundary algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('data.csv')

headData=data.head()
# malignant : M
# benign : B

data.drop(['Unnamed: 32','id'],axis=1,inplace=True)

M=data[data.diagnosis=='M']
B=data[data.diagnosis=='B']

info_M=M.info()

#scatter for visualiation 
#%% radius_mean - area_mean
plt.scatter(M.radius_mean,M.area_mean,color='red',label='M',alpha=0.5)
plt.scatter(B.radius_mean,B.area_mean,color='green',label='B',alpha=0.5)
plt.xlabel('radius_mean')
plt.ylabel('area_mean')
plt.legend()
plt.show()

#%% radius_mean - texture_mean
plt.scatter(M.radius_mean,M.texture_mean,color='red',label='M',alpha=0.5)
plt.scatter(B.radius_mean,B.texture_mean,color='green',label='B',alpha=0.5)
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
plt.legend()
plt.show()

#%% KNN
# K Nearest Neighbour
#   1.) Select the K value
#   2.) Find nearest data points to K
#   3.) Calculate how many classes of K are among the nearest neighbors
#   4.) Determine which class the point or data we tested belong to


# bir nokta sec. K=3 olsun. O noktaya en yakin 3 veriye bak. 2 iyi, 1 kotu mesela. O zaman o nokta iyi.
# noktaya en yakin uzaklik euclidian distance yani hipotenus teoremi.
# hipotenus bulurken x uzunlugu 500 (x2-x1), y uzunlugu 0.01(y2-y1) olsun. Yine normalize etmemiz gerek.

#%% KNN with sklearn
data.diagnosis=[1 if each=='M' else 0 for each in data.diagnosis]
y=data.diagnosis.values # classes
x_data=data.drop(['diagnosis'],axis=1) # features

#normalization
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%% train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=40)

#KNN model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3) # K=3
knn.fit(x_train,y_train)
prediction=knn.predict(x_test)

print('KNN score : {} for K= {}'.format(knn.score(x_test,y_test),3)) # KNN score : 0.9766081871345029 for K= 3.

# K is hyperparameter. The K value is found by trying.

# find k value
scoreList=[]
for each in range(1,100):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    scoreList.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,100),scoreList)
plt.xlabel('K values')
plt.ylabel('Accuracy')
plt.show()
# K=20 is suitable.

#%% SVM
from sklearn.svm import SVC
svm=SVC(random_state=20)
svm.fit(x_train,y_train)

# test
print('Accuracy of SVM algorithm : {}'.format(svm.score(x_test,y_test)))
# 0.9883040935672515
