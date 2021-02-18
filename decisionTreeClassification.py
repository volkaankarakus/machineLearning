# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 21:04:05 2021

@author: VolkanKarakuş
"""

import pandas as pd
import numpy as np

data=pd.read_csv(('data.csv'))
data.drop(['Unnamed: 32','id'],axis=1,inplace=True)

data.diagnosis=([1 if each=='M' else 0 for each in data.diagnosis])

y=data.diagnosis.values
x_data=data.drop(['diagnosis'],axis=1)

#normalization
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#train-test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=40)

#%% decision tree Classification
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)

print('score:',dt.score(x_test,y_test)) # score: 0.9418604651162791
