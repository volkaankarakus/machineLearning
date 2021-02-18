# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 22:01:09 2021

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

#%% random forest 
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,random_state=3) # 100 tane decision tree olsun.

rf.fit(x_train,y_train)
print('Random forest algorithm result :',rf.score(x_test,y_test)) # 0.9534883720930233


#%%
# bu 95%'in M'sini mi dogru bildim, yoksa B'sini mi ?
# confisuion matrix
    # bu bir degerlendirme oldugu icin metrik diye gecer
    
y_prediction=rf.predict(x_test)
y_true=y_test
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_true,y_prediction)

# cm visualization
import seaborn as sns
import matplotlib.pyplot as plt

f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor='red',fmt='.0f',ax=ax) # fmt, ondalikli kisim. ax=eksen
plt.xlabel('y_predicted')
plt.ylabel('y_true')
plt.show()