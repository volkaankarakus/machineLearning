# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 01:28:09 2021

@author: VolkanKarakuş
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('decisionTreeRegressionDataset.csv',sep=';',header=None)

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

#%% decision tree regression
from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor()  # random state=0
tree_reg.fit(x,y)
y_head=tree_reg.predict(x)

#%% visualize
plt.scatter(x,y,color='red')
plt.plot(x,y_head,color='green')
plt.xlabel('Stadium Level')
plt.ylabel('Price')
plt.show()

#%%
#bizim istedigimiz sey boyle bir ayrim degildi, ortalama aldigi icin yesil linelar boyle cikti. Bunu duzeltelim.
x_=np.arange(min(x),max(x),0.001).reshape(-1,1)
y_head2=tree_reg.predict(x_)

plt.scatter(x,y,color='red')
plt.plot(x_,y_head2,color='green')
plt.xlabel('Stadium Level')
plt.ylabel('Price')
plt.show()                # split'ler arasinda hemen peak yapip belli bi bolgede ortalama alindigi icin sabit deger cikar.



