#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install mglearn')


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


import mglearn
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o') 
plt.plot(X, -3 * np.ones(len(X)), 'o') 
plt.ylim(-3.1, 3.1)


# In[6]:


mglearn.plots.plot_knn_regression(n_neighbors=1)


# In[7]:


mglearn.plots.plot_knn_regression(n_neighbors=3)


# In[14]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_wave(n_samples=40)
# split the wave dataset into a training and a test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Instantiate the model, set the number of neighbors to consider to 3: 
reg = KNeighborsRegressor(n_neighbors=3) 
# Fit the model using the training data and training targets: 
reg.fit(X_train, y_train)


# In[15]:


reg.predict(X_test)


# In[17]:


reg.score(X_test,y_test)


# In[18]:


# APPLYING K NEAREST NEIGHBORS REGRESSION 


# In[19]:


fig,axes=plt.subplots(1,3,figsize=(15,4))
# CREATE 1000 DATA POINTS, EVENLY SPACED BETWEEN -3 TO 3
line=np.linspace(-3,3,1000).reshape(-1,1)
plt.suptitle("nearest neigbors regression")
for n_neighbors, ax in zip([1,3,9],axes):
    # MAKE PREDICTIONS USING 1,3,6 NEIGHBORS
    reg=KNeighborsRegressor(n_neighbors=n_neighbors).fit(X,y)
    ax.plot(X,y,'o')
    ax.plot(X,-3*np.ones(len(x)),'o')
    ax.plot(line,reg.predict(line))
    ax.set_title("%d neighbor(S)"% n_neighbors)


# In[ ]:




