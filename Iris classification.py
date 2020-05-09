#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from sklearn.datasets import load_iris 
iris = load_iris() 
iris.keys()


# In[4]:



print(iris['DESCR'][:193]+'\n...')


# In[8]:


iris['target_names']


# In[7]:


iris['feature_names']


# In[11]:


type(iris['data'])


# In[12]:


iris['data'].shape


# In[13]:


iris['data'][:5]


# In[14]:


type(iris['target'])


# In[15]:


iris['target'].shape


# In[16]:


iris['target']


# # meaning of the above numbers are 0=setosa, versicolor=1, Virginica=2 

# In[17]:


# MEASURING SUCCESS: TRAINING AND TESTING DATA


# In[18]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(iris['data'],iris['target'],random_state=0)


# # THE ABOVE FUNCTION SPILTS THE DATA INTO 75% TRAINING AND 25% TESTING.

# In[19]:


X_train.shape


# In[20]:


X_test.shape


# In[27]:


fig,ax=plt.subplots(3, 3,figsize=(15,15))
plt.suptitle("iris_pairplot")
for i in range(3):
    for j in range(3):
        ax[i,j].scatter(X_train[:,j],X_train[:,i+1],c=Y_train,s=60)
        ax[i,j].set_xticks(())
        ax[i,j].set_yticks(())
        if i==2:
            ax[i,j].set_xlabel(iris['feature_names'][j])
        if j==0:
            ax[i,j].set_ylabel(iris['feature_names'][i+1])
        if j>i:
            ax[i,j].set_visible(False)


# #  DATA POINTS ARE COLORED ACCORDING TO TO DIFFERENT SPECIES OF IRIS. 

# In[28]:


# BULIDING MODEL USING K NEAREST NEIGHBORS


# In[30]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)


# In[32]:


knn.fit(X_train,Y_train)


# # Imagine we found an iris in the wild with a sepal length of 5cm, a sepal width of 2.9cm, a petal length of 1cm and a petal width of 0.2cm. What species of iris would this be? We can put this data into a numpy array, again with the shape number of samples (one) times number of features (four):

# In[33]:


X_new=np.array([[5,2.9,1,0.2]])
X_new.shape


# In[34]:


prediction=knn.predict(X_new)
prediction


# In[35]:


iris['target_names'][prediction]


# In[36]:


Y_pred=knn.predict(X_test)
np.mean(Y_pred==Y_test)


# In[37]:


knn.score(X_test,Y_test)


# In[ ]:




