#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
cancer.keys()


# In[7]:


print(cancer.data.shape)


# In[8]:


print(cancer.target_names)
np.bincount(cancer.target)


# In[9]:


cancer.feature_names


# # Applying KNN alogorithm on the Breast cancer dataset

# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(    cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy=[]
test_accuracy=[]
neighbors_setting=range(1,11)
for n_neighbors in neighbors_setting:
    #BUILDING MODEL
    clf=KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,y_train)
    #RECORD TRAINING SET ACCURACY
    training_accuracy.append(clf.score(X_train,y_train))
    #TEST ACCURACY
    test_accuracy.append(clf.score(X_test,y_test))
plt.plot(neighbors_setting,training_accuracy,label="training accuracy")
plt.plot(neighbors_setting, test_accuracy, label="test accuracy")
plt.legend()


# # Testing score of the simple KNN model
# 

# In[22]:


Y_pred=clf.predict(X_test)
np.mean(Y_pred==y_test)


# In[20]:


clf.score(X_test,y_test)

