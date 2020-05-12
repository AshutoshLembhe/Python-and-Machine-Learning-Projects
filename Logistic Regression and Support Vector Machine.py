#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
# IGNORE THE WARNING


# In[7]:


from sklearn.linear_model import LogisticRegression 
from sklearn.svm import LinearSVC
X, y = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1, 2, figsize=(10, 3)) 
plt.suptitle("linear_classifiers")
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):    
    clf = model.fit(X, y)    
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)    
    ax.scatter(X[:, 0], X[:, 1], c=y, s=60, cmap=mglearn.cm2)    
    ax.set_title("%s" % clf.__class__.__name__)


# In[8]:


mglearn.plots.plot_linear_svc_regularization()


# In[12]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer=load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(    cancer.data, cancer.target, stratify=cancer.target, random_state=42) 
logisticregression = LogisticRegression().fit(X_train, y_train) 
print("training set score: %f" % logisticregression.score(X_train, y_train)) 
print("test set score: %f" % logisticregression.score(X_test, y_test))


# In[15]:


logisticregression100=LogisticRegression(C=100).fit(X_train,y_train)
print("training set score:%f"% logisticregression100.score(X_train,y_train))
print("test set score:%f"% logisticregression100.score(X_test,y_test))


# In[16]:


logisticregression001=LogisticRegression(C=0.01).fit(X_train,y_train)
print("training set score:%f"% logisticregression001.score(X_train,y_train))
print("test set score:%f"% logisticregression001.score(X_test,y_test))


# In[17]:


plt.plot(logisticregression.coef_.T, 'o', label="C=1") 
plt.plot(logisticregression100.coef_.T, 'o', label="C=100") 
plt.plot(logisticregression001.coef_.T, 'o', label="C=0.001") 
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.ylim(-5, 5) 
plt.legend()


# In[18]:


for C in [0.001,1,100]:
    lr_l1=LogisticRegression(C=C,penalty="l1").fit(X_train,y_train)
    print("training accurracy of l1 logreg with C=%f:%f" %(C,lr_l1.score(X_train,y_train)))
    print("test accuracy of l1 logreg with C=%f:%f" %(C,lr_l1.score(X_test,y_test)))
    plt.plot(lr_l1.coef_.T,'o',label="C=%f"%C)
plt.xticks(range(cancer.data.shape[1]),cancer.feature_names, rotation=90)
plt.ylim(-5,5)
plt.legend(loc=2)   


# In[ ]:




