#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[32]:


data = pd.read_csv('../train.csv')


# In[33]:


x = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values


# In[34]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler((-1,1))
x = sc.fit_transform(x)


# In[35]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.05)


# In[36]:


r_state = random.randint(0,round(random.random() * 1000))


# In[37]:


max_i = 0
max_acc = 0
for i in [j*10 for j in range(1,10)]:
 rf = RandomForestClassifier(max_depth = i, random_state=r_state)
 rf.fit(x_train, y_train)
 pred = rf.predict(x_test)
 cm = confusion_matrix(y_test,pred)
 correct_pred = sum([cm[k][k] for k in range(cm.shape[0])])
 acc = correct_pred/x_test.shape[0]
 print('Max Depth: {} \tAccuracy: {}'.format(i,acc))
 if(acc > max_acc):
  max_acc = acc
  max_i = i


# In[38]:


max_dep = 0
max_acc = 0
for i in [j for j in range(max_i-10,max_i+10)]:
 rf = RandomForestClassifier(max_depth = i, random_state=r_state)
 rf.fit(x_train , list(y_train))
 pred = rf.predict(x_test)
 cm = confusion_matrix(y_test,pred)
 correct_pred = sum([cm[k][k] for k in range(cm.shape[0])])
 acc = correct_pred/x_test.shape[0]
 print('Max Depth: {} \tAccuracy: {}'.format(i,acc))
 if(acc > max_acc):
  max_dep = i
  max_acc = acc


# In[39]:


print('Best Max Depth: {}'.format(max_dep))


# In[40]:


svc = SVC(kernel = 'linear')
svc.fit(x_train,y_train)
pred = svc.predict(x_test)
cm = confusion_matrix(y_test,pred)
correct_pred = sum([cm[k][k] for k in range(cm.shape[0])])
acc = correct_pred/x_test.shape[0]
print('SVM Linear Accuracy: {}'.format(acc))


# In[41]:


svc = SVC(kernel = 'rbf')
svc.fit(x_train,y_train)
pred = svc.predict(x_test)
cm = confusion_matrix(y_test,pred)
correct_pred = sum([cm[k][k] for k in range(cm.shape[0])])
acc = correct_pred/x_test.shape[0]
print('SVM Rbf Accuracy: {}'.format(acc))


# In[42]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(x_train, y_train)
pred = log.predict(x_test)
cm = confusion_matrix(y_test,pred)
correct_pred = sum([cm[k][k] for k in range(cm.shape[0])])
acc = correct_pred/x_test.shape[0]
print('Linear Model Accuracy: {}'.format(acc))

