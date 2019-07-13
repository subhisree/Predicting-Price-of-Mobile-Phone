#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_csv('C:/Users/Subhiksha/Downloads/train.csv')
test  = pd.read_csv('C:/Users/Subhiksha/Downloads/test.csv')


# In[3]:


train.head(2)


# In[4]:


desired_factors = ['battery_power','blue','clock_speed','dual_sim','fc','four_g','int_memory','m_dep','mobile_wt','n_cores','px_height','px_width','ram','sc_h','sc_w','talk_time','three_g','touch_screen','wifi']


# In[5]:


model = LogisticRegression()


# In[6]:


train_data = train[desired_factors]
test_data = test[desired_factors]
target = train.price_range
model.fit(train_data , target)
model.predict(test_data)

