#!/usr/bin/env python
# coding: utf-8

# In[14]:


import datetime as dt

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing 
get_ipython().run_line_magic('config', 'InlineBackend.rc')


# In[10]:


forest = pd.read_csv('forestfires.csv')


# In[12]:


forest


# In[9]:


forest.describe()


# In[4]:


forest.isnull().sum()


# In[8]:


forest.head()


# In[6]:


plt.figure(figsize=(10, 10))
sns.heatmap(forest.corr(),annot=True,cmap='magma',linewidths=.5)


# In[16]:


# Convert categorical variables to numerical
label_encoder = preprocessing.LabelEncoder()
forest["day"] = label_encoder.fit_transform(forest["day"])
forest["month"] = label_encoder.fit_transform(forest["month"])


# In[19]:


plt.figure(figsize=(10, 10))
sns.heatmap(forest.corr(),annot=True,cmap='magma',linewidths=.5)


# In[20]:


# Split the dataset into features and target
y = forest["month"]
X = forest.drop(["month"], axis=1)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


# Train the random forest classifier
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)
random_model_accuracy = round(rf.score(X_train, y_train)*100,2)
print("Accuracy is : ", round(random_model_accuracy, 2), '%')


# In[23]:


#Checking the accuracy 

random_model_accuracy = round(rf.score(X_test, y_test)*100,2)
print(round(random_model_accuracy, 2), '%')


# In[ ]:




