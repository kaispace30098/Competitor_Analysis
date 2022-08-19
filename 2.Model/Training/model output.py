#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import sklearn
import random
from sklearn.model_selection import KFold
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost as xgb
from sklearn.svm import SVR


# In[2]:


import sklearn.utils.fixes
sklearn.utils.fixes.MaskedArray = MaskedArray
import skopt


# In[2]:


# Loads the Boston House Price Dataset
df1 = pd.read_csv("C:/Users/Tomc/Downloads/bike info 5.csv")
#X = df1[['battery WH', 'motor watts','Rating','Absorbing fork','WHEEL SIZE','Foldable','SPEED GEAR','Fat Tire','external','top speed','Throttle','Charge time','Weight'
        #]]
X = df1[['battery WH', 'motor watts','WHEEL SIZE','top speed','Throttle','Charge time','Weight','Foldable','Fat Tire']]
Y = df1[["Price"]]


# In[ ]:





# In[3]:


from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X) 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled[:-7], Y[:-7], test_size=0.25, random_state=0)


# In[4]:


X_train.shape, y_test.shape


# In[5]:


#Baseline model
model1=RandomForestRegressor(random_state=42)
model1.fit(X_train,y_train.values.ravel())
model1.score(X_test,y_test)


# In[6]:


import pickle
filename = 'finalized_model.pkl'
pickle.dump(model1, open(filename, 'wb'))


# In[7]:


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)


# In[8]:


model1.get_params()


# In[9]:


#define the model
model=RandomForestRegressor(random_state=42)


# In[10]:


#define the evaluation
from sklearn.model_selection import KFold
cv=KFold(n_splits=5, random_state=42,shuffle=True)


# In[11]:


#Define the search space
from sklearn.model_selection import RandomizedSearchCV# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 110, num = 22)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[12]:


#define the search
from sklearn.model_selection import RandomizedSearchCV
search=RandomizedSearchCV(estimator=model,cv=cv,param_distributions=random_grid,n_jobs=-1,random_state=0)


# In[13]:


history=search.fit(X_train,y_train.values.ravel())


# In[14]:


history.best_params_


# In[15]:


acc_score = []
avg_acc_score=0.0
model=RandomForestRegressor(**history.best_params_)
model.fit(X_train,y_train)
model.score(X_test,y_test)


# In[16]:


##XGboost
#acc_score = []
#avg_acc_score=0.0
#kfold = KFold(n_splits=10, shuffle=True, random_state=1)
#for train_index, test_index in kfold.split(X):   
    #X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    #y_train , y_test = Y.iloc[train_index,:],Y.iloc[test_index,:]
    #xg_reg = xgb.XGBRegressor(objective ='reg:squarederror',max_depth = 6,eta=0.31,n_estimators = 500,alpha = 1)
    
    ##xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
    ##xg_reg = xgb.XGBRegressor(objective="reg:linear")
    ## fit scaler on training data
    #norm = MinMaxScaler().fit(X_train)

    ## transform training data
    #X_train_norm = norm.transform(X_train)

    ## transform testing dataabs
    #X_test_norm = norm.transform(X_test)
    #xg_reg.fit(X_train, y_train.values.ravel())
    
    #y_pred = xg_reg.predict(X_test)
    
    #acc_score.append(xg_reg.score(X_test,y_test))
#avg_acc_score = sum(acc_score)/10
#print(acc_score)
#avg_acc_score  


# In[17]:


y_train.shape


# In[18]:


#Basemodel
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror',max_depth = 6,eta=0.31,n_estimators = 200,alpha = 1)
xg_reg.fit(X_train,y_train.values.ravel())
xg_reg.score(X_train,y_train.values.ravel()),xg_reg.score(X_test,y_test.values.ravel())


# In[5]:


#Basemodel
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror',max_depth = 10,eta=0.01,n_estimators = 1000,colsample_bylevel=0.8,colsample_bytree=0.8,subsample=0.6)
xg_reg.fit(X_train,y_train.values.ravel())
xg_reg.score(X_train,y_train.values.ravel()),xg_reg.score(X_test,y_test.values.ravel())


# In[19]:


#define the model
xg_reg1 = xgb.XGBRegressor()


# In[20]:


#define the evaluation
from sklearn.model_selection import KFold
cv=KFold(n_splits=10, random_state=42,shuffle=True)


# In[21]:


#define the search space
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

params1 = { 'max_depth': [3, 5, 6, 10, 15, 20],
           'learning_rate': [0.01, 0.1, 0.2, 0.3,0.35],
           'subsample': np.arange(0.5, 1.0, 0.1),
           'colsample_bytree': np.arange(0.4, 1.0, 0.1),
           'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
           'n_estimators': [100, 200,500, 1000]}


# In[22]:


#define the search

from sklearn.model_selection import RandomizedSearchCV
search=RandomizedSearchCV(estimator=xg_reg1,cv=cv,param_distributions=params1,n_jobs=-1,random_state=0)


# In[23]:


history=search.fit(X_train,y_train.values.ravel())


# In[24]:


model1=xgb.XGBRegressor(**history.best_params_)
model1.fit(X_train,y_train.values.ravel())
model1.score(X_test,y_test)


# In[7]:


import pickle
filename = 'finalized_model1.pkl'
pickle.dump(xg_reg, open(filename, 'wb'))


# In[26]:


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

