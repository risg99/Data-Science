#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the necessary libraries

import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_boston
boston_dataset = load_boston()


# In[4]:


print(boston_dataset.keys())


# In[6]:


print(boston_dataset['data'])


# In[7]:


print(boston_dataset['target'])


# In[8]:


print(boston_dataset['feature_names'])


# In[10]:


print(boston_dataset['DESCR'])


# In[11]:


print(boston_dataset['filename'])


# From the above findings, we come to know that, MEDV is our target variable. 

# In[12]:


boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()


# Here, MEDV is absent in the dataframe. So, we create an empty column and add it to the dataframe.

# In[13]:


boston['MEDV'] = boston_dataset.target


# In[15]:


boston.isnull().sum()


# Since, no null values are present in the dataset, it is clean.

# In[16]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt.show()


# MEDV values are distributed normally with a few outliers.

# In[17]:


correlation_matrix = boston.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)


# 1 means a strong positive correlation, while -1 means strong negative correlation.

# Some of the conclusions drawn from the above exploratory data analysis:
# 
# 1. To fit our linear regression model, we need to choose a variable with high correlation with the variable MEDV. Since, RM has the value of 0.7, i.e it has a strong positive corelation and LSTAT has -0.74, i.e it has a strong negative correlation, therefore we choose those variables.
# 
# 2. To create a linear regession model we mustnot select the features which are strongly correlated to each other, i.e for it we need to check for multi-co-linearity. If such features are present we shouldn't select them to train the model. Here, RAD and TAX have a strong positive correlation of 0.91.
# 
# 

# --------------------------------------------------------------
# 

# Considering, RM and STAT for training the model, we will plot a scatter plot against the variable MEDV.

# In[18]:


plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')


# Prices decrease linearly as LSTAT increases, though it is not an exact line. Some, outliers are present at 50.
# 
# Also, prices are increasing linearly as RM increases. Here too some outliers are capped at 50.
# 

# In[19]:


# Training the model

X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']


# In[20]:


print(len(X),len(Y))


# In[21]:


# Using 80-20 rule to split the dataset

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[24]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


# In[25]:


# model evaluation for training set
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[26]:


print("Printing the prices for the testing part of the model:")
print(y_train_predict)

