#!/usr/bin/env python
# coding: utf-8

# In[108]:


# Importing the necessary libraries

import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn import metrics
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('winequality-red.csv', sep = ';')
print(df.head())


# In[4]:


print("Printing the columns of the dataset")
print(df.columns)
print("----------------------------------------")
print()
print("Printing the shape of the dataset")
print(df.shape)
print("----------------------------------------")
print()
print("Printing the size of the dataset")
print(df.size)
print("----------------------------------------")


# In[5]:


print(df.isnull().sum())


# Inference: No null values in the dataset, i.e. data is clean.

# In[47]:


# Since all values are numbers and there is no categorical variable as such, it is a regression problem.

correlations = df.corr()['quality'].drop('quality')
featuress = df.columns
for i in zip(df.columns,correlations):
    print(i)
print(correlations)


# In[17]:


# To get a heatmap

sns.heatmap(df.corr(),annot=True)
plt.show()


# In[106]:


plt.hist(df.alcohol,facecolor ='red',alpha = 0.5, label ="Red wine")
plt.xlabel("Alcohol in % Vol")
plt.ylabel("Frequency") 
plt.title("Distribution of Alcohol in % Vol") 
plt.show() 


# In[101]:


# Defining a function to get a list of features with threshold above a given threshold

def get_features(correlation_threshold):
    abs_corrs = correlations.abs()
    high_correlations = {}
    print(abs_corrs)
    index = 0
    for i in abs_corrs:
        if i > correlation_threshold:
            high_correlations[featuress[index]] = i
        else:
            print("not appending: ",i)
        index = index + 1
    return high_correlations


# In[72]:


features = get_features(0.05)
print("Printing the features with threshold of correlation above 0.05 are: ",features)
print()


# In[74]:


print(features.keys())


# In[75]:


# Creating the x and y variables

x = df[features.keys()] 
y = df['quality']


# In[76]:


print(x)
print('--------------')
print(y)


# In[77]:


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=3)


# In[79]:


print("Printing the x_train shape")
print(x_train.shape)
print('-------------------------')
print("Printing the x_test shape")
print(x_test.shape)
print('-------------------------')
print("Printing the y_train shape")
print(y_train.shape)
print('-------------------------')
print("Printing the y_test shape")
print(y_test.shape)
print('-------------------------')


# In[83]:


# Time to build a linear regression model

regressor = LinearRegression()
regressor.fit(x_train,y_train)
print(regressor)
print()
print(regressor.coef_)


# In[85]:


train_pred = regressor.predict(x_train)
print("Printing the train predict values")
print(train_pred)
print('---------------------------------')


# In[86]:


test_pred = regressor.predict(x_test) 
print("Printing the test predict values")
print(test_pred)
print('---------------------------------')


# In[90]:


# calculating rmse

train_rmse = metrics.mean_squared_error(train_pred, y_train) ** 0.5
print(train_rmse)
test_rmse = metrics.mean_squared_error(test_pred, y_test) ** 0.5
print(test_rmse)


# In[91]:


# rounding off the predicted values for test set

predicted_data = np.round_(test_pred)
print(predicted_data)


# In[92]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))


# In[94]:


# displaying coefficients of each feature
coeffecients = pd.DataFrame(regressor.coef_,features) 
coeffecients.columns = ['Coeffecient'] 
print(coeffecients)


# These numbers mean that holding all other features fixed, a 1 unit increase in sulphates will lead to an increase of 0.8 in quality of wine, and similarly for the other features.
# Also holding all other features fixed, a 1 unit increase in volatile acidity will lead to a decrease of 0.99 in quality of wine, and similarly for the other features.

# In[109]:


# Importing the necessary libraries for keras models

# Import `Sequential` from `keras.models` 
from keras.models import Sequential 
  
# Import `Dense` from `keras.layers` 
from keras.layers import Dense 


# In[112]:


# Time to build a neural network model

# Initialize the constructor 
model = Sequential() 
  
# Add an input layer 
model.add(Dense(12, activation ='relu', input_shape =(10, ))) 
  
# Add one hidden layer 
model.add(Dense(9, activation ='relu')) 
  
# Add an output layer 
model.add(Dense(1, activation ='sigmoid')) 
  
# Model output shape 
model.output_shape 
  
# Model summary 
model.summary() 
  
# Model config 
model.get_config() 
  
# List all weight tensors 
model.get_weights() 
model.compile(loss ='binary_crossentropy', optimizer ='adam', metrics =['accuracy']) 


# In[113]:


# Training Model 
model.fit(x_train, y_train, epochs = 3, batch_size = 1, verbose = 1) 
   
# Predicting the Value 
y_pred = model.predict(x_test) 
print(y_pred)

