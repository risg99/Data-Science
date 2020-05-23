#!/usr/bin/env python
# coding: utf-8

# Step 1 - Hypothesis Generation: Identifying the factors that are probable to affect the outcome.

# a) As time increases, number of passengers will increase. Since, population has a general upward trend with time, so more people can travel by JetRail. Also, companies try to expand their businesses over time, attracting more customers.  

# b) High traffic from May to October, as it is the time tourists come to visit.

# c) Weekdays, there will be more traffic than in weekends/holidays.

# d) Traffic during peak hours will be high, as people travel to work, colleges, etc.

# In[50]:


# Importing the necessary libraries

import pandas as pd          
import numpy as np          # For mathematical calculations 
import matplotlib.pyplot as plt  # For plotting graphs 
from datetime import datetime    # To access datetime 
from pandas import Series        # To work on series 
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings                    
warnings.filterwarnings("ignore") # To ignore the warnings


# In[51]:


# Reading the datasets for train and test
train = pd.read_csv("train.csv") 
test = pd.read_csv("test.csv")


# In[52]:


# Making a copy of dataset to prevent losing the original data.
train_original = train.copy() 
test_original = test.copy()


# In[53]:


# Peeking the data we have: first its features(columns) and then the data itself
print("Printing the training data columns")
print(train.columns)
print("------------------------------")
print("Printing the testing data columns")
print(test.columns)
print("------------------------------")

print()

print("Printing the training data columns datatypes")
print(train.dtypes)
print("------------------------------")
print("Printing the testing data columns datatypes")
print(test.dtypes)
print("------------------------------")

print()

print("Printing the training data")
print(train.head())
print("------------------------------")

print()

print("Printing the training data shape")
print(train.shape)
print("------------------------------")
print("Printing the testing data shape")
print(test.shape)
print("------------------------------")


# -------------------------------------------------------------------------------------------

# Step 2 - Feature Extraction : Getting features that are needed to validate hypotheses made.

# In[54]:


# Extracting the features to validate our hypothesis

train['Datetime'] = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M')
test['Datetime'] = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M')


# In[55]:


# Making the changes on the original dataset as well

test_original['Datetime'] = pd.to_datetime(test_original.Datetime,format='%d-%m-%Y %H:%M') 
train_original['Datetime'] = pd.to_datetime(train_original.Datetime,format='%d-%m-%Y %H:%M')


# In[56]:


# Extracting year, month, day, hour from the datetime to validate the hypothesis

for i in (train, test, test_original, train_original):
    i['year'] = i.Datetime.dt.year 
    i['month'] = i.Datetime.dt.month 
    i['day'] = i.Datetime.dt.day
    i['hour'] = i.Datetime.dt.hour 


# In[57]:


# Getting the weekday for each Datetime

train['day of week'] = train['Datetime'].dt.dayofweek 
temp = train['Datetime']


# In[58]:


# Function which is assigning 1 if weekend, else 0
def assign(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0 
    
temp2 = train['Datetime'].apply(assign) 
train['weekend'] = temp2


# In[59]:


# We can see the new columns have been added

print(train.head())
print(train.tail())


# In[60]:


# Plotting the time-series for now

train.index = train['Datetime'] # indexing the Datetime to get the time period on the x-axis. 
df = train.drop('ID',1)           # drop ID variable to get only the Datetime on x-axis. 
ts = df['Count'] 
plt.figure(figsize=(16,8)) 
plt.plot(ts, label='Passenger Count') 
plt.title('Time Series') 
plt.xlabel("Time(year-month)") 
plt.ylabel("Passenger count") 
plt.legend(loc='best')


# We can see that there is an increase in the traffic as time increases. Also, in some places there is an irregular increase.

# -----------------------------------------------------------------------------------------------------

# Step 3 - Exploratory Analysis : Verifying our hypotheses made using the actual data

# In[61]:


train.groupby('year')['Count'].mean().plot.bar()


# As, years passby, traffic is seen to increase. 

# In[62]:


train.groupby('month')['Count'].mean().plot.bar()


# There is a decrease in traffic in the last three months.

# In[63]:


temp = train.groupby(['year','month'])['Count'].mean()
temp.plot(figsize=(20,8), title = 'Passenger Count(Monthwise)', fontsize=14)


# Some observations made:
# 
# 1. The last three months of the year 10,11 and 12 aren't present in 2014. The mean in 2012 is very less.
# 
# 2. Since there is an increasing trend in our time series, the mean value for rest of the months will be more because of their larger passenger counts in year 2014 and we will get smaller value for these 3 months.
# 
# 3. In the above line plot we can see an increasing trend in monthly passenger count and the growth is approximately exponential.

# In[64]:


train.groupby('day')['Count'].mean().plot.bar()


# Day wise count of the passengers isn't that very insightful.

# In[65]:


train.groupby('hour')['Count'].mean().plot.bar()


# Peak traffic is at 7 p.m. and then 11 a.m. to 12 noon. We see a decreasing trend from 12 at night till 5 a.m.

# In[66]:


train.groupby('weekend')['Count'].mean().plot.bar()


# Inference : Traffic is more on weekdays than in weekends.

# In[67]:


new = train
new['weekend'] = new['weekend'].apply(lambda x : 'weekday' if x == True else 'not a weekday')


# In[68]:


new.groupby('weekend')['Count'].mean().plot.bar()
# Just labelled weekday as 0 and weekend as 1 in the bar plot


# In[69]:


train.groupby('day of week')['Count'].mean().plot.bar()


# 0 - Monday, 6 - Sunday. More traffic on weekdays than on weekends.

# In[70]:


# Since, ID column has nothing to do with passenger count.
train = train.drop('ID',1)


# In[71]:


train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
train.index = train.Timestamp 
# Hourly time series 
hourly = train.resample('H').mean() 
# Converting to daily mean 
daily = train.resample('D').mean() 
# Converting to weekly mean 
weekly = train.resample('W').mean() 
# Converting to monthly mean 
monthly = train.resample('M').mean()


# In[72]:


# Let’s look at the hourly, daily, weekly and monthly time series.

fig, axs = plt.subplots(4,1) 
hourly.Count.plot(figsize=(15,8), title= 'Hourly', fontsize=14, ax=axs[0])
daily.Count.plot(figsize=(15,8), title= 'Daily', fontsize=14, ax=axs[1]) 
weekly.Count.plot(figsize=(15,8), title= 'Weekly', fontsize=14, ax=axs[2]) 
monthly.Count.plot(figsize=(15,8), title= 'Monthly', fontsize=14, ax=axs[3]) 

plt.show()


# In[73]:


test.Timestamp = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 
test.index = test.Timestamp  

# Converting to daily mean 
test = test.resample('D').mean() 

train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 

train.index = train.Timestamp 
# Converting to daily mean 
train = train.resample('D').mean()


# We have done time based validation here by selecting the last 3 months for the validation data and rest in the train data. If we would have done it randomly it may work well for the train dataset but will not work effectively on validation dataset.
# 
# Lets understand it in this way: If we choose the split randomly it will take some values from the starting and some from the last years as well. It is similar to predicting the old values based on the future values which is not the case in real scenario. So, this kind of split is used while working with time related problems.

# In[74]:


# Splitting the training and validation part
Train=train.ix['2012-08-25':'2014-06-24'] 
valid=train.ix['2014-06-25':'2014-09-25']


# In[75]:


# Plotting the training and validation part : Blue = Training and Orange = Validation
Train.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='train') 
valid.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='valid')
plt.xlabel("Datetime") 
plt.ylabel("Passenger count") 
plt.legend(loc='best') 
plt.show()


# In[76]:


# Using Naive Method to validate

dd = np.asarray(Train.Count) 
y_hat = valid.copy() 
y_hat['naive'] = dd[len(dd)-1] 
plt.figure(figsize=(12,8)) 
plt.plot(Train.index, Train['Count'], label='Train') 
plt.plot(valid.index,valid['Count'], label='Valid') 
plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast') 
plt.legend(loc='best') 
plt.title("Naive Forecast") 
plt.show()


# In[77]:


# Calculating rmse for the predicted values

from sklearn.metrics import mean_squared_error 
from math import sqrt 
rms = sqrt(mean_squared_error(valid.Count, y_hat.naive)) 
print(rms)


# We can infer that this method is not suitable for datasets with high variability. We can reduce the rmse value by adopting different techniques.

# In[78]:


# Using rolling mean to validate
# a) Mean of past 10 obserevations

y_hat_avg = valid.copy() 
y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(10).mean().iloc[-1] # average of last 10 observations. 
plt.figure(figsize=(15,5)) 
plt.plot(Train['Count'], label='Train') 
plt.plot(valid['Count'], label='Valid') 
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 10 observations') 
plt.legend(loc='best') 
plt.show()


# In[79]:


#  b) Mean of past 20 observations

y_hat_avg = valid.copy() 
y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(20).mean().iloc[-1] # average of last 20 observations. 
plt.figure(figsize=(15,5)) 
plt.plot(Train['Count'], label='Train') 
plt.plot(valid['Count'], label='Valid') 
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 20 observations') 
plt.legend(loc='best') 
plt.show() 


# In[80]:


# c) Mean of past 50 observations

y_hat_avg = valid.copy() 
y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(50).mean().iloc[-1] # average of last 50 observations. 
plt.figure(figsize=(15,5)) 
plt.plot(Train['Count'], label='Train') 
plt.plot(valid['Count'], label='Valid') 
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 50 observations') 
plt.legend(loc='best') 
plt.show()


# In[81]:


# Calculating rmse for the same

rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.moving_avg_forecast)) 
print(rms)


# We took the average of last 10, 20 and 50 observations and predicted based on that. This value can be changed in the above code in .rolling().mean() part. We can see that the predictions are getting weaker as we increase the number of observations.
# 

# In[82]:


# Using Simple Exponential Smoothing to validate

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt 
y_hat_avg = valid.copy() 
fit2 = SimpleExpSmoothing(np.asarray(Train['Count'])).fit(smoothing_level=0.6,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(valid)) 
plt.figure(figsize=(16,8)) 
plt.plot(Train['Count'], label='Train') 
plt.plot(valid['Count'], label='Valid') 
plt.plot(y_hat_avg['SES'], label='SES') 
plt.legend(loc='best') 
plt.show()


# In[83]:


rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.SES)) 
print(rms)


# We can infer that the fit of the model has improved as the rmse value has reduced.

# Using Holt’s Linear Trend Model:
# 
# It is an extension of simple exponential smoothing to allow forecasting of data with a trend.
# 
# This method takes into account the trend of the dataset. The forecast function in this method is a function of level and trend.
# First of all let us visualize the trend, seasonality and error in the series.
# 
# We can decompose the time series in four parts-
# 
# - Observed, which is the original time series.
# - Trend, which shows the trend in the time series, i.e., increasing or decreasing behaviour of the time series.
# - Seasonal, which tells us about the seasonality in the time series.
# - Residual, which is obtained by removing any trend or seasonality in the time series.

# In[84]:


# Using Holt’s Linear Trend Model to decompose 

import statsmodels.api as sm 
sm.tsa.seasonal_decompose(Train.Count).plot() 
result = sm.tsa.stattools.adfuller(train.Count) 
plt.show()


# An increasing trend can be seen in the dataset, so now we will make a model based on the trend.

# In[85]:


y_hat_avg = valid.copy() 
fit1 = Holt(np.asarray(Train['Count'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1) 
y_hat_avg['Holt_linear'] = fit1.forecast(len(valid)) 
plt.figure(figsize=(16,8)) 
plt.plot(Train['Count'], label='Train') 
plt.plot(valid['Count'], label='Valid') 
plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear') 
plt.legend(loc='best') 
plt.show()


# We can see an inclined line here as the model has taken into consideration the trend of the time series.

# In[86]:


rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.Holt_linear)) 
print(rms)


# It can be inferred that the rmse value has decreased.

# In[87]:


predict = fit1.forecast(len(test))


# In[88]:


test['prediction'] = predict


# Remember this is the daily predictions. We have to convert these predictions to hourly basis. 
# 
# * To do so we will first calculate the ratio of passenger count for each hour of every day. 
# * Then we will find the average ratio of passenger count for every hour and we will get 24 ratios. 
# * Then to calculate the hourly predictions we will multiply the daily prediction with the hourly ratio.

# In[89]:


# Calculating the hourly ratio of count 
train_original['ratio']=train_original['Count']/train_original['Count'].sum() 

# Grouping the hourly ratio 
temp=train_original.groupby(['hour'])['ratio'].sum() 

# Groupby to csv format 
pd.DataFrame(temp, columns=['hour','ratio']).to_csv('GROUPby.csv')


# In[90]:


# Groupby to csv format 
pd.DataFrame(temp, columns=['hour','ratio']).to_csv('GROUPby.csv') 

temp2=pd.read_csv("GROUPby.csv") 
temp2=temp2.drop('hour.1',1) 


# In[92]:


# Merge Test and test_original on day, month and year 
merge = pd.merge(test, test_original, on=('day','month', 'year'), how='left') 
merge['hour'] = merge['hour_y'] 
merge = merge.drop(['year', 'month', 'Datetime','hour_x','hour_y'], axis=1) 

# Predicting by merging merge and temp2 
prediction = pd.merge(merge, temp2, on='hour', how='left') 


# In[93]:


# Converting the ratio to the original scale 
prediction['Count'] = prediction['prediction']*prediction['ratio']*24 
prediction['ID']= prediction['ID_y']


# In[95]:


# Let’s drop all other features from the submission file and keep ID and Count only.

submission = prediction.drop(['ID_x', 'day', 'ID_y','prediction','hour', 'ratio'],axis=1) 

# Converting the final submission to csv format 
pd.DataFrame(submission, columns=['ID','Count']).to_csv('Holt linear.csv')


# Holt winter’s model on daily time series:- 
# 
# - Datasets which show a similar set of pattern after fixed intervals of a time period suffer from seasonality.
# 
# - The above mentioned models don’t take into account the seasonality of the dataset while forecasting. Hence we need a method that takes into account both trend and seasonality to forecast future prices.
# 
# - One such algorithm that we can use in such a scenario is Holt’s Winter method. The idea behind Holt’s Winter is to apply exponential smoothing to the seasonal components in addition to level and trend.

# In[96]:


# Let’s first fit the model on training dataset and validate it using the validation dataset.

y_hat_avg = valid.copy() 
fit1 = ExponentialSmoothing(np.asarray(Train['Count']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit() 
y_hat_avg['Holt_Winter'] = fit1.forecast(len(valid)) 
plt.figure(figsize=(16,8)) 
plt.plot( Train['Count'], label='Train') 
plt.plot(valid['Count'], label='Valid') 
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter') 
plt.legend(loc='best') 
plt.show()


# In[97]:


rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.Holt_Winter)) 
print(rms)


# We can see that the rmse value has reduced a lot from this method. 

# In[100]:


predict=fit1.forecast(len(test))
# Now we will convert these daily passenger count into hourly passenger count using the same approach which we followed above.

test['prediction']=predict
# Merge Test and test_original on day, month and year 
merge=pd.merge(test, test_original, on=('day','month', 'year'), how='left') 
merge['hour']=merge['hour_y'] 
merge=merge.drop(['year', 'month', 'Datetime','hour_x','hour_y'], axis=1) 

# Predicting by merging merge and temp2 
prediction=pd.merge(merge, temp2, on='hour', how='left') 

# Converting the ratio to the original scale prediction['Count']=prediction['prediction']*prediction['ratio']*24
# Let’s drop all features other than ID and Count

prediction['ID']=prediction['ID_y'] 
submission=prediction.drop(['day','hour','ratio','prediction', 'ID_x', 'ID_y'],axis=1) 

# Converting the final submission to csv format pd.DataFrame(submission, columns=['ID','Count']).to_csv('Holt winters.csv')


# In[111]:


# Let’s make a function which we can use to calculate the results of Dickey-Fuller test.

from statsmodels.tsa.stattools import adfuller 

def test_stationarity(timeseries):
        #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=24).mean() # 24 hours on each day
    rolstd = pd.Series(timeseries).rolling(window=24).std()
        #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
        #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# In[112]:


from matplotlib.pylab import rcParams 
rcParams['figure.figsize'] = 20,10
test_stationarity(train_original['Count'])


# The statistics shows that the time series is stationary as Test Statistic < Critical value but we can see an increasing trend in the data. 

# So, firstly we will try to make the data more stationary. For doing so, we need to remove the trend and seasonality from the data.
# 
# Removing Trend:- 
# 
# - A trend exists when there is a long-term increase or decrease in the data. It does not have to be linear.
# 
# - We see an increasing trend in the data so we can apply transformation which penalizes higher values more than smaller ones, for example log transformation.
# 
# - We will take rolling average here to remove the trend. We will take the window size of 24 based on the fact that each day has 24 hours.
# 
# 

# In[114]:


Train_log = np.log(Train['Count']) 
valid_log = np.log(valid['Count'])
moving_avg = Train_log.rolling(24).mean()
plt.plot(Train_log) 
plt.plot(moving_avg, color = 'red') 
plt.show()


# In[115]:


train_log_moving_avg_diff = Train_log - moving_avg


# In[116]:


# Since we took the average of 24 values, rolling mean is not defined for the first 23 values. So let’s drop those null values.

train_log_moving_avg_diff.dropna(inplace = True) 
test_stationarity(train_log_moving_avg_diff)


# In[117]:


# Differencing can help to make the series stable and eliminate the trend.
train_log_diff = Train_log - Train_log.shift(1) 
test_stationarity(train_log_diff.dropna())


# Removing Seasonality
# 
# - By seasonality, we mean periodic fluctuations. 
# - A seasonal pattern exists when a series is influenced by seasonal factors (e.g., the quarter of the year, the month, or day of the week).
# - Seasonality is always of a fixed and known period.
# - We will use seasonal decompose to decompose the time series into trend, seasonality and residuals.

# In[118]:


from statsmodels.tsa.seasonal import seasonal_decompose 
decomposition = seasonal_decompose(pd.DataFrame(Train_log).Count.values, freq = 24) 

trend = decomposition.trend 
seasonal = decomposition.seasonal 
residual = decomposition.resid 

plt.subplot(411) 
plt.plot(Train_log, label='Original') 
plt.legend(loc='best') 
plt.subplot(412) 
plt.plot(trend, label='Trend') 
plt.legend(loc='best') 
plt.subplot(413) 
plt.plot(seasonal,label='Seasonality') 
plt.legend(loc='best') 
plt.subplot(414) 
plt.plot(residual, label='Residuals') 
plt.legend(loc='best') 
plt.tight_layout() 
plt.show()


# In[120]:


# Let’s check stationarity of residuals.

train_log_decompose = pd.DataFrame(residual) 
train_log_decompose['date'] = Train_log.index 
train_log_decompose.set_index('date', inplace = True) 
train_log_decompose.dropna(inplace=True) 
test_stationarity(train_log_decompose[0])


# It can be interpreted from the results that the residuals are stationary.
# 
# Now we will forecast the time series using different models.

# In[121]:


from statsmodels.tsa.stattools import acf, pacf 
lag_acf = acf(train_log_diff.dropna(), nlags=25) 
lag_pacf = pacf(train_log_diff.dropna(), nlags=25, method='ols')


# In[124]:


# ACF and PACF plot
plt.plot(lag_acf) 
plt.axhline(y=0,linestyle='--',color='gray') 
plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray') 
plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray') 
plt.title('Autocorrelation Function') 
plt.show() 


plt.plot(lag_pacf) 
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray') 
plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray') 
plt.title('Partial Autocorrelation Function') 
plt.show()


# - p value is the lag value where the PACF chart crosses the upper confidence interval for the first time. It can be noticed that in this case p=1.
# - q value is the lag value where the ACF chart crosses the upper confidence interval for the first time. It can be noticed that in this case q=1.
# 
# Now we will make the ARIMA model as we have the p,q values. We will make the AR and MA model separately and then combine them together.

# In[125]:


# AR MODEL
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(Train_log, order=(2, 1, 0))  # here the q value is zero since it is just the AR model 
results_AR = model.fit(disp=-1)  
plt.plot(train_log_diff.dropna(), label='original') 
plt.plot(results_AR.fittedvalues, color='red', label='predictions') 
plt.legend(loc='best') 
plt.show()


# In[126]:


AR_predict=results_AR.predict(start="2014-06-25", end="2014-09-25") 
AR_predict=AR_predict.cumsum().shift().fillna(0) 
AR_predict1=pd.Series(np.ones(valid.shape[0]) * np.log(valid['Count'])[0], index = valid.index) 
AR_predict1=AR_predict1.add(AR_predict,fill_value=0) 
AR_predict = np.exp(AR_predict1)


plt.plot(valid['Count'], label = "Valid") 
plt.plot(AR_predict, color = 'red', label = "Predict") 
plt.legend(loc= 'best') 
plt.title('RMSE: %.4f'% (np.sqrt(np.dot(AR_predict, valid['Count']))/valid.shape[0])) 
plt.show()


# In[127]:


# MA MODEL

model = ARIMA(Train_log, order=(0, 1, 2))  # here the p value is zero since it is just the MA model 
results_MA = model.fit(disp=-1)  
plt.plot(train_log_diff.dropna(), label='original') 
plt.plot(results_MA.fittedvalues, color='red', label='prediction') 
plt.legend(loc='best') 
plt.show()


# In[129]:


MA_predict=results_MA.predict(start="2014-06-25", end="2014-09-25") 
MA_predict=MA_predict.cumsum().shift().fillna(0) 
MA_predict1=pd.Series(np.ones(valid.shape[0]) * np.log(valid['Count'])[0], index = valid.index) 
MA_predict1=MA_predict1.add(MA_predict,fill_value=0) 
MA_predict = np.exp(MA_predict1)


plt.plot(valid['Count'], label = "Valid") 
plt.plot(MA_predict, color = 'red', label = "Predict") 
plt.legend(loc= 'best') 
plt.title('RMSE: %.4f'% (np.sqrt(np.dot(MA_predict, valid['Count']))/valid.shape[0])) 
plt.show()


# In[130]:


model = ARIMA(Train_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(train_log_diff.dropna(),  label='original') 
plt.plot(results_ARIMA.fittedvalues, color='red', label='predicted') 
plt.legend(loc='best') 
plt.show()


# In[131]:


def check_prediction_diff(predict_diff, given_set):
    predict_diff= predict_diff.cumsum().shift().fillna(0)
    predict_base = pd.Series(np.ones(given_set.shape[0]) * np.log(given_set['Count'])[0], index = given_set.index)
    predict_log = predict_base.add(predict_diff,fill_value=0)
    predict = np.exp(predict_log)

    plt.plot(given_set['Count'], label = "Given set")
    plt.plot(predict, color = 'red', label = "Predict")
    plt.legend(loc= 'best')
    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['Count']))/given_set.shape[0]))
    plt.show()

def check_prediction_log(predict_log, given_set):
    predict = np.exp(predict_log)
 
    plt.plot(given_set['Count'], label = "Given set")
    plt.plot(predict, color = 'red', label = "Predict")
    plt.legend(loc= 'best')
    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['Count']))/given_set.shape[0]))
    plt.show()
    
# Let’s predict the values for validation set.

ARIMA_predict_diff=results_ARIMA.predict(start="2014-06-25", end="2014-09-25")
check_prediction_diff(ARIMA_predict_diff, valid)


# In[133]:


import statsmodels.api as sm
y_hat_avg = valid.copy() 
fit1 = sm.tsa.statespace.SARIMAX(Train.Count, order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit() 
y_hat_avg['SARIMA'] = fit1.predict(start="2014-6-25", end="2014-9-25", dynamic=True) 
plt.figure(figsize=(16,8)) 
plt.plot( Train['Count'], label='Train') 
plt.plot(valid['Count'], label='Valid') 
plt.plot(y_hat_avg['SARIMA'], label='SARIMA') 
plt.legend(loc='best') 
plt.show()


# In[134]:


rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.SARIMA)) 
print(rms)


# In[135]:


predict=fit1.predict(start="2014-9-26", end="2015-4-26", dynamic=True)


# In[137]:


test['prediction']=predict
# Merge Test and test_original on day, month and year 
merge=pd.merge(test, test_original, on=('day','month', 'year'), how='left') 
merge['hour']=merge['hour_y'] 
merge=merge.drop(['year', 'month', 'Datetime','hour_x','hour_y'], axis=1) 

# Predicting by merging merge and temp2 
prediction=pd.merge(merge, temp2, on='hour', how='left') 

# Converting the ratio to the original scale 
prediction['Count']=prediction['prediction']*prediction['ratio']*24


# In[139]:


prediction['ID']=prediction['ID_y'] 
submission=prediction.drop(['day','hour','ratio','prediction', 'ID_x', 'ID_y'],axis=1) 

# Converting the final submission to csv format 
pd.DataFrame(submission, columns=['ID','Count']).to_csv('SARIMAX.csv')

