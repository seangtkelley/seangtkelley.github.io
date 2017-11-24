---
layout: post
title: "Techniques for Reading Bitcoin Price Action"
desc: "A comparison of price action prediction accuracy between a basic artificial neural network and a Long-Short Term Memory (LTSM) network."
tag: "Machine Learning"
author: "Sean Kelley"
thumb: "/img/blog/bitcoin.jpg"
date: 2017-11-24
---

# Techniques for Reading Bitcoin Price Action

This notebook was heavily influenced by the

Plotting training/val loss from this: <link from reddit keras post>

## Feature Extraction


```python
import numpy as np
import pandas as pd
import datetime, pytz

# define a conversion function for the native timestamps in the csv file
def dateparse (time_in_secs):    
    return pytz.utc.localize(datetime.datetime.fromtimestamp(float(time_in_secs)))

# read data
df = pd.read_csv('coinbaseUSD_1-min_data_2014-12-01_to_2017-10-20.csv.csv', parse_dates=[0], date_parser=dateparse)
```


```python
# resample data
df.set_index('Timestamp', inplace=True)

df['V'] = df['Weighted_Price'] - df['Weighted_Price'].shift(1)
df['V+1'] = df['V'].shift(-1)
df['WC'] = df['Weighted_Price'] - df['Close']

df['HO'] = df['High'] - df['Open']
df['LO'] = df['Low'] - df['Open']
df['CO'] = df['Close'] - df['Open']
df['WO'] = df['Weighted_Price'] - df['Open']

df['day'] = df.index.dayofweek
df['time'] = df.index.hour + df.index.minute/60

df_Xt = df.iloc[:,-6:]
df_Xt.tail()
```


```python
UP = np.logical_and(df['V+1'].round(decimals=2)>0, np.logical_and(df['CO']>0, df['WO']>0)).astype(int)
DN = np.logical_and(df['V+1'].round(decimals=2)<0, np.logical_and(df['CO']<0, df['WO']<0)).astype(int)
FLAT = np.logical_and(UP==0, DN==0).astype(int)
df_Yt = pd.concat([UP, DN, FLAT], join = 'outer', axis =1)
df_Yt.columns = ['UP', 'DN', 'FLAT']
```

## Basic ANN Setup


```python
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

model = Sequential()
model.add(Dense(32, activation = 'tanh', input_dim=6))
model.add(Dropout(0.2))
model.add(Dense(32, activation = 'tanh'))
model.add(Dropout(0.1))
model.add(Dense(32, activation = 'tanh'))
model.add(Dropout(0.1))
model.add(Dense(3, activation = 'softmax'))
# out shaped on df_Yt.shape[1]
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

## Training


```python
batch_size = 60*24 # Total 'blocks/snapshot' in a day
epochs = 100

train_Xt_array = df_Xt.loc['2015-1-1':'2016-12-31'].values
train_y_array = df_Yt.loc['2015-1-1':'2016-12-31'].values

#train_Xt_array = np.reshape(train_Xt_array, (train_Xt_array.shape[0], 1, train_Xt_array.shape[1]))

test_Xt_array = df_Xt.loc['2017-1-1':'2017-10-19'].values
test_y_array = df_Yt.loc['2017-1-1':'2017-10-19'].values

#test_Xt_array = np.reshape(test_Xt_array, (test_Xt_array.shape[0], 1, test_Xt_array.shape[1]))

history = model.fit(train_Xt_array, train_y_array, epochs=epochs, batch_size=batch_size, verbose=1,
                   validation_data=(test_Xt_array, test_y_array))
```


```python
Series_pred = np.argmax(model.predict(test_Xt_array, batch_size=batch_size, verbose = 0), axis = 1)

Series_actual = np.argmax(test_y_array, axis = 1)

classreport= classification_report(Series_actual, Series_pred, target_names = df_Yt.columns, digits = 4)
print(classreport)
```

## Checking for Overfitting


```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

%matplotlib inline
```


```python
[test_loss, test_acc] = model.evaluate(test_Xt_array, test_y_array)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
```


```python
#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-8564cbd62e5e> in <module>()
          1 #Plot the Loss Curves
    ----> 2 plt.figure(figsize=[8,6])
          3 plt.plot(history.history['loss'],'r',linewidth=3.0)
          4 plt.plot(history.history['val_loss'],'b',linewidth=3.0)
          5 plt.legend(['Training loss', 'Validation Loss'],fontsize=18)


    NameError: name 'plt' is not defined


## Basic LTSM Setup


```python
model = Sequential()
model.add(LSTM(units = 32, activation = 'tanh', input_shape=(None, 6)))
model.add(Dropout(0.2))
model.add(Dense(units = 3, activation = 'softmax'))
# out shaped on df_Yt.shape[1]
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-2-b820e448e755> in <module>()
    ----> 1 model = Sequential()
          2 model.add(LSTM(units = 32, activation = 'tanh', input_shape=(None, 6)))
          3 model.add(Dropout(0.2))
          4 model.add(Dense(units = 3, activation = 'softmax'))
          5 # out shaped on df_Yt.shape[1]


    NameError: name 'Sequential' is not defined



```python
batch_size = 60*24 # Total 'blocks/snapshot' in a day
epochs = 100

train_Xt_array = df_Xt.loc['2015-1-1':'2016-12-31'].values
train_y_array = df_Yt.loc['2015-1-1':'2016-12-31'].values

train_Xt_array = np.reshape(train_Xt_array, (train_Xt_array.shape[0], 1, train_Xt_array.shape[1]))

test_Xt_array = df_Xt.loc['2017-1-1':'2017-10-19'].values
test_y_array = df_Yt.loc['2017-1-1':'2017-10-19'].values

test_Xt_array = np.reshape(test_Xt_array, (test_Xt_array.shape[0], 1, test_Xt_array.shape[1]))

history = model.fit(train_Xt_array, train_y_array, epochs=epochs, batch_size=batch_size, verbose=1,
                   validation_data=(test_Xt_array, test_y_array))
```


```python
Series_pred = np.argmax(model.predict(test_Xt_array, batch_size=batch_size, verbose = 0), axis = 1)

Series_actual = np.argmax(test_y_array, axis = 1)

classreport= classification_report(Series_actual, Series_pred, target_names = df_Yt.columns, digits = 4)
print(classreport)
```

## Checking for Overfitting


```python
[test_loss, test_acc] = model.evaluate(test_Xt_array, test_y_array)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
```


```python
#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
```
