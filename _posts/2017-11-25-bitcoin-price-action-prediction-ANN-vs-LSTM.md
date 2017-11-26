---
layout: post
title: "Techniques for Predicting Bitcoin Price Action"
desc: "According to Abernathy MacGregor, a strategic communications firm, as of 2014, more than 75 percent of trades on US stock exchanges are from automated trading systems. If you've ever wondered how they do it, this notebook and the one it references could serve as a quick introduction. "
tag: "Machine Learning"
author: "Sean Kelley"
thumb: "/img/blog/bitcoin.jpg"
date: 2017-11-25
---

## Feature Extraction

Take a look at the [original kernel](https://www.kaggle.com/marklam/a-neutral-network-to-read-btc-price-action) for a more detailed explanation of the philosophy behind the feature extraction

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

**The variables used as input are as follows:**

- **O**: Price at the start of snapshot which is fixed
- **C**: Price at the close of snapshot which may equal **O**
- **H**: Highest price recorded during the snapshot which may equal **C** and or **O**
- **L**: Lowest price recorded during the snapshot which may equal **C** and or **O**
- **WgtPx**, W: A derived price based on the ratio of value traded to volume traded ([further reading](https://en.wikipedia.org/wiki/Volume-weighted_average_price))

Hence:

> Change in WgtPX or **V[t] – V[t-1]= f(HO[t], LO[t], CO[t], WO[t])**,

where **HO**, **LO**, **CO** and **WO** are the relative distance of **H**, **L**, **C** and **W** from a fixed datum **O[t]**

As a hunch, I also included the day of the week and time of day as inputs to the network.

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




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table class="dataframe table table-striped">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HO</th>
      <th>LO</th>
      <th>CO</th>
      <th>WO</th>
      <th>day</th>
      <th>time</th>
    </tr>
    <tr>
      <th>Timestamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-10-19 19:56:00+00:00</th>
      <td>0.00</td>
      <td>-0.01</td>
      <td>0.00</td>
      <td>-0.000122</td>
      <td>3</td>
      <td>19.933333</td>
    </tr>
    <tr>
      <th>2017-10-19 19:57:00+00:00</th>
      <td>0.53</td>
      <td>0.00</td>
      <td>0.53</td>
      <td>0.397366</td>
      <td>3</td>
      <td>19.950000</td>
    </tr>
    <tr>
      <th>2017-10-19 19:58:00+00:00</th>
      <td>3.47</td>
      <td>-0.01</td>
      <td>3.47</td>
      <td>0.999951</td>
      <td>3</td>
      <td>19.966667</td>
    </tr>
    <tr>
      <th>2017-10-19 19:59:00+00:00</th>
      <td>0.05</td>
      <td>-1.09</td>
      <td>0.05</td>
      <td>-0.567248</td>
      <td>3</td>
      <td>19.983333</td>
    </tr>
    <tr>
      <th>2017-10-19 20:00:00+00:00</th>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.009412</td>
      <td>3</td>
      <td>20.000000</td>
    </tr>
  </tbody>
</table>
</div>




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
from sklearn.metrics import classification_report

ann_model = Sequential()
ann_model.add(Dense(32, activation = 'tanh', input_dim=6))
ann_model.add(Dropout(0.2))
ann_model.add(Dense(32, activation = 'tanh'))
ann_model.add(Dropout(0.1))
ann_model.add(Dense(32, activation = 'tanh'))
ann_model.add(Dropout(0.1))
ann_model.add(Dense(3, activation = 'softmax'))
# out shaped on df_Yt.shape[1]
ann_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

    Using TensorFlow backend.


## Training

Both models are trained on the data between Jan 1, 2015 and Dec 31, 2016.

The models are then validated using the data from Jan 1, 2017 to Oct 19, 2017.


```python
batch_size = 60*24 # Total 'blocks/snapshot' in a day
epochs = 100

train_Xt_array = df_Xt.loc['2015-1-1':'2016-12-31'].values
train_y_array = df_Yt.loc['2015-1-1':'2016-12-31'].values

#train_Xt_array = np.reshape(train_Xt_array, (train_Xt_array.shape[0], 1, train_Xt_array.shape[1]))

test_Xt_array = df_Xt.loc['2017-1-1':'2017-10-19'].values
test_y_array = df_Yt.loc['2017-1-1':'2017-10-19'].values

#test_Xt_array = np.reshape(test_Xt_array, (test_Xt_array.shape[0], 1, test_Xt_array.shape[1]))

ann_history = ann_model.fit(train_Xt_array, train_y_array, epochs=epochs, batch_size=batch_size, verbose=1,
                   validation_data=(test_Xt_array, test_y_array))
```

    Train on 1034775 samples, validate on 420183 samples
    Epoch 1/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.6868 - acc: 0.6919 - val_loss: 0.6072 - val_acc: 0.6612
    Epoch 2/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.5503 - acc: 0.6948 - val_loss: 0.5533 - val_acc: 0.7086
    Epoch 3/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.5186 - acc: 0.7006 - val_loss: 0.5187 - val_acc: 0.7210
    Epoch 4/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.5018 - acc: 0.7061 - val_loss: 0.5110 - val_acc: 0.7219
    Epoch 5/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4904 - acc: 0.7113 - val_loss: 0.4923 - val_acc: 0.7263
    Epoch 6/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4833 - acc: 0.7159 - val_loss: 0.5013 - val_acc: 0.7155
    Epoch 7/100
    1034775/1034775 [==============================] - 4s 4us/step - loss: 0.4782 - acc: 0.7195 - val_loss: 0.4859 - val_acc: 0.7267
    Epoch 8/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4746 - acc: 0.7224 - val_loss: 0.4822 - val_acc: 0.7297
    Epoch 9/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4715 - acc: 0.7250 - val_loss: 0.4785 - val_acc: 0.7298
    Epoch 10/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4678 - acc: 0.7269 - val_loss: 0.4713 - val_acc: 0.7349
    Epoch 11/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4632 - acc: 0.7283 - val_loss: 0.4579 - val_acc: 0.7398
    Epoch 12/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4581 - acc: 0.7310 - val_loss: 0.4529 - val_acc: 0.7404
    Epoch 13/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4542 - acc: 0.7330 - val_loss: 0.4444 - val_acc: 0.7548
    Epoch 14/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4515 - acc: 0.7352 - val_loss: 0.4469 - val_acc: 0.7526
    Epoch 15/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4494 - acc: 0.7353 - val_loss: 0.4482 - val_acc: 0.7501
    Epoch 16/100
    1034775/1034775 [==============================] - 4s 4us/step - loss: 0.4475 - acc: 0.7370 - val_loss: 0.4399 - val_acc: 0.7609
    Epoch 17/100
    1034775/1034775 [==============================] - 4s 4us/step - loss: 0.4454 - acc: 0.7378 - val_loss: 0.4431 - val_acc: 0.7561
    Epoch 18/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4436 - acc: 0.7392 - val_loss: 0.4378 - val_acc: 0.7652
    Epoch 19/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4424 - acc: 0.7399 - val_loss: 0.4384 - val_acc: 0.7605
    Epoch 20/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4412 - acc: 0.7402 - val_loss: 0.4419 - val_acc: 0.7572
    Epoch 21/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4404 - acc: 0.7403 - val_loss: 0.4365 - val_acc: 0.7631
    Epoch 22/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4395 - acc: 0.7407 - val_loss: 0.4360 - val_acc: 0.7646
    Epoch 23/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4385 - acc: 0.7415 - val_loss: 0.4348 - val_acc: 0.7661
    Epoch 24/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4379 - acc: 0.7418 - val_loss: 0.4367 - val_acc: 0.7621
    Epoch 25/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4370 - acc: 0.7419 - val_loss: 0.4299 - val_acc: 0.7715
    Epoch 26/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4364 - acc: 0.7428 - val_loss: 0.4307 - val_acc: 0.7704
    Epoch 27/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4358 - acc: 0.7431 - val_loss: 0.4353 - val_acc: 0.7648
    Epoch 28/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4355 - acc: 0.7432 - val_loss: 0.4293 - val_acc: 0.7715
    Epoch 29/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4353 - acc: 0.7432 - val_loss: 0.4401 - val_acc: 0.7601
    Epoch 30/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4348 - acc: 0.7432 - val_loss: 0.4331 - val_acc: 0.7658
    Epoch 31/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4346 - acc: 0.7434 - val_loss: 0.4343 - val_acc: 0.7640
    Epoch 32/100
    1034775/1034775 [==============================] - 4s 4us/step - loss: 0.4338 - acc: 0.7439 - val_loss: 0.4321 - val_acc: 0.7691
    Epoch 33/100
    1034775/1034775 [==============================] - 4s 4us/step - loss: 0.4337 - acc: 0.7438 - val_loss: 0.4317 - val_acc: 0.7657
    Epoch 34/100
    1034775/1034775 [==============================] - 4s 4us/step - loss: 0.4330 - acc: 0.7444 - val_loss: 0.4329 - val_acc: 0.7668
    Epoch 35/100
    1034775/1034775 [==============================] - 4s 4us/step - loss: 0.4329 - acc: 0.7440 - val_loss: 0.4275 - val_acc: 0.7693
    Epoch 36/100
    1034775/1034775 [==============================] - 4s 4us/step - loss: 0.4326 - acc: 0.7444 - val_loss: 0.4314 - val_acc: 0.7688
    Epoch 37/100
    1034775/1034775 [==============================] - 4s 4us/step - loss: 0.4321 - acc: 0.7450 - val_loss: 0.4319 - val_acc: 0.7689
    Epoch 38/100
    1034775/1034775 [==============================] - 4s 4us/step - loss: 0.4319 - acc: 0.7452 - val_loss: 0.4307 - val_acc: 0.7704
    Epoch 39/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4316 - acc: 0.7455 - val_loss: 0.4375 - val_acc: 0.7659
    Epoch 40/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4315 - acc: 0.7453 - val_loss: 0.4319 - val_acc: 0.7676
    Epoch 41/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4315 - acc: 0.7452 - val_loss: 0.4323 - val_acc: 0.7649
    Epoch 42/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4311 - acc: 0.7452 - val_loss: 0.4317 - val_acc: 0.7681
    Epoch 43/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4309 - acc: 0.7453 - val_loss: 0.4317 - val_acc: 0.7698
    Epoch 44/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4307 - acc: 0.7458 - val_loss: 0.4301 - val_acc: 0.7716
    Epoch 45/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4306 - acc: 0.7457 - val_loss: 0.4306 - val_acc: 0.7717
    Epoch 46/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4304 - acc: 0.7460 - val_loss: 0.4315 - val_acc: 0.7683
    Epoch 47/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4304 - acc: 0.7457 - val_loss: 0.4298 - val_acc: 0.7729
    Epoch 48/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4303 - acc: 0.7461 - val_loss: 0.4270 - val_acc: 0.7744
    Epoch 49/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4302 - acc: 0.7458 - val_loss: 0.4303 - val_acc: 0.7703
    Epoch 50/100
    1034775/1034775 [==============================] - 4s 4us/step - loss: 0.4300 - acc: 0.7462 - val_loss: 0.4326 - val_acc: 0.7663
    Epoch 51/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4300 - acc: 0.7462 - val_loss: 0.4336 - val_acc: 0.7688
    Epoch 52/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4296 - acc: 0.7463 - val_loss: 0.4259 - val_acc: 0.7742
    Epoch 53/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4297 - acc: 0.7464 - val_loss: 0.4302 - val_acc: 0.7726
    Epoch 54/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4295 - acc: 0.7467 - val_loss: 0.4321 - val_acc: 0.7700
    Epoch 55/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4300 - acc: 0.7461 - val_loss: 0.4262 - val_acc: 0.7744
    Epoch 56/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4291 - acc: 0.7471 - val_loss: 0.4290 - val_acc: 0.7717
    Epoch 57/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4293 - acc: 0.7467 - val_loss: 0.4306 - val_acc: 0.7715
    Epoch 58/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4294 - acc: 0.7464 - val_loss: 0.4284 - val_acc: 0.7738
    Epoch 59/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4292 - acc: 0.7470 - val_loss: 0.4350 - val_acc: 0.7582
    Epoch 60/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4295 - acc: 0.7467 - val_loss: 0.4277 - val_acc: 0.7745
    Epoch 61/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4288 - acc: 0.7470 - val_loss: 0.4268 - val_acc: 0.7747
    Epoch 62/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4290 - acc: 0.7469 - val_loss: 0.4256 - val_acc: 0.7768
    Epoch 63/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4292 - acc: 0.7465 - val_loss: 0.4309 - val_acc: 0.7727
    Epoch 64/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4290 - acc: 0.7467 - val_loss: 0.4309 - val_acc: 0.7731
    Epoch 65/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4289 - acc: 0.7471 - val_loss: 0.4323 - val_acc: 0.7680
    Epoch 66/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4288 - acc: 0.7469 - val_loss: 0.4222 - val_acc: 0.7812
    Epoch 67/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4288 - acc: 0.7469 - val_loss: 0.4283 - val_acc: 0.7740
    Epoch 68/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4289 - acc: 0.7465 - val_loss: 0.4307 - val_acc: 0.7708
    Epoch 69/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4287 - acc: 0.7471 - val_loss: 0.4284 - val_acc: 0.7747
    Epoch 70/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4289 - acc: 0.7470 - val_loss: 0.4284 - val_acc: 0.7743
    Epoch 71/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4287 - acc: 0.7473 - val_loss: 0.4341 - val_acc: 0.7650
    Epoch 72/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4285 - acc: 0.7471 - val_loss: 0.4322 - val_acc: 0.7688
    Epoch 73/100
    1034775/1034775 [==============================] - 4s 4us/step - loss: 0.4287 - acc: 0.7469 - val_loss: 0.4307 - val_acc: 0.7713
    Epoch 74/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4285 - acc: 0.7471 - val_loss: 0.4261 - val_acc: 0.7767
    Epoch 75/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4284 - acc: 0.7471 - val_loss: 0.4247 - val_acc: 0.7760
    Epoch 76/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4283 - acc: 0.7473 - val_loss: 0.4242 - val_acc: 0.7777
    Epoch 77/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4284 - acc: 0.7472 - val_loss: 0.4290 - val_acc: 0.7731
    Epoch 78/100
    1034775/1034775 [==============================] - 4s 4us/step - loss: 0.4285 - acc: 0.7475 - val_loss: 0.4273 - val_acc: 0.7759
    Epoch 79/100
    1034775/1034775 [==============================] - 4s 4us/step - loss: 0.4286 - acc: 0.7469 - val_loss: 0.4232 - val_acc: 0.7800
    Epoch 80/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4283 - acc: 0.7474 - val_loss: 0.4253 - val_acc: 0.7755
    Epoch 81/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4282 - acc: 0.7474 - val_loss: 0.4301 - val_acc: 0.7711
    Epoch 82/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4284 - acc: 0.7470 - val_loss: 0.4295 - val_acc: 0.7709
    Epoch 83/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4283 - acc: 0.7473 - val_loss: 0.4265 - val_acc: 0.7768
    Epoch 84/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4283 - acc: 0.7471 - val_loss: 0.4334 - val_acc: 0.7655
    Epoch 85/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4283 - acc: 0.7472 - val_loss: 0.4314 - val_acc: 0.7731
    Epoch 86/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4281 - acc: 0.7475 - val_loss: 0.4301 - val_acc: 0.7680
    Epoch 87/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4283 - acc: 0.7473 - val_loss: 0.4277 - val_acc: 0.7742
    Epoch 88/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4280 - acc: 0.7473 - val_loss: 0.4276 - val_acc: 0.7736
    Epoch 89/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4284 - acc: 0.7471 - val_loss: 0.4318 - val_acc: 0.7704
    Epoch 90/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4281 - acc: 0.7472 - val_loss: 0.4302 - val_acc: 0.7706
    Epoch 91/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4281 - acc: 0.7474 - val_loss: 0.4261 - val_acc: 0.7766
    Epoch 92/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4280 - acc: 0.7471 - val_loss: 0.4262 - val_acc: 0.7749
    Epoch 93/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4279 - acc: 0.7475 - val_loss: 0.4314 - val_acc: 0.7704
    Epoch 94/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4278 - acc: 0.7478 - val_loss: 0.4323 - val_acc: 0.7681
    Epoch 95/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4280 - acc: 0.7478 - val_loss: 0.4271 - val_acc: 0.7761
    Epoch 96/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4276 - acc: 0.7474 - val_loss: 0.4246 - val_acc: 0.7769
    Epoch 97/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4281 - acc: 0.7473 - val_loss: 0.4304 - val_acc: 0.7704
    Epoch 98/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4277 - acc: 0.7476 - val_loss: 0.4269 - val_acc: 0.7783
    Epoch 99/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4277 - acc: 0.7476 - val_loss: 0.4302 - val_acc: 0.7729
    Epoch 100/100
    1034775/1034775 [==============================] - 5s 4us/step - loss: 0.4281 - acc: 0.7473 - val_loss: 0.4294 - val_acc: 0.7707



```python
Series_pred = np.argmax(ann_model.predict(test_Xt_array, batch_size=batch_size, verbose = 0), axis = 1)

Series_actual = np.argmax(test_y_array, axis = 1)

classreport = classification_report(Series_actual, Series_pred, target_names = df_Yt.columns, digits = 4)
print(classreport)
```

                 precision    recall  f1-score   support

             UP     0.7616    0.8109    0.7855    103852
             DN     0.7397    0.5951    0.6595     81930
           FLAT     0.7833    0.8143    0.7985    234401

    avg / total     0.7694    0.7707    0.7682    420183



## Checking for Overfitting

I want to credit [this post](https://www.learnopencv.com/image-classification-using-feedforward-neural-network-in-keras/) for the strategy for plotting training/val loss.


```python
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
```


```python
[test_loss, test_acc] = ann_model.evaluate(test_Xt_array, test_y_array)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
```

    420183/420183 [==============================] - 8s 20us/step
    Evaluation result on Test Data : Loss = 0.42935008491756116, accuracy = 0.7707356080554967



```python
#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(ann_history.history['loss'],'r',linewidth=3.0)
plt.plot(ann_history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(ann_history.history['acc'],'r',linewidth=3.0)
plt.plot(ann_history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
```




    <matplotlib.text.Text at 0x7f7708b6c828>




![png](/img/blog/2017-11-25-bitcoin-price-action-prediction-ANN-vs-LSTM/ANN%20vs%20LTSM_14_1.png)



![png](/img/blog/2017-11-25-bitcoin-price-action-prediction-ANN-vs-LSTM/ANN%20vs%20LTSM_14_2.png)


## Basic LTSM Setup


```python
from keras.layers import LSTM

lstm_model = Sequential()
lstm_model.add(LSTM(units = 32, activation = 'tanh', input_shape=(None, 6)))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units = 3, activation = 'softmax'))
# out shaped on df_Yt.shape[1]
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```


```python
batch_size = 60*24 # Total 'blocks/snapshot' in a day
epochs = 100

# reshape data for LSTM network
train_Xt_array = np.reshape(train_Xt_array, (train_Xt_array.shape[0], 1, train_Xt_array.shape[1]))
test_Xt_array = np.reshape(test_Xt_array, (test_Xt_array.shape[0], 1, test_Xt_array.shape[1]))

lstm_history = lstm_model.fit(train_Xt_array, train_y_array, epochs=epochs, batch_size=batch_size, verbose=1,
                   validation_data=(test_Xt_array, test_y_array))
```

    Train on 1034775 samples, validate on 420183 samples
    Epoch 1/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.7472 - acc: 0.6840 - val_loss: 0.6537 - val_acc: 0.6861
    Epoch 2/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.6415 - acc: 0.7065 - val_loss: 0.6412 - val_acc: 0.6862
    Epoch 3/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.6135 - acc: 0.7114 - val_loss: 0.6030 - val_acc: 0.7214
    Epoch 4/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.5931 - acc: 0.7158 - val_loss: 0.5867 - val_acc: 0.7243
    Epoch 5/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.5789 - acc: 0.7183 - val_loss: 0.5761 - val_acc: 0.7198
    Epoch 6/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.5677 - acc: 0.7200 - val_loss: 0.5628 - val_acc: 0.7242
    Epoch 7/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.5588 - acc: 0.7203 - val_loss: 0.5543 - val_acc: 0.7279
    Epoch 8/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.5510 - acc: 0.7214 - val_loss: 0.5508 - val_acc: 0.7249
    Epoch 9/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.5441 - acc: 0.7224 - val_loss: 0.5418 - val_acc: 0.7308
    Epoch 10/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.5382 - acc: 0.7225 - val_loss: 0.5371 - val_acc: 0.7295
    Epoch 11/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.5321 - acc: 0.7233 - val_loss: 0.5295 - val_acc: 0.7321
    Epoch 12/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.5271 - acc: 0.7239 - val_loss: 0.5232 - val_acc: 0.7353
    Epoch 13/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.5224 - acc: 0.7243 - val_loss: 0.5197 - val_acc: 0.7363
    Epoch 14/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.5182 - acc: 0.7251 - val_loss: 0.5185 - val_acc: 0.7319
    Epoch 15/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.5140 - acc: 0.7254 - val_loss: 0.5118 - val_acc: 0.7366
    Epoch 16/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.5100 - acc: 0.7262 - val_loss: 0.5117 - val_acc: 0.7312
    Epoch 17/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.5062 - acc: 0.7270 - val_loss: 0.5071 - val_acc: 0.7352
    Epoch 18/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.5027 - acc: 0.7279 - val_loss: 0.5041 - val_acc: 0.7347
    Epoch 19/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4998 - acc: 0.7277 - val_loss: 0.5002 - val_acc: 0.7362
    Epoch 20/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4967 - acc: 0.7284 - val_loss: 0.4991 - val_acc: 0.7343
    Epoch 21/100
    1034775/1034775 [==============================] - 7s 6us/step - loss: 0.4939 - acc: 0.7298 - val_loss: 0.4959 - val_acc: 0.7391
    Epoch 22/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4909 - acc: 0.7298 - val_loss: 0.4937 - val_acc: 0.7391
    Epoch 23/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4884 - acc: 0.7307 - val_loss: 0.4906 - val_acc: 0.7403
    Epoch 24/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4857 - acc: 0.7309 - val_loss: 0.4879 - val_acc: 0.7430
    Epoch 25/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4835 - acc: 0.7321 - val_loss: 0.4871 - val_acc: 0.7402
    Epoch 26/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4815 - acc: 0.7326 - val_loss: 0.4873 - val_acc: 0.7430
    Epoch 27/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4793 - acc: 0.7333 - val_loss: 0.4836 - val_acc: 0.7444
    Epoch 28/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4775 - acc: 0.7337 - val_loss: 0.4828 - val_acc: 0.7453
    Epoch 29/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4756 - acc: 0.7340 - val_loss: 0.4808 - val_acc: 0.7457
    Epoch 30/100
    1034775/1034775 [==============================] - 7s 6us/step - loss: 0.4738 - acc: 0.7342 - val_loss: 0.4799 - val_acc: 0.7459
    Epoch 31/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4721 - acc: 0.7347 - val_loss: 0.4828 - val_acc: 0.7405
    Epoch 32/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4705 - acc: 0.7355 - val_loss: 0.4781 - val_acc: 0.7469
    Epoch 33/100
    1034775/1034775 [==============================] - 7s 6us/step - loss: 0.4690 - acc: 0.7364 - val_loss: 0.4774 - val_acc: 0.7477
    Epoch 34/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4678 - acc: 0.7362 - val_loss: 0.4738 - val_acc: 0.7509
    Epoch 35/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4664 - acc: 0.7372 - val_loss: 0.4766 - val_acc: 0.7453
    Epoch 36/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4648 - acc: 0.7376 - val_loss: 0.4725 - val_acc: 0.7494
    Epoch 37/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4637 - acc: 0.7379 - val_loss: 0.4682 - val_acc: 0.7535
    Epoch 38/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4621 - acc: 0.7390 - val_loss: 0.4694 - val_acc: 0.7510
    Epoch 39/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4609 - acc: 0.7392 - val_loss: 0.4718 - val_acc: 0.7470
    Epoch 40/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4601 - acc: 0.7394 - val_loss: 0.4666 - val_acc: 0.7523
    Epoch 41/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4589 - acc: 0.7403 - val_loss: 0.4679 - val_acc: 0.7504
    Epoch 42/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4580 - acc: 0.7402 - val_loss: 0.4701 - val_acc: 0.7452
    Epoch 43/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4570 - acc: 0.7404 - val_loss: 0.4634 - val_acc: 0.7521
    Epoch 44/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4560 - acc: 0.7411 - val_loss: 0.4651 - val_acc: 0.7488
    Epoch 45/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4549 - acc: 0.7418 - val_loss: 0.4624 - val_acc: 0.7524
    Epoch 46/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4545 - acc: 0.7419 - val_loss: 0.4649 - val_acc: 0.7484
    Epoch 47/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4533 - acc: 0.7426 - val_loss: 0.4600 - val_acc: 0.7535
    Epoch 48/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4525 - acc: 0.7425 - val_loss: 0.4610 - val_acc: 0.7524
    Epoch 49/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4517 - acc: 0.7431 - val_loss: 0.4600 - val_acc: 0.7513
    Epoch 50/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4511 - acc: 0.7430 - val_loss: 0.4581 - val_acc: 0.7541
    Epoch 51/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4504 - acc: 0.7437 - val_loss: 0.4586 - val_acc: 0.7517
    Epoch 52/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4498 - acc: 0.7435 - val_loss: 0.4590 - val_acc: 0.7525
    Epoch 53/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4491 - acc: 0.7438 - val_loss: 0.4590 - val_acc: 0.7517
    Epoch 54/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4484 - acc: 0.7438 - val_loss: 0.4558 - val_acc: 0.7540
    Epoch 55/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4481 - acc: 0.7443 - val_loss: 0.4584 - val_acc: 0.7515
    Epoch 56/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4472 - acc: 0.7451 - val_loss: 0.4545 - val_acc: 0.7559
    Epoch 57/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4469 - acc: 0.7447 - val_loss: 0.4563 - val_acc: 0.7540
    Epoch 58/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4462 - acc: 0.7448 - val_loss: 0.4532 - val_acc: 0.7565
    Epoch 59/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4458 - acc: 0.7449 - val_loss: 0.4586 - val_acc: 0.7500
    Epoch 60/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4452 - acc: 0.7458 - val_loss: 0.4523 - val_acc: 0.7571
    Epoch 61/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4447 - acc: 0.7457 - val_loss: 0.4548 - val_acc: 0.7527
    Epoch 62/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4445 - acc: 0.7457 - val_loss: 0.4532 - val_acc: 0.7541
    Epoch 63/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4439 - acc: 0.7461 - val_loss: 0.4558 - val_acc: 0.7523
    Epoch 64/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4437 - acc: 0.7459 - val_loss: 0.4536 - val_acc: 0.7534
    Epoch 65/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4431 - acc: 0.7461 - val_loss: 0.4511 - val_acc: 0.7551
    Epoch 66/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4430 - acc: 0.7464 - val_loss: 0.4515 - val_acc: 0.7554
    Epoch 67/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4425 - acc: 0.7463 - val_loss: 0.4519 - val_acc: 0.7541
    Epoch 68/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4419 - acc: 0.7463 - val_loss: 0.4485 - val_acc: 0.7588
    Epoch 69/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4417 - acc: 0.7465 - val_loss: 0.4518 - val_acc: 0.7552
    Epoch 70/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4416 - acc: 0.7463 - val_loss: 0.4500 - val_acc: 0.7578
    Epoch 71/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4413 - acc: 0.7466 - val_loss: 0.4500 - val_acc: 0.7564
    Epoch 72/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4409 - acc: 0.7469 - val_loss: 0.4495 - val_acc: 0.7567
    Epoch 73/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4407 - acc: 0.7473 - val_loss: 0.4499 - val_acc: 0.7564
    Epoch 74/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4404 - acc: 0.7473 - val_loss: 0.4495 - val_acc: 0.7555
    Epoch 75/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4401 - acc: 0.7472 - val_loss: 0.4470 - val_acc: 0.7598
    Epoch 76/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4396 - acc: 0.7475 - val_loss: 0.4493 - val_acc: 0.7563
    Epoch 77/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4396 - acc: 0.7475 - val_loss: 0.4498 - val_acc: 0.7560
    Epoch 78/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4391 - acc: 0.7480 - val_loss: 0.4484 - val_acc: 0.7552
    Epoch 79/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4388 - acc: 0.7473 - val_loss: 0.4491 - val_acc: 0.7552
    Epoch 80/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4388 - acc: 0.7476 - val_loss: 0.4440 - val_acc: 0.7611
    Epoch 81/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4384 - acc: 0.7478 - val_loss: 0.4459 - val_acc: 0.7586
    Epoch 82/100
    1034775/1034775 [==============================] - 5s 5us/step - loss: 0.4382 - acc: 0.7478 - val_loss: 0.4469 - val_acc: 0.7579
    Epoch 83/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4379 - acc: 0.7481 - val_loss: 0.4503 - val_acc: 0.7520
    Epoch 84/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4378 - acc: 0.7481 - val_loss: 0.4547 - val_acc: 0.7471
    Epoch 85/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4375 - acc: 0.7483 - val_loss: 0.4478 - val_acc: 0.7564
    Epoch 86/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4375 - acc: 0.7483 - val_loss: 0.4452 - val_acc: 0.7585
    Epoch 87/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4373 - acc: 0.7484 - val_loss: 0.4468 - val_acc: 0.7578
    Epoch 88/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4370 - acc: 0.7485 - val_loss: 0.4481 - val_acc: 0.7545
    Epoch 89/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4367 - acc: 0.7485 - val_loss: 0.4514 - val_acc: 0.7514
    Epoch 90/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4365 - acc: 0.7488 - val_loss: 0.4482 - val_acc: 0.7561
    Epoch 91/100
    1034775/1034775 [==============================] - 7s 6us/step - loss: 0.4364 - acc: 0.7484 - val_loss: 0.4502 - val_acc: 0.7525
    Epoch 92/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4362 - acc: 0.7485 - val_loss: 0.4515 - val_acc: 0.7502
    Epoch 93/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4360 - acc: 0.7488 - val_loss: 0.4474 - val_acc: 0.7562
    Epoch 94/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4359 - acc: 0.7488 - val_loss: 0.4488 - val_acc: 0.7536
    Epoch 95/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4354 - acc: 0.7491 - val_loss: 0.4432 - val_acc: 0.7596
    Epoch 96/100
    1034775/1034775 [==============================] - 6s 5us/step - loss: 0.4355 - acc: 0.7491 - val_loss: 0.4438 - val_acc: 0.7588
    Epoch 97/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4352 - acc: 0.7488 - val_loss: 0.4478 - val_acc: 0.7548
    Epoch 98/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4351 - acc: 0.7489 - val_loss: 0.4460 - val_acc: 0.7571
    Epoch 99/100
    1034775/1034775 [==============================] - 7s 7us/step - loss: 0.4348 - acc: 0.7489 - val_loss: 0.4458 - val_acc: 0.7585
    Epoch 100/100
    1034775/1034775 [==============================] - 6s 6us/step - loss: 0.4346 - acc: 0.7495 - val_loss: 0.4442 - val_acc: 0.7585



```python
Series_pred = np.argmax(lstm_model.predict(test_Xt_array, batch_size=batch_size, verbose = 0), axis = 1)

Series_actual = np.argmax(test_y_array, axis = 1)

classreport = classification_report(Series_actual, Series_pred, target_names = df_Yt.columns, digits = 4)
print(classreport)
```

                 precision    recall  f1-score   support

             UP     0.7631    0.7484    0.7557    103852
             DN     0.7287    0.5976    0.6567     81930
           FLAT     0.7647    0.8193    0.7911    234401

    avg / total     0.7573    0.7585    0.7561    420183



## Checking for Overfitting


```python
[test_loss, test_acc] = lstm_model.evaluate(test_Xt_array, test_y_array)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
```

    420183/420183 [==============================] - 9s 21us/step
    Evaluation result on Test Data : Loss = 0.4442229792565888, accuracy = 0.7585456812860615



```python
#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(lstm_history.history['loss'],'r',linewidth=3.0)
plt.plot(lstm_history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(lstm_history.history['acc'],'r',linewidth=3.0)
plt.plot(lstm_history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
```




    <matplotlib.text.Text at 0x7f7680e67fd0>




![png](/img/blog/2017-11-25-bitcoin-price-action-prediction-ANN-vs-LSTM/ANN%20vs%20LTSM_21_1.png)



![png](/img/blog/2017-11-25-bitcoin-price-action-prediction-ANN-vs-LSTM/ANN%20vs%20LTSM_21_2.png)


# Conclusion

Let's take a look at the plots side by side:


```python
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=[16, 12])

gs = gridspec.GridSpec(2, 2)

ann_loss = plt.subplot(gs[0, 0])
ann_loss.plot(ann_history.history['loss'],'r',linewidth=3.0)
ann_loss.plot(ann_history.history['val_loss'],'b',linewidth=3.0)
ann_loss.legend(['Training loss', 'Validation Loss'],fontsize=18)
ann_loss.set_xlabel('Epochs ',fontsize=16)
ann_loss.set_ylabel('Loss',fontsize=16)
ann_loss.set_ylim([0.4, 0.8])
ann_loss.set_title('ANN Loss Curves',fontsize=16)

ann_acc = plt.subplot(gs[1, 0])
ann_acc.plot(ann_history.history['acc'],'r',linewidth=3.0)
ann_acc.plot(ann_history.history['val_acc'],'b',linewidth=3.0)
ann_acc.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
ann_acc.set_xlabel('Epochs ',fontsize=16)
ann_acc.set_ylabel('Accuracy',fontsize=16)
ann_acc.set_ylim([0.64, 0.8])
ann_acc.set_title('ANN Accuracy Curves',fontsize=16)

lstm_loss = plt.subplot(gs[0, 1])
lstm_loss.plot(lstm_history.history['loss'],'r',linewidth=3.0)
lstm_loss.plot(lstm_history.history['val_loss'],'b',linewidth=3.0)
lstm_loss.legend(['Training loss', 'Validation Loss'],fontsize=18)
lstm_loss.set_xlabel('Epochs ',fontsize=16)
lstm_loss.set_ylabel('Loss',fontsize=16)
lstm_loss.set_ylim([0.4, 0.8])
lstm_loss.set_title('LSTM Loss Curves',fontsize=16)

lstm_acc = plt.subplot(gs[1, 1])
lstm_acc.plot(lstm_history.history['acc'],'r',linewidth=3.0)
lstm_acc.plot(lstm_history.history['val_acc'],'b',linewidth=3.0)
lstm_acc.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
lstm_acc.set_xlabel('Epochs ',fontsize=16)
lstm_acc.set_ylabel('Accuracy',fontsize=16)
lstm_acc.set_ylim([0.64, 0.8])
lstm_acc.set_title('LSTM Accuracy Curves',fontsize=16)

fig.add_subplot(ann_loss)
fig.add_subplot(ann_acc)
fig.add_subplot(lstm_loss)
fig.add_subplot(lstm_acc)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f767277ab00>




![png](/img/blog/2017-11-25-bitcoin-price-action-prediction-ANN-vs-LSTM/ANN%20vs%20LTSM_23_1.png)


100 Epochs is admittedly a rather short training time but the results are rather promissing.

From the loss and accuracy curves, it seems that the ANN reaches a minimum quickly and after 50 epochs there aren't signigicant gains. In contrast, the LSTM seems to still have a bit to go until it reaches its minimum.

Again, as stated by the original kernel, the structure of the networks is rather arbitrary and more for demonstration.

I'm pretty new to the price action prediction and even the machine learning field so there could certainly be errors. Let me know if you have any corrections, suggestions or comments!
