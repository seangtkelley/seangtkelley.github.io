---
layout: post
title: "Basic Algo-Trading Part 2 - Backtrading"
desc: "Welcome back to part 2 of my series on basic algo-trading! In this post, I actually use the prediction function we created previously to create a strategy and backtrade it on a mainstream stock! Let's dive in..."
tag: "Algorithmic Trading"
author: "Sean Kelley"
thumb: "/img/blog/2018-08-05-algo-trading-pt1/thumbnail.jpg"
date: 2018-08-15
---

Welcome back to part 2 of my series on basic algo-trading! In this post, I actually use the prediction function we created previously to create a strategy and backtrade it on a mainstream stock! Let's dive in...

For backtrading, we could potentially use the pandas data source from the last part but there's something even better. Thanks to the beautiful world of open source software, we can use the [backtrader](https://www.backtrader.com/) python library created by Daniel Rodriguez.

From its site, backtrader is...

    A feature-rich Python framework for backtesting and trading. Backtrader allows you to focus on writing reusable trading strategies, indicators and analyzers instead of having to spend time building infrastructure.
    
Sound like exactly what we need.

Backtrader has a very impressive range of functionality, but for this example, we going to stick to the most important features that we need to test our predictions.

Like always, let's snag the libraries we'll need.


```python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Import the backtrader platform
import backtrader as bt

import pandas as pd
from pandas_datareader import data, wb
import datetime as dt
from fbprophet import Prophet
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
```

First, we are going to need to create a strategy. The most important function in our strategy will be the `next` function. The `next` function is called when Backtrader makes one step through the data. For example, if you have minute-to-minute prices, each call of `next` will represent the passing of one minute. Since we are going to use Yahoo Finance API data, each call of our `next` function will represent the passing of one day.

Within next we will do some basic sanity checks before making a prediction. Next comes our basic strategy.

You will notice what was `make_predictions` has now become `get_prophet_moves`. Although the name is now hopefully more intuitive, it still has the functionality. 

In our strategy, we will do a dead simple analysis:

1. if the predicted movement is mostly positive: buy
2. if the predicted movement is mostly negative: sell

Anyone with experience will cringe at this but it is quick and easy to implement for an example.


```python
class TestStrategy(bt.Strategy):
    
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # keep array of dates and closes
        self.date_array = []
        self.close_array = []

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.datas[0].close[0])
        
        # append date and close to arrays
        self.date_array.append(self.datas[0].datetime.date(0))
        self.close_array.append(self.datas[0].close[0])
        
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return
        
        # make sure we have a decent amount of data
        if len(self.date_array) < 90:
            return
        
        # only invest once a week
        if len(self.date_array) % 5 != 0:
            return
        
        # get predictions
        max_move, expected_move, min_move = self.get_prophet_moves(7, False)
        
        # if the predicted movement is mostly positive, buy
        if max_move > 0 and abs(max_move) > abs(min_move):
            self.log('BUY CREATE, %.2f' % self.datas[0].close[0])

            # Keep track of the created order to avoid a 2nd order
            self.order = self.buy()
        
        # if the predicted movement is mostly negative, sell
        elif min_move < 0 and abs(min_move) > abs(max_move):
            
            # make sure we have some stock to sell
            if self.position:
                self.log('SELL CREATE, %.2f' % self.datas[0].close[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()
                
    def get_prophet_moves(self, daysOut=7, showCharts=False):
        # create stock dataframe for prophet
        stock_df = pd.DataFrame({
            'ds': self.date_array,
            'y': self.close_array
        })

        # fit data using prophet model
        m = Prophet()
        m.fit(stock_df)

        # create future dates
        future_prices = m.make_future_dataframe(periods=365)

        # predict prices
        forecast = m.predict(future_prices)

        # view results
        if showCharts:
            fig = m.plot(forecast)
            ax1 = fig.add_subplot(111)
            ax1.set_title("Stock Price Forecast", fontsize=16)
            ax1.set_xlabel("Date", fontsize=12)
            ax1.set_ylabel("Close Price", fontsize=12)

            fig2 = m.plot_components(forecast)
            plt.show()

        # calculate predicted returns
        end_of_period = self.datas[0].datetime.date(0) + dt.timedelta(days=daysOut)

        future_close_max = forecast.loc[forecast.ds > end_of_period].iloc[0].yhat_upper
        future_close_expected = forecast.loc[forecast.ds > end_of_period].iloc[0].yhat
        future_close_min = forecast.loc[forecast.ds > end_of_period].iloc[0].yhat_lower

        # calculate percent changes based on predictions
        max_move = (future_close_max - self.datas[0].close[0])/self.datas[0].close[0]
        expected_move = (future_close_expected - self.datas[0].close[0])/self.datas[0].close[0]
        min_move = (future_close_min - self.datas[0].close[0])/self.datas[0].close[0]

        return (max_move, expected_move, min_move)
```

Next comes the fun part...backtrading!

Hopefully, the comments are explanatory enough but I just want to take the time to commend the dev/team behind Backtrader because the library is very readible.

You'll notice that we are going to backtrade this strategy on Wal-Mart (WMT). Although not the most seasonal stock, it's likely affected by shoppers' habits during the year. For our date range, we'll start the algorithm ten years ago from August 1st. Being casual investors, we'll start with a modest $1000. Finally, for our own sake, we'll pretend Robinhood existed ten years ago and set the commision for our broker to 0%.

It really is as simple as the code below. You can change a wide range of parameters easily and test your algorithm in a variety of situations.


```python
# Create a cerebro entity
cerebro = bt.Cerebro()

# Add a strategy
cerebro.addstrategy(TestStrategy)

# Create a Data Feed
data = bt.feeds.YahooFinanceData(
    dataname='WMT',
    fromdate=dt.datetime(2008, 8, 1),
    todate=dt.datetime(2018, 8, 1),
    reverse=False)

cerebro.adddata(data)

# Set our desired cash start
cerebro.broker.setcash(1000.0)# Add the Data Feed to Cerebro

# Add a FixedSize sizer according to the stake
cerebro.addsizer(bt.sizers.FixedSize, stake=1)

# Set the commission
cerebro.broker.setcommission(commission=0.0)

# Print out the starting conditions
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Run over everything
cerebro.run()

# Print out the final result
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
```

    Starting Portfolio Value: 1000.00


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2008-08-01, Close, 44.87
    2008-08-04, Close, 45.39
    2008-08-05, Close, 46.88
    2008-08-06, Close, 47.20
    2008-08-07, Close, 44.25
    2008-08-08, Close, 44.95
    2008-08-11, Close, 45.50
    2008-08-12, Close, 46.03
    2008-08-13, Close, 45.15
    2008-08-14, Close, 45.32
    2008-08-15, Close, 46.31
    2008-08-18, Close, 45.89
    2008-08-19, Close, 45.40
    2008-08-20, Close, 45.52
    2008-08-21, Close, 45.63
    2008-08-22, Close, 46.36
    2008-08-25, Close, 45.67
    2008-08-26, Close, 46.02
    2008-08-27, Close, 46.25
    2008-08-28, Close, 46.71
    2008-08-29, Close, 46.08
    2008-09-02, Close, 46.53
    2008-09-03, Close, 46.64
    2008-09-04, Close, 46.63
    2008-09-05, Close, 47.38
    2008-09-08, Close, 48.36
    2008-09-09, Close, 47.68
    2008-09-10, Close, 48.38
    2008-09-11, Close, 49.27
    2008-09-12, Close, 48.68
    2008-09-15, Close, 48.07
    2008-09-16, Close, 48.47
    2008-09-17, Close, 46.52
    2008-09-18, Close, 47.96
    2008-09-19, Close, 46.57
    2008-09-22, Close, 45.94
    2008-09-23, Close, 45.55
    2008-09-24, Close, 45.96
    2008-09-25, Close, 46.90
    2008-09-26, Close, 47.36
    2008-09-29, Close, 45.59
    2008-09-30, Close, 46.72
    2008-10-01, Close, 46.54
    2008-10-02, Close, 45.90
    2008-10-03, Close, 46.59
    2008-10-06, Close, 45.16
    2008-10-07, Close, 42.78
    2008-10-08, Close, 42.55
    2008-10-09, Close, 40.09
    2008-10-10, Close, 39.74
    2008-10-13, Close, 42.51
    2008-10-14, Close, 42.46
    2008-10-15, Close, 39.04
    2008-10-16, Close, 42.61
    2008-10-17, Close, 41.94
    2008-10-20, Close, 42.46
    2008-10-21, Close, 41.86
    2008-10-22, Close, 40.77
    2008-10-23, Close, 41.15
    2008-10-24, Close, 40.09
    2008-10-27, Close, 38.74
    2008-10-28, Close, 43.03
    2008-10-29, Close, 42.92
    2008-10-30, Close, 42.71
    2008-10-31, Close, 43.53
    2008-11-03, Close, 43.66
    2008-11-04, Close, 43.78
    2008-11-05, Close, 42.22
    2008-11-06, Close, 41.72
    2008-11-07, Close, 42.43
    2008-11-10, Close, 43.04
    2008-11-11, Close, 42.71
    2008-11-12, Close, 41.04
    2008-11-13, Close, 42.85
    2008-11-14, Close, 41.12
    2008-11-17, Close, 40.41
    2008-11-18, Close, 41.12
    2008-11-19, Close, 39.78
    2008-11-20, Close, 39.52
    2008-11-21, Close, 41.28
    2008-11-24, Close, 41.16
    2008-11-25, Close, 42.65
    2008-11-26, Close, 44.22
    2008-11-28, Close, 43.59
    2008-12-01, Close, 41.35
    2008-12-02, Close, 41.69
    2008-12-03, Close, 42.42
    2008-12-04, Close, 42.99
    2008-12-05, Close, 45.41
    2008-12-08, Close, 44.90


    /home/sean/.virtualenvs/ml/lib/python3.6/site-packages/pystan/misc.py:399: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      elif np.issubdtype(np.asarray(v).dtype, float):
    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2008-12-09, Close, 43.53
    2008-12-10, Close, 43.10
    2008-12-11, Close, 42.92
    2008-12-12, Close, 42.80
    2008-12-15, Close, 42.86


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2008-12-15, BUY CREATE, 42.86
    2008-12-16, BUY EXECUTED, Price: 55.01, Cost: 55.01, Comm 0.00
    2008-12-16, Close, 43.27
    2008-12-17, Close, 43.24
    2008-12-18, Close, 43.41
    2008-12-19, Close, 43.67
    2008-12-22, Close, 43.86


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2008-12-22, BUY CREATE, 43.86
    2008-12-23, BUY EXECUTED, Price: 56.15, Cost: 56.15, Comm 0.00
    2008-12-23, Close, 43.31
    2008-12-24, Close, 43.43
    2008-12-26, Close, 43.36
    2008-12-29, Close, 43.17
    2008-12-30, Close, 43.13


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2008-12-30, BUY CREATE, 43.13
    2008-12-31, BUY EXECUTED, Price: 55.27, Cost: 55.27, Comm 0.00
    2008-12-31, Close, 43.92
    2009-01-02, Close, 44.79
    2009-01-05, Close, 44.28
    2009-01-06, Close, 43.89
    2009-01-07, Close, 43.51


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-01-07, BUY CREATE, 43.51
    2009-01-08, BUY EXECUTED, Price: 51.31, Cost: 51.31, Comm 0.00
    2009-01-08, Close, 40.25
    2009-01-09, Close, 40.41
    2009-01-12, Close, 40.26
    2009-01-13, Close, 40.83
    2009-01-14, Close, 40.39


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-01-14, BUY CREATE, 40.39
    2009-01-15, BUY EXECUTED, Price: 51.56, Cost: 51.56, Comm 0.00
    2009-01-15, Close, 40.23
    2009-01-16, Close, 40.39
    2009-01-20, Close, 39.61
    2009-01-21, Close, 38.50
    2009-01-22, Close, 38.28


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-01-22, BUY CREATE, 38.28
    2009-01-23, BUY EXECUTED, Price: 48.09, Cost: 48.09, Comm 0.00
    2009-01-23, Close, 37.88
    2009-01-26, Close, 38.07
    2009-01-27, Close, 38.22
    2009-01-28, Close, 38.17
    2009-01-29, Close, 37.49


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-01-29, BUY CREATE, 37.49
    2009-01-30, BUY EXECUTED, Price: 48.00, Cost: 48.00, Comm 0.00
    2009-01-30, Close, 36.91
    2009-02-02, Close, 36.48
    2009-02-03, Close, 37.45
    2009-02-04, Close, 36.37
    2009-02-05, Close, 38.04


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-02-05, SELL CREATE, 38.04
    2009-02-06, SELL EXECUTED, Price: 48.84, Cost: 52.20, Comm 0.00
    2009-02-06, Close, 38.88
    2009-02-09, Close, 38.61
    2009-02-10, Close, 37.38
    2009-02-11, Close, 37.78
    2009-02-12, Close, 37.70


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-02-12, SELL CREATE, 37.70
    2009-02-13, SELL EXECUTED, Price: 47.75, Cost: 52.20, Comm 0.00
    2009-02-13, Close, 36.45
    2009-02-17, Close, 37.79
    2009-02-18, Close, 39.17
    2009-02-19, Close, 39.52
    2009-02-20, Close, 39.19


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-02-20, SELL CREATE, 39.19
    2009-02-23, SELL EXECUTED, Price: 50.35, Cost: 52.20, Comm 0.00
    2009-02-23, Close, 38.29
    2009-02-24, Close, 39.18
    2009-02-25, Close, 38.55
    2009-02-26, Close, 37.80
    2009-02-27, Close, 38.57


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-02-27, SELL CREATE, 38.57
    2009-03-02, SELL EXECUTED, Price: 48.81, Cost: 52.20, Comm 0.00
    2009-03-02, Close, 37.63
    2009-03-03, Close, 37.12
    2009-03-04, Close, 37.99
    2009-03-05, Close, 38.97
    2009-03-06, Close, 38.32


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-03-06, SELL CREATE, 38.32
    2009-03-09, SELL EXECUTED, Price: 48.56, Cost: 52.20, Comm 0.00
    2009-03-09, Close, 37.22
    2009-03-10, Close, 38.13
    2009-03-11, Close, 37.39
    2009-03-12, Close, 38.56
    2009-03-13, Close, 38.75


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-03-13, SELL CREATE, 38.75
    2009-03-16, SELL EXECUTED, Price: 49.34, Cost: 52.20, Comm 0.00
    2009-03-16, Close, 38.44
    2009-03-17, Close, 39.39
    2009-03-18, Close, 39.74
    2009-03-19, Close, 39.35
    2009-03-20, Close, 39.07


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-03-20, SELL CREATE, 39.07
    2009-03-23, SELL EXECUTED, Price: 50.28, Cost: 52.20, Comm 0.00
    2009-03-23, OPERATION PROFIT, GROSS -21.46, NET -21.46
    2009-03-23, Close, 40.56
    2009-03-24, Close, 40.24
    2009-03-25, Close, 40.71
    2009-03-26, Close, 41.56
    2009-03-27, Close, 41.41


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-03-30, Close, 40.78
    2009-03-31, Close, 41.04
    2009-04-01, Close, 41.61
    2009-04-02, Close, 42.26
    2009-04-03, Close, 42.38


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-04-06, Close, 42.09
    2009-04-07, Close, 41.27
    2009-04-08, Close, 41.45
    2009-04-09, Close, 39.91
    2009-04-13, Close, 40.60


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-04-13, BUY CREATE, 40.60
    2009-04-14, BUY EXECUTED, Price: 51.20, Cost: 51.20, Comm 0.00
    2009-04-14, Close, 40.27
    2009-04-15, Close, 40.41
    2009-04-16, Close, 40.00
    2009-04-17, Close, 39.55
    2009-04-20, Close, 38.82


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-04-20, BUY CREATE, 38.82
    2009-04-21, BUY EXECUTED, Price: 49.66, Cost: 49.66, Comm 0.00
    2009-04-21, Close, 39.26
    2009-04-22, Close, 38.57
    2009-04-23, Close, 38.49
    2009-04-24, Close, 37.71
    2009-04-27, Close, 38.22


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-04-27, BUY CREATE, 38.22
    2009-04-28, BUY EXECUTED, Price: 48.28, Cost: 48.28, Comm 0.00
    2009-04-28, Close, 38.18
    2009-04-29, Close, 39.74
    2009-04-30, Close, 39.71
    2009-05-01, Close, 39.43
    2009-05-04, Close, 40.05


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-05-04, SELL CREATE, 40.05
    2009-05-05, SELL EXECUTED, Price: 50.67, Cost: 49.71, Comm 0.00
    2009-05-05, Close, 39.75
    2009-05-06, Close, 39.00
    2009-05-07, Close, 39.30
    2009-05-08, Close, 39.50
    2009-05-11, Close, 39.89


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-05-11, BUY CREATE, 39.89
    2009-05-12, BUY EXECUTED, Price: 50.88, Cost: 50.88, Comm 0.00
    2009-05-12, Close, 40.10
    2009-05-13, Close, 39.63
    2009-05-14, Close, 38.89
    2009-05-15, Close, 38.14
    2009-05-18, Close, 39.54


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-05-18, BUY CREATE, 39.54
    2009-05-19, BUY EXECUTED, Price: 49.92, Cost: 49.92, Comm 0.00
    2009-05-19, Close, 39.10
    2009-05-20, Close, 38.76
    2009-05-21, Close, 38.90
    2009-05-22, Close, 39.01
    2009-05-26, Close, 39.60


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-05-26, BUY CREATE, 39.60
    2009-05-27, BUY EXECUTED, Price: 50.24, Cost: 50.24, Comm 0.00
    2009-05-27, Close, 39.06
    2009-05-28, Close, 39.25
    2009-05-29, Close, 39.40
    2009-06-01, Close, 40.07
    2009-06-02, Close, 39.55


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-06-02, BUY CREATE, 39.55
    2009-06-03, BUY EXECUTED, Price: 49.75, Cost: 49.75, Comm 0.00
    2009-06-03, Close, 40.30
    2009-06-04, Close, 40.29
    2009-06-05, Close, 40.45
    2009-06-08, Close, 40.24
    2009-06-09, Close, 40.09


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-06-09, SELL CREATE, 40.09
    2009-06-10, SELL EXECUTED, Price: 50.81, Cost: 50.04, Comm 0.00
    2009-06-10, Close, 39.65
    2009-06-11, Close, 39.06
    2009-06-12, Close, 39.48
    2009-06-15, Close, 38.38
    2009-06-16, Close, 38.22


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-06-16, BUY CREATE, 38.22
    2009-06-17, BUY EXECUTED, Price: 48.30, Cost: 48.30, Comm 0.00
    2009-06-17, Close, 38.46
    2009-06-18, Close, 38.56
    2009-06-19, Close, 38.15
    2009-06-22, Close, 38.49
    2009-06-23, Close, 38.30


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-06-23, BUY CREATE, 38.30
    2009-06-24, BUY EXECUTED, Price: 48.50, Cost: 48.50, Comm 0.00
    2009-06-24, Close, 38.42
    2009-06-25, Close, 38.93
    2009-06-26, Close, 38.52
    2009-06-29, Close, 38.62
    2009-06-30, Close, 38.37


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-06-30, BUY CREATE, 38.37
    2009-07-01, BUY EXECUTED, Price: 48.55, Cost: 48.55, Comm 0.00
    2009-07-01, Close, 38.31
    2009-07-02, Close, 37.85
    2009-07-06, Close, 37.80
    2009-07-07, Close, 37.89
    2009-07-08, Close, 38.31


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-07-08, BUY CREATE, 38.31
    2009-07-09, BUY EXECUTED, Price: 48.80, Cost: 48.80, Comm 0.00
    2009-07-09, Close, 38.08
    2009-07-10, Close, 37.68
    2009-07-13, Close, 37.88
    2009-07-14, Close, 38.12
    2009-07-15, Close, 38.45


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-07-15, SELL CREATE, 38.45
    2009-07-16, SELL EXECUTED, Price: 48.35, Cost: 49.37, Comm 0.00
    2009-07-16, Close, 38.42
    2009-07-17, Close, 38.41
    2009-07-20, Close, 38.67
    2009-07-21, Close, 38.70
    2009-07-22, Close, 38.94


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-07-22, SELL CREATE, 38.94
    2009-07-23, SELL EXECUTED, Price: 49.18, Cost: 49.37, Comm 0.00
    2009-07-23, Close, 38.62
    2009-07-24, Close, 38.76
    2009-07-27, Close, 38.79
    2009-07-28, Close, 38.75
    2009-07-29, Close, 39.10


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-07-29, SELL CREATE, 39.10
    2009-07-30, SELL EXECUTED, Price: 49.55, Cost: 49.37, Comm 0.00
    2009-07-30, Close, 39.59
    2009-07-31, Close, 39.51
    2009-08-03, Close, 39.48
    2009-08-04, Close, 39.48
    2009-08-05, Close, 38.97


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-08-05, SELL CREATE, 38.97
    2009-08-06, SELL EXECUTED, Price: 49.25, Cost: 49.37, Comm 0.00
    2009-08-06, Close, 38.79
    2009-08-07, Close, 39.04
    2009-08-10, Close, 39.38
    2009-08-11, Close, 39.63
    2009-08-12, Close, 40.23


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-08-12, SELL CREATE, 40.23
    2009-08-13, SELL EXECUTED, Price: 51.83, Cost: 49.37, Comm 0.00
    2009-08-13, Close, 41.32
    2009-08-14, Close, 41.24
    2009-08-17, Close, 41.07
    2009-08-18, Close, 40.90
    2009-08-19, Close, 41.15


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-08-19, SELL CREATE, 41.15
    2009-08-20, SELL EXECUTED, Price: 51.60, Cost: 49.37, Comm 0.00
    2009-08-20, Close, 41.18
    2009-08-21, Close, 40.90
    2009-08-24, Close, 41.05
    2009-08-25, Close, 41.15
    2009-08-26, Close, 41.25


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-08-26, SELL CREATE, 41.25
    2009-08-27, SELL EXECUTED, Price: 51.65, Cost: 49.37, Comm 0.00
    2009-08-27, Close, 40.81
    2009-08-28, Close, 40.72
    2009-08-31, Close, 40.51
    2009-09-01, Close, 40.59
    2009-09-02, Close, 40.55


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-09-02, SELL CREATE, 40.55
    2009-09-03, SELL EXECUTED, Price: 51.15, Cost: 49.37, Comm 0.00
    2009-09-03, Close, 41.20
    2009-09-04, Close, 41.16
    2009-09-08, Close, 40.93
    2009-09-09, Close, 40.70
    2009-09-10, Close, 40.64


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-09-10, SELL CREATE, 40.64
    2009-09-11, SELL EXECUTED, Price: 50.99, Cost: 49.37, Comm 0.00
    2009-09-11, OPERATION PROFIT, GROSS 10.95, NET 10.95
    2009-09-11, Close, 40.39
    2009-09-14, Close, 40.12
    2009-09-15, Close, 39.76
    2009-09-16, Close, 39.85
    2009-09-17, Close, 39.79


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-09-17, BUY CREATE, 39.79
    2009-09-18, BUY EXECUTED, Price: 50.00, Cost: 50.00, Comm 0.00
    2009-09-18, Close, 39.91
    2009-09-21, Close, 40.54
    2009-09-22, Close, 40.61
    2009-09-23, Close, 40.14
    2009-09-24, Close, 40.38


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-09-24, BUY CREATE, 40.38
    2009-09-25, BUY EXECUTED, Price: 50.40, Cost: 50.40, Comm 0.00
    2009-09-25, Close, 39.40
    2009-09-28, Close, 39.42
    2009-09-29, Close, 39.21
    2009-09-30, Close, 39.09
    2009-10-01, Close, 39.02


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-10-01, BUY CREATE, 39.02
    2009-10-02, BUY EXECUTED, Price: 48.89, Cost: 48.89, Comm 0.00
    2009-10-02, Close, 39.09
    2009-10-05, Close, 39.07
    2009-10-06, Close, 39.40
    2009-10-07, Close, 39.41
    2009-10-08, Close, 39.61


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-10-08, BUY CREATE, 39.61
    2009-10-09, BUY EXECUTED, Price: 49.81, Cost: 49.81, Comm 0.00
    2009-10-09, Close, 39.80
    2009-10-12, Close, 39.51
    2009-10-13, Close, 40.09
    2009-10-14, Close, 39.97
    2009-10-15, Close, 40.58


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-10-15, SELL CREATE, 40.58
    2009-10-16, SELL EXECUTED, Price: 50.80, Cost: 49.78, Comm 0.00
    2009-10-16, Close, 40.79
    2009-10-19, Close, 41.32
    2009-10-20, Close, 41.17
    2009-10-21, Close, 40.32
    2009-10-22, Close, 40.20


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-10-22, BUY CREATE, 40.20
    2009-10-23, BUY EXECUTED, Price: 50.68, Cost: 50.68, Comm 0.00
    2009-10-23, Close, 40.17
    2009-10-26, Close, 39.69
    2009-10-27, Close, 39.72
    2009-10-28, Close, 39.74
    2009-10-29, Close, 40.14


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-10-29, BUY CREATE, 40.14
    2009-10-30, BUY EXECUTED, Price: 50.39, Cost: 50.39, Comm 0.00
    2009-10-30, Close, 39.56
    2009-11-02, Close, 40.04
    2009-11-03, Close, 39.74
    2009-11-04, Close, 40.12
    2009-11-05, Close, 40.84


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-11-05, SELL CREATE, 40.84
    2009-11-06, SELL EXECUTED, Price: 51.03, Cost: 50.08, Comm 0.00
    2009-11-06, Close, 40.81
    2009-11-09, Close, 41.41
    2009-11-10, Close, 41.66
    2009-11-11, Close, 42.18
    2009-11-12, Close, 42.40


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-11-12, SELL CREATE, 42.40
    2009-11-13, SELL EXECUTED, Price: 53.28, Cost: 50.08, Comm 0.00
    2009-11-13, Close, 42.37
    2009-11-16, Close, 42.34
    2009-11-17, Close, 42.73
    2009-11-18, Close, 43.12
    2009-11-19, Close, 43.43


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-11-19, SELL CREATE, 43.43
    2009-11-20, SELL EXECUTED, Price: 54.53, Cost: 50.08, Comm 0.00
    2009-11-20, Close, 43.23
    2009-11-23, Close, 43.55
    2009-11-24, Close, 43.68
    2009-11-25, Close, 43.77
    2009-11-27, Close, 43.51


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-11-27, SELL CREATE, 43.51
    2009-11-30, SELL EXECUTED, Price: 54.53, Cost: 50.08, Comm 0.00
    2009-11-30, Close, 43.44
    2009-12-01, Close, 43.60
    2009-12-02, Close, 43.46
    2009-12-03, Close, 43.36
    2009-12-04, Close, 43.20


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-12-04, SELL CREATE, 43.20
    2009-12-07, SELL EXECUTED, Price: 54.14, Cost: 50.08, Comm 0.00
    2009-12-07, OPERATION PROFIT, GROSS 18.14, NET 18.14
    2009-12-07, Close, 43.75
    2009-12-08, Close, 43.33
    2009-12-09, Close, 43.28
    2009-12-10, Close, 43.77
    2009-12-11, Close, 43.74


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-12-14, Close, 43.28
    2009-12-15, Close, 43.21
    2009-12-16, Close, 42.68
    2009-12-17, Close, 42.23
    2009-12-18, Close, 42.30


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-12-21, Close, 42.74
    2009-12-22, Close, 42.69
    2009-12-23, Close, 42.68
    2009-12-24, Close, 42.90
    2009-12-28, Close, 43.21


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2009-12-28, BUY CREATE, 43.21
    2009-12-29, BUY EXECUTED, Price: 53.97, Cost: 53.97, Comm 0.00
    2009-12-29, Close, 43.31
    2009-12-30, Close, 43.46
    2009-12-31, Close, 42.78
    2010-01-04, Close, 43.41
    2010-01-05, Close, 42.97


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-01-05, BUY CREATE, 42.97
    2010-01-06, BUY EXECUTED, Price: 53.50, Cost: 53.50, Comm 0.00
    2010-01-06, Close, 42.88
    2010-01-07, Close, 42.90
    2010-01-08, Close, 42.68
    2010-01-11, Close, 43.39
    2010-01-12, Close, 43.81


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-01-12, SELL CREATE, 43.81
    2010-01-13, SELL EXECUTED, Price: 54.79, Cost: 53.73, Comm 0.00
    2010-01-13, Close, 44.03
    2010-01-14, Close, 43.39
    2010-01-15, Close, 42.96
    2010-01-19, Close, 43.25
    2010-01-20, Close, 43.11


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-01-20, BUY CREATE, 43.11
    2010-01-21, BUY EXECUTED, Price: 54.07, Cost: 54.07, Comm 0.00
    2010-01-21, Close, 42.36
    2010-01-22, Close, 42.37
    2010-01-25, Close, 42.32
    2010-01-26, Close, 42.91
    2010-01-27, Close, 42.74


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-01-27, BUY CREATE, 42.74
    2010-01-28, BUY EXECUTED, Price: 53.38, Cost: 53.38, Comm 0.00
    2010-01-28, Close, 42.11
    2010-01-29, Close, 42.76
    2010-02-01, Close, 42.80
    2010-02-02, Close, 42.81
    2010-02-03, Close, 43.44


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-02-03, BUY CREATE, 43.44
    2010-02-04, BUY EXECUTED, Price: 53.88, Cost: 53.88, Comm 0.00
    2010-02-04, Close, 42.40
    2010-02-05, Close, 42.78
    2010-02-08, Close, 42.36
    2010-02-09, Close, 42.62
    2010-02-10, Close, 42.61


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-02-10, BUY CREATE, 42.61
    2010-02-11, BUY EXECUTED, Price: 53.13, Cost: 53.13, Comm 0.00
    2010-02-11, Close, 42.48
    2010-02-12, Close, 42.34
    2010-02-16, Close, 42.87
    2010-02-17, Close, 43.27
    2010-02-18, Close, 42.80


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-02-18, BUY CREATE, 42.80
    2010-02-19, BUY EXECUTED, Price: 53.19, Cost: 53.19, Comm 0.00
    2010-02-19, Close, 42.81
    2010-02-22, Close, 43.09
    2010-02-23, Close, 42.92
    2010-02-24, Close, 43.16
    2010-02-25, Close, 43.34


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-02-25, BUY CREATE, 43.34
    2010-02-26, BUY EXECUTED, Price: 54.22, Cost: 54.22, Comm 0.00
    2010-02-26, Close, 43.28
    2010-03-01, Close, 43.14
    2010-03-02, Close, 42.89
    2010-03-03, Close, 42.95
    2010-03-04, Close, 43.19


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-03-04, BUY CREATE, 43.19
    2010-03-05, BUY EXECUTED, Price: 53.97, Cost: 53.97, Comm 0.00
    2010-03-05, Close, 43.33
    2010-03-08, Close, 43.34
    2010-03-09, Close, 43.27
    2010-03-10, Close, 43.17
    2010-03-11, Close, 43.44


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-03-11, BUY CREATE, 43.44
    2010-03-12, BUY EXECUTED, Price: 54.16, Cost: 54.16, Comm 0.00
    2010-03-12, Close, 43.38
    2010-03-15, Close, 44.61
    2010-03-16, Close, 45.07
    2010-03-17, Close, 45.01
    2010-03-18, Close, 45.03


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-03-18, SELL CREATE, 45.03
    2010-03-19, SELL EXECUTED, Price: 55.34, Cost: 53.75, Comm 0.00
    2010-03-19, Close, 44.54
    2010-03-22, Close, 44.77
    2010-03-23, Close, 44.99
    2010-03-24, Close, 44.74
    2010-03-25, Close, 44.76


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-03-25, SELL CREATE, 44.76
    2010-03-26, SELL EXECUTED, Price: 55.61, Cost: 53.75, Comm 0.00
    2010-03-26, Close, 44.68
    2010-03-29, Close, 44.86
    2010-03-30, Close, 45.00
    2010-03-31, Close, 44.75
    2010-04-01, Close, 44.66


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-04-01, SELL CREATE, 44.66
    2010-04-05, SELL EXECUTED, Price: 55.70, Cost: 53.75, Comm 0.00
    2010-04-05, Close, 44.66
    2010-04-06, Close, 44.70
    2010-04-07, Close, 44.49
    2010-04-08, Close, 44.58
    2010-04-09, Close, 44.33


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-04-09, SELL CREATE, 44.33
    2010-04-12, SELL EXECUTED, Price: 55.02, Cost: 53.75, Comm 0.00
    2010-04-12, Close, 44.29
    2010-04-13, Close, 44.04
    2010-04-14, Close, 43.98
    2010-04-15, Close, 43.57
    2010-04-16, Close, 43.55


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-04-16, BUY CREATE, 43.55
    2010-04-19, BUY EXECUTED, Price: 54.00, Cost: 54.00, Comm 0.00
    2010-04-19, Close, 43.78
    2010-04-20, Close, 43.88
    2010-04-21, Close, 43.84
    2010-04-22, Close, 43.86
    2010-04-23, Close, 43.89


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-04-23, SELL CREATE, 43.89
    2010-04-26, SELL EXECUTED, Price: 54.53, Cost: 53.79, Comm 0.00
    2010-04-26, Close, 43.50
    2010-04-27, Close, 43.50
    2010-04-28, Close, 43.15
    2010-04-29, Close, 43.22
    2010-04-30, Close, 43.17


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-04-30, BUY CREATE, 43.17
    2010-05-03, BUY EXECUTED, Price: 53.88, Cost: 53.88, Comm 0.00
    2010-05-03, Close, 43.26
    2010-05-04, Close, 43.48
    2010-05-05, Close, 44.08
    2010-05-06, Close, 42.84
    2010-05-07, Close, 42.18


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-05-07, BUY CREATE, 42.18
    2010-05-10, BUY EXECUTED, Price: 53.06, Cost: 53.06, Comm 0.00
    2010-05-10, Close, 42.32
    2010-05-11, Close, 42.22
    2010-05-12, Close, 42.49
    2010-05-13, Close, 42.42
    2010-05-14, Close, 42.19


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-05-14, BUY CREATE, 42.19
    2010-05-17, BUY EXECUTED, Price: 52.41, Cost: 52.41, Comm 0.00
    2010-05-17, Close, 42.69
    2010-05-18, Close, 43.48
    2010-05-19, Close, 42.94
    2010-05-20, Close, 41.53
    2010-05-21, Close, 41.59


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-05-21, BUY CREATE, 41.59
    2010-05-24, BUY EXECUTED, Price: 51.04, Cost: 51.04, Comm 0.00
    2010-05-24, Close, 41.29
    2010-05-25, Close, 40.70
    2010-05-26, Close, 40.49
    2010-05-27, Close, 41.04
    2010-05-28, Close, 40.93


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-05-28, BUY CREATE, 40.93
    2010-06-01, BUY EXECUTED, Price: 50.80, Cost: 50.80, Comm 0.00
    2010-06-01, Close, 41.22
    2010-06-02, Close, 41.87
    2010-06-03, Close, 41.87
    2010-06-04, Close, 40.80
    2010-06-07, Close, 41.08


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-06-07, BUY CREATE, 41.08
    2010-06-08, BUY EXECUTED, Price: 50.79, Cost: 50.79, Comm 0.00
    2010-06-08, Close, 41.11
    2010-06-09, Close, 41.28
    2010-06-10, Close, 41.47
    2010-06-11, Close, 41.17
    2010-06-14, Close, 41.48


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-06-14, BUY CREATE, 41.48
    2010-06-15, BUY EXECUTED, Price: 51.20, Cost: 51.20, Comm 0.00
    2010-06-15, Close, 41.81
    2010-06-16, Close, 41.27
    2010-06-17, Close, 41.62
    2010-06-18, Close, 41.73
    2010-06-21, Close, 41.30


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-06-21, BUY CREATE, 41.30
    2010-06-22, BUY EXECUTED, Price: 51.07, Cost: 51.07, Comm 0.00
    2010-06-22, Close, 41.03
    2010-06-23, Close, 41.13
    2010-06-24, Close, 40.50
    2010-06-25, Close, 39.51
    2010-06-28, Close, 40.13


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-06-28, BUY CREATE, 40.13
    2010-06-29, BUY EXECUTED, Price: 49.17, Cost: 49.17, Comm 0.00
    2010-06-29, Close, 39.59
    2010-06-30, Close, 38.92
    2010-07-01, Close, 39.13
    2010-07-02, Close, 38.86
    2010-07-06, Close, 39.32


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-07-06, BUY CREATE, 39.32
    2010-07-07, BUY EXECUTED, Price: 48.66, Cost: 48.66, Comm 0.00
    2010-07-07, Close, 39.60
    2010-07-08, Close, 39.81
    2010-07-09, Close, 40.02
    2010-07-12, Close, 40.58
    2010-07-13, Close, 40.92


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-07-13, SELL CREATE, 40.92
    2010-07-14, SELL EXECUTED, Price: 50.05, Cost: 52.07, Comm 0.00
    2010-07-14, Close, 40.76
    2010-07-15, Close, 40.81
    2010-07-16, Close, 40.21
    2010-07-19, Close, 40.09
    2010-07-20, Close, 41.19


    INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-07-20, SELL CREATE, 41.19
    2010-07-21, SELL EXECUTED, Price: 51.00, Cost: 52.07, Comm 0.00
    2010-07-21, Close, 40.76
    2010-07-22, Close, 41.17
    2010-07-23, Close, 41.83
    2010-07-26, Close, 41.39
    2010-07-27, Close, 41.26


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-07-27, SELL CREATE, 41.26
    2010-07-28, SELL EXECUTED, Price: 50.80, Cost: 52.07, Comm 0.00
    2010-07-28, Close, 41.39
    2010-07-29, Close, 41.34
    2010-07-30, Close, 41.44
    2010-08-02, Close, 41.62
    2010-08-03, Close, 41.52


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-08-03, SELL CREATE, 41.52
    2010-08-04, SELL EXECUTED, Price: 51.04, Cost: 52.07, Comm 0.00
    2010-08-04, Close, 41.77
    2010-08-05, Close, 41.79
    2010-08-06, Close, 41.93
    2010-08-09, Close, 42.15
    2010-08-10, Close, 42.28


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-08-10, SELL CREATE, 42.28
    2010-08-11, SELL EXECUTED, Price: 51.62, Cost: 52.07, Comm 0.00
    2010-08-11, Close, 41.54
    2010-08-12, Close, 41.06
    2010-08-13, Close, 41.04
    2010-08-16, Close, 41.05
    2010-08-17, Close, 41.54


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-08-17, BUY CREATE, 41.54
    2010-08-18, BUY EXECUTED, Price: 51.00, Cost: 51.00, Comm 0.00
    2010-08-18, Close, 41.41
    2010-08-19, Close, 40.76
    2010-08-20, Close, 40.89
    2010-08-23, Close, 41.64
    2010-08-24, Close, 41.77


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-08-24, BUY CREATE, 41.77
    2010-08-25, BUY EXECUTED, Price: 51.25, Cost: 51.25, Comm 0.00
    2010-08-25, Close, 41.98
    2010-08-26, Close, 41.50
    2010-08-27, Close, 41.53
    2010-08-30, Close, 41.16
    2010-08-31, Close, 40.83


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-08-31, BUY CREATE, 40.83
    2010-09-01, BUY EXECUTED, Price: 50.49, Cost: 50.49, Comm 0.00
    2010-09-01, Close, 41.69
    2010-09-02, Close, 42.15
    2010-09-03, Close, 42.38
    2010-09-07, Close, 42.23
    2010-09-08, Close, 42.20


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-09-08, SELL CREATE, 42.20
    2010-09-09, SELL EXECUTED, Price: 52.06, Cost: 51.80, Comm 0.00
    2010-09-09, Close, 42.27
    2010-09-10, Close, 42.32
    2010-09-13, Close, 42.51
    2010-09-14, Close, 42.88
    2010-09-15, Close, 43.04


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-09-15, SELL CREATE, 43.04
    2010-09-16, SELL EXECUTED, Price: 52.66, Cost: 51.80, Comm 0.00
    2010-09-16, Close, 43.28
    2010-09-17, Close, 43.16
    2010-09-20, Close, 43.60
    2010-09-21, Close, 43.62
    2010-09-22, Close, 43.82


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-09-22, SELL CREATE, 43.82
    2010-09-23, SELL EXECUTED, Price: 53.73, Cost: 51.80, Comm 0.00
    2010-09-23, Close, 43.69
    2010-09-24, Close, 44.04
    2010-09-27, Close, 43.55
    2010-09-28, Close, 43.82
    2010-09-29, Close, 43.44


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-09-29, SELL CREATE, 43.44
    2010-09-30, SELL EXECUTED, Price: 53.53, Cost: 51.80, Comm 0.00
    2010-09-30, Close, 43.58
    2010-10-01, Close, 43.45
    2010-10-04, Close, 43.62
    2010-10-05, Close, 43.97
    2010-10-06, Close, 44.43


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-10-06, SELL CREATE, 44.43
    2010-10-07, SELL EXECUTED, Price: 54.66, Cost: 51.80, Comm 0.00
    2010-10-07, Close, 44.26
    2010-10-08, Close, 44.30
    2010-10-11, Close, 44.47
    2010-10-12, Close, 43.91
    2010-10-13, Close, 43.82


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-10-13, SELL CREATE, 43.82
    2010-10-14, SELL EXECUTED, Price: 53.87, Cost: 51.80, Comm 0.00
    2010-10-14, Close, 43.36
    2010-10-15, Close, 43.44
    2010-10-18, Close, 43.78
    2010-10-19, Close, 43.42
    2010-10-20, Close, 43.54


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-10-20, SELL CREATE, 43.54
    2010-10-21, SELL EXECUTED, Price: 53.69, Cost: 51.80, Comm 0.00
    2010-10-21, Close, 44.00
    2010-10-22, Close, 44.02
    2010-10-25, Close, 43.93
    2010-10-26, Close, 44.43
    2010-10-27, Close, 43.87


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-10-27, BUY CREATE, 43.87
    2010-10-28, BUY EXECUTED, Price: 54.19, Cost: 54.19, Comm 0.00
    2010-10-28, Close, 44.04
    2010-10-29, Close, 44.11
    2010-11-01, Close, 44.22
    2010-11-02, Close, 44.61
    2010-11-03, Close, 44.71


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-11-03, BUY CREATE, 44.71
    2010-11-04, BUY EXECUTED, Price: 55.00, Cost: 55.00, Comm 0.00
    2010-11-04, Close, 45.08
    2010-11-05, Close, 44.95
    2010-11-08, Close, 44.71
    2010-11-09, Close, 44.83
    2010-11-10, Close, 44.39


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-11-10, BUY CREATE, 44.39
    2010-11-11, BUY EXECUTED, Price: 54.52, Cost: 54.52, Comm 0.00
    2010-11-11, Close, 44.25
    2010-11-12, Close, 44.08
    2010-11-15, Close, 43.93
    2010-11-16, Close, 44.18
    2010-11-17, Close, 43.78


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-11-17, BUY CREATE, 43.78
    2010-11-18, BUY EXECUTED, Price: 54.14, Cost: 54.14, Comm 0.00
    2010-11-18, Close, 43.95
    2010-11-19, Close, 44.29
    2010-11-22, Close, 44.28
    2010-11-23, Close, 43.70
    2010-11-24, Close, 43.98


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-11-24, BUY CREATE, 43.98
    2010-11-26, BUY EXECUTED, Price: 53.69, Cost: 53.69, Comm 0.00
    2010-11-26, Close, 43.76
    2010-11-29, Close, 43.85
    2010-11-30, Close, 44.04
    2010-12-01, Close, 44.54
    2010-12-02, Close, 44.58


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-12-02, BUY CREATE, 44.58
    2010-12-03, BUY EXECUTED, Price: 54.63, Cost: 54.63, Comm 0.00
    2010-12-03, Close, 44.48
    2010-12-06, Close, 44.37
    2010-12-07, Close, 44.86
    2010-12-08, Close, 44.62
    2010-12-09, Close, 44.49


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-12-09, BUY CREATE, 44.49
    2010-12-10, BUY EXECUTED, Price: 54.31, Cost: 54.31, Comm 0.00
    2010-12-10, Close, 44.44
    2010-12-13, Close, 44.39
    2010-12-14, Close, 44.58
    2010-12-15, Close, 44.40
    2010-12-16, Close, 44.73


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-12-16, BUY CREATE, 44.73
    2010-12-17, BUY EXECUTED, Price: 54.68, Cost: 54.68, Comm 0.00
    2010-12-17, Close, 44.55
    2010-12-20, Close, 44.03
    2010-12-21, Close, 43.93
    2010-12-22, Close, 43.65
    2010-12-23, Close, 43.89


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-12-23, BUY CREATE, 43.89
    2010-12-27, BUY EXECUTED, Price: 53.56, Cost: 53.56, Comm 0.00
    2010-12-27, Close, 43.86
    2010-12-28, Close, 44.00
    2010-12-29, Close, 44.28
    2010-12-30, Close, 44.27
    2010-12-31, Close, 44.16


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2010-12-31, SELL CREATE, 44.16
    2011-01-03, SELL EXECUTED, Price: 54.23, Cost: 53.30, Comm 0.00
    2011-01-03, Close, 44.67
    2011-01-04, Close, 44.84
    2011-01-05, Close, 44.55
    2011-01-06, Close, 44.18
    2011-01-07, Close, 44.28


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-01-07, SELL CREATE, 44.28
    2011-01-10, SELL EXECUTED, Price: 53.65, Cost: 53.30, Comm 0.00
    2011-01-10, Close, 43.99
    2011-01-11, Close, 44.45
    2011-01-12, Close, 44.91
    2011-01-13, Close, 44.86
    2011-01-14, Close, 44.88


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-01-14, SELL CREATE, 44.88
    2011-01-18, SELL EXECUTED, Price: 55.11, Cost: 53.30, Comm 0.00
    2011-01-18, Close, 45.15
    2011-01-19, Close, 45.06
    2011-01-20, Close, 45.84
    2011-01-21, Close, 45.63
    2011-01-24, Close, 45.89


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-01-24, SELL CREATE, 45.89
    2011-01-25, SELL EXECUTED, Price: 56.12, Cost: 53.30, Comm 0.00
    2011-01-25, Close, 46.88
    2011-01-26, Close, 46.93
    2011-01-27, Close, 47.14
    2011-01-28, Close, 46.42
    2011-01-31, Close, 45.91


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-01-31, SELL CREATE, 45.91
    2011-02-01, SELL EXECUTED, Price: 56.37, Cost: 53.30, Comm 0.00
    2011-02-01, Close, 46.12
    2011-02-02, Close, 45.74
    2011-02-03, Close, 45.79
    2011-02-04, Close, 45.88
    2011-02-07, Close, 45.91


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-02-07, SELL CREATE, 45.91
    2011-02-08, SELL EXECUTED, Price: 56.10, Cost: 53.30, Comm 0.00
    2011-02-08, Close, 46.16
    2011-02-09, Close, 46.45
    2011-02-10, Close, 45.52
    2011-02-11, Close, 45.60
    2011-02-14, Close, 44.87


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-02-14, BUY CREATE, 44.87
    2011-02-15, BUY EXECUTED, Price: 54.78, Cost: 54.78, Comm 0.00
    2011-02-15, Close, 44.99
    2011-02-16, Close, 44.66
    2011-02-17, Close, 44.83
    2011-02-18, Close, 45.34
    2011-02-22, Close, 43.94


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-02-22, BUY CREATE, 43.94
    2011-02-23, BUY EXECUTED, Price: 53.46, Cost: 53.46, Comm 0.00
    2011-02-23, Close, 43.42
    2011-02-24, Close, 42.65
    2011-02-25, Close, 42.37
    2011-02-28, Close, 42.56
    2011-03-01, Close, 42.63


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-03-01, BUY CREATE, 42.63
    2011-03-02, BUY EXECUTED, Price: 52.27, Cost: 52.27, Comm 0.00
    2011-03-02, Close, 42.55
    2011-03-03, Close, 42.58
    2011-03-04, Close, 42.63
    2011-03-07, Close, 42.59
    2011-03-08, Close, 42.94


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-03-08, BUY CREATE, 42.94
    2011-03-09, BUY EXECUTED, Price: 52.27, Cost: 52.27, Comm 0.00
    2011-03-09, Close, 43.43
    2011-03-10, Close, 43.41
    2011-03-11, Close, 43.36
    2011-03-14, Close, 43.14
    2011-03-15, Close, 42.92


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-03-15, BUY CREATE, 42.92
    2011-03-16, BUY EXECUTED, Price: 51.52, Cost: 51.52, Comm 0.00
    2011-03-16, Close, 42.36
    2011-03-17, Close, 42.36
    2011-03-18, Close, 42.48
    2011-03-21, Close, 42.81
    2011-03-22, Close, 42.87


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-03-22, BUY CREATE, 42.87
    2011-03-23, BUY EXECUTED, Price: 51.85, Cost: 51.85, Comm 0.00
    2011-03-23, Close, 42.58
    2011-03-24, Close, 43.36
    2011-03-25, Close, 43.16
    2011-03-28, Close, 43.03
    2011-03-29, Close, 43.09


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-03-29, BUY CREATE, 43.09
    2011-03-30, BUY EXECUTED, Price: 52.53, Cost: 52.53, Comm 0.00
    2011-03-30, Close, 43.17
    2011-03-31, Close, 42.92
    2011-04-01, Close, 42.98
    2011-04-04, Close, 43.41
    2011-04-05, Close, 43.48


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-04-05, BUY CREATE, 43.48
    2011-04-06, BUY EXECUTED, Price: 52.67, Cost: 52.67, Comm 0.00
    2011-04-06, Close, 43.68
    2011-04-07, Close, 43.70
    2011-04-08, Close, 43.32
    2011-04-11, Close, 43.55
    2011-04-12, Close, 44.13


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-04-12, SELL CREATE, 44.13
    2011-04-13, SELL EXECUTED, Price: 53.72, Cost: 53.00, Comm 0.00
    2011-04-13, Close, 44.22
    2011-04-14, Close, 44.11
    2011-04-15, Close, 44.15
    2011-04-18, Close, 43.95
    2011-04-19, Close, 43.99


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-04-19, SELL CREATE, 43.99
    2011-04-20, SELL EXECUTED, Price: 53.75, Cost: 53.00, Comm 0.00
    2011-04-20, Close, 44.27
    2011-04-21, Close, 44.18
    2011-04-25, Close, 44.00
    2011-04-26, Close, 44.45
    2011-04-27, Close, 44.87


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-04-27, SELL CREATE, 44.87
    2011-04-28, SELL EXECUTED, Price: 54.29, Cost: 53.00, Comm 0.00
    2011-04-28, Close, 45.09
    2011-04-29, Close, 45.33
    2011-05-02, Close, 45.38
    2011-05-03, Close, 45.73
    2011-05-04, Close, 45.65


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-05-04, SELL CREATE, 45.65
    2011-05-05, SELL EXECUTED, Price: 55.15, Cost: 53.00, Comm 0.00
    2011-05-05, Close, 45.41
    2011-05-06, Close, 45.36
    2011-05-09, Close, 45.43
    2011-05-10, Close, 45.79
    2011-05-11, Close, 45.79


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-05-11, SELL CREATE, 45.79
    2011-05-12, SELL EXECUTED, Price: 55.24, Cost: 53.00, Comm 0.00
    2011-05-12, Close, 46.25
    2011-05-13, Close, 46.25
    2011-05-16, Close, 46.53
    2011-05-17, Close, 46.10
    2011-05-18, Close, 45.80


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-05-18, SELL CREATE, 45.80
    2011-05-19, SELL EXECUTED, Price: 55.23, Cost: 53.00, Comm 0.00
    2011-05-19, Close, 46.05
    2011-05-20, Close, 45.89
    2011-05-23, Close, 45.83
    2011-05-24, Close, 45.47
    2011-05-25, Close, 45.28


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-05-25, SELL CREATE, 45.28
    2011-05-26, SELL EXECUTED, Price: 54.53, Cost: 53.00, Comm 0.00
    2011-05-26, Close, 45.33
    2011-05-27, Close, 45.40
    2011-05-31, Close, 45.83
    2011-06-01, Close, 45.07
    2011-06-02, Close, 44.44


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-06-02, SELL CREATE, 44.44
    2011-06-03, SELL EXECUTED, Price: 53.15, Cost: 53.00, Comm 0.00
    2011-06-03, Close, 44.54
    2011-06-06, Close, 44.62
    2011-06-07, Close, 44.68
    2011-06-08, Close, 44.56
    2011-06-09, Close, 44.50


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-06-09, SELL CREATE, 44.50
    2011-06-10, SELL EXECUTED, Price: 53.67, Cost: 53.00, Comm 0.00
    2011-06-10, Close, 43.76
    2011-06-13, Close, 43.67
    2011-06-14, Close, 43.91
    2011-06-15, Close, 43.42
    2011-06-16, Close, 43.85


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-06-16, SELL CREATE, 43.85
    2011-06-17, SELL EXECUTED, Price: 53.06, Cost: 53.00, Comm 0.00
    2011-06-17, Close, 43.84
    2011-06-20, Close, 44.02
    2011-06-21, Close, 44.23
    2011-06-22, Close, 44.00
    2011-06-23, Close, 44.23


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-06-23, SELL CREATE, 44.23
    2011-06-24, SELL EXECUTED, Price: 53.19, Cost: 53.00, Comm 0.00
    2011-06-24, Close, 43.50
    2011-06-27, Close, 43.40
    2011-06-28, Close, 43.60
    2011-06-29, Close, 43.69
    2011-06-30, Close, 44.10


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-06-30, SELL CREATE, 44.10
    2011-07-01, SELL EXECUTED, Price: 53.19, Cost: 53.00, Comm 0.00
    2011-07-01, Close, 44.41
    2011-07-05, Close, 44.31
    2011-07-06, Close, 44.59
    2011-07-07, Close, 45.23
    2011-07-08, Close, 44.88


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-07-08, SELL CREATE, 44.88
    2011-07-11, SELL EXECUTED, Price: 53.82, Cost: 53.00, Comm 0.00
    2011-07-11, Close, 44.71
    2011-07-12, Close, 44.77
    2011-07-13, Close, 44.83
    2011-07-14, Close, 44.51
    2011-07-15, Close, 44.51


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-07-15, SELL CREATE, 44.51
    2011-07-18, SELL EXECUTED, Price: 53.41, Cost: 53.00, Comm 0.00
    2011-07-18, Close, 44.25
    2011-07-19, Close, 44.79
    2011-07-20, Close, 44.73
    2011-07-21, Close, 45.21
    2011-07-22, Close, 45.25


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-07-22, SELL CREATE, 45.25
    2011-07-25, SELL EXECUTED, Price: 54.03, Cost: 53.00, Comm 0.00
    2011-07-25, Close, 44.79
    2011-07-26, Close, 44.48
    2011-07-27, Close, 44.20
    2011-07-28, Close, 43.98
    2011-07-29, Close, 43.75


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-07-29, SELL CREATE, 43.75
    2011-08-01, SELL EXECUTED, Price: 52.79, Cost: 53.00, Comm 0.00
    2011-08-01, Close, 43.67
    2011-08-02, Close, 42.89
    2011-08-03, Close, 42.56
    2011-08-04, Close, 41.58
    2011-08-05, Close, 42.20


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-08-05, BUY CREATE, 42.20
    2011-08-08, BUY EXECUTED, Price: 50.80, Cost: 50.80, Comm 0.00
    2011-08-08, Close, 40.60
    2011-08-09, Close, 42.18
    2011-08-10, Close, 40.47
    2011-08-11, Close, 41.57
    2011-08-12, Close, 41.59


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-08-12, BUY CREATE, 41.59
    2011-08-15, BUY EXECUTED, Price: 49.88, Cost: 49.88, Comm 0.00
    2011-08-15, Close, 41.78
    2011-08-16, Close, 43.40
    2011-08-17, Close, 43.09
    2011-08-18, Close, 43.30
    2011-08-19, Close, 43.72


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-08-19, SELL CREATE, 43.72
    2011-08-22, SELL EXECUTED, Price: 52.24, Cost: 51.23, Comm 0.00
    2011-08-22, Close, 43.63
    2011-08-23, Close, 44.48
    2011-08-24, Close, 44.62
    2011-08-25, Close, 44.06
    2011-08-26, Close, 44.22


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-08-26, SELL CREATE, 44.22
    2011-08-29, SELL EXECUTED, Price: 53.11, Cost: 51.23, Comm 0.00
    2011-08-29, Close, 44.47
    2011-08-30, Close, 44.16
    2011-08-31, Close, 44.47
    2011-09-01, Close, 44.01
    2011-09-02, Close, 43.50


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-09-02, BUY CREATE, 43.50
    2011-09-06, BUY EXECUTED, Price: 51.48, Cost: 51.48, Comm 0.00
    2011-09-06, Close, 43.20
    2011-09-07, Close, 43.82
    2011-09-08, Close, 43.65
    2011-09-09, Close, 42.94
    2011-09-12, Close, 43.32


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-09-12, BUY CREATE, 43.32
    2011-09-13, BUY EXECUTED, Price: 51.79, Cost: 51.79, Comm 0.00
    2011-09-13, Close, 43.13
    2011-09-14, Close, 43.64
    2011-09-15, Close, 43.90
    2011-09-16, Close, 44.01
    2011-09-19, Close, 43.85


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-09-19, BUY CREATE, 43.85
    2011-09-20, BUY EXECUTED, Price: 52.44, Cost: 52.44, Comm 0.00
    2011-09-20, Close, 43.71
    2011-09-21, Close, 42.90
    2011-09-22, Close, 42.03
    2011-09-23, Close, 42.47
    2011-09-26, Close, 43.33


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-09-26, BUY CREATE, 43.33
    2011-09-27, BUY EXECUTED, Price: 52.36, Cost: 52.36, Comm 0.00
    2011-09-27, Close, 43.50
    2011-09-28, Close, 42.89
    2011-09-29, Close, 43.41
    2011-09-30, Close, 43.39
    2011-10-03, Close, 43.44


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-10-03, SELL CREATE, 43.44
    2011-10-04, SELL EXECUTED, Price: 51.74, Cost: 51.86, Comm 0.00
    2011-10-04, Close, 44.21
    2011-10-05, Close, 44.01
    2011-10-06, Close, 44.10
    2011-10-07, Close, 44.89
    2011-10-10, Close, 45.82


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-10-10, SELL CREATE, 45.82
    2011-10-11, SELL EXECUTED, Price: 54.78, Cost: 51.86, Comm 0.00
    2011-10-11, Close, 45.74
    2011-10-12, Close, 46.15
    2011-10-13, Close, 46.00
    2011-10-14, Close, 46.36
    2011-10-17, Close, 45.79


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-10-17, SELL CREATE, 45.79
    2011-10-18, SELL EXECUTED, Price: 54.94, Cost: 51.86, Comm 0.00
    2011-10-18, Close, 46.72
    2011-10-19, Close, 47.02
    2011-10-20, Close, 47.12
    2011-10-21, Close, 47.58
    2011-10-24, Close, 47.47


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-10-24, SELL CREATE, 47.47
    2011-10-25, SELL EXECUTED, Price: 56.66, Cost: 51.86, Comm 0.00
    2011-10-25, Close, 47.41
    2011-10-26, Close, 47.96
    2011-10-27, Close, 48.33
    2011-10-28, Close, 47.78
    2011-10-31, Close, 47.42


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-10-31, SELL CREATE, 47.42
    2011-11-01, SELL EXECUTED, Price: 55.82, Cost: 51.86, Comm 0.00
    2011-11-01, OPERATION PROFIT, GROSS 57.68, NET 57.68
    2011-11-01, Close, 47.01
    2011-11-02, Close, 47.53
    2011-11-03, Close, 48.00
    2011-11-04, Close, 48.07
    2011-11-07, Close, 48.44


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-11-08, Close, 49.59
    2011-11-09, Close, 48.53
    2011-11-10, Close, 48.60
    2011-11-11, Close, 49.49
    2011-11-14, Close, 49.23


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-11-15, Close, 48.04
    2011-11-16, Close, 47.38
    2011-11-17, Close, 47.42
    2011-11-18, Close, 47.84
    2011-11-21, Close, 47.37


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-11-21, BUY CREATE, 47.37
    2011-11-22, BUY EXECUTED, Price: 56.56, Cost: 56.56, Comm 0.00
    2011-11-22, Close, 47.53
    2011-11-23, Close, 47.35
    2011-11-25, Close, 47.56
    2011-11-28, Close, 47.86
    2011-11-29, Close, 48.63


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-11-29, SELL CREATE, 48.63
    2011-11-30, SELL EXECUTED, Price: 58.76, Cost: 56.56, Comm 0.00
    2011-11-30, OPERATION PROFIT, GROSS 2.20, NET 2.20
    2011-11-30, Close, 49.24
    2011-12-01, Close, 49.00
    2011-12-02, Close, 48.56
    2011-12-05, Close, 48.77
    2011-12-06, Close, 49.14


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-12-07, Close, 49.22
    2011-12-08, Close, 48.77
    2011-12-09, Close, 49.06
    2011-12-12, Close, 48.87
    2011-12-13, Close, 48.45


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-12-13, BUY CREATE, 48.45
    2011-12-14, BUY EXECUTED, Price: 57.72, Cost: 57.72, Comm 0.00
    2011-12-14, Close, 48.50
    2011-12-15, Close, 48.75
    2011-12-16, Close, 49.02
    2011-12-19, Close, 48.60
    2011-12-20, Close, 49.79


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-12-20, SELL CREATE, 49.79
    2011-12-21, SELL EXECUTED, Price: 59.19, Cost: 57.72, Comm 0.00
    2011-12-21, OPERATION PROFIT, GROSS 1.47, NET 1.47
    2011-12-21, Close, 49.96
    2011-12-22, Close, 49.79
    2011-12-23, Close, 50.46
    2011-12-27, Close, 50.33
    2011-12-28, Close, 50.24


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2011-12-29, Close, 50.46
    2011-12-30, Close, 50.27
    2012-01-03, Close, 50.75
    2012-01-04, Close, 50.23
    2012-01-05, Close, 49.98


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-01-06, Close, 49.63
    2012-01-09, Close, 49.78
    2012-01-10, Close, 49.66
    2012-01-11, Close, 49.97
    2012-01-12, Close, 50.05


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-01-13, Close, 50.08
    2012-01-17, Close, 50.35
    2012-01-18, Close, 50.48
    2012-01-19, Close, 50.98
    2012-01-20, Close, 51.32


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-01-23, Close, 51.24
    2012-01-24, Close, 51.64
    2012-01-25, Close, 51.71
    2012-01-26, Close, 51.29
    2012-01-27, Close, 51.07


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-01-30, Close, 51.57
    2012-01-31, Close, 51.62
    2012-02-01, Close, 52.31
    2012-02-02, Close, 52.10
    2012-02-03, Close, 52.18


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-02-06, Close, 52.05
    2012-02-07, Close, 51.89
    2012-02-08, Close, 51.83
    2012-02-09, Close, 52.12
    2012-02-10, Close, 52.07


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-02-13, Close, 51.98
    2012-02-14, Close, 52.34
    2012-02-15, Close, 51.95
    2012-02-16, Close, 52.19
    2012-02-17, Close, 52.56


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-02-21, Close, 50.53
    2012-02-22, Close, 49.29
    2012-02-23, Close, 49.24
    2012-02-24, Close, 49.45
    2012-02-27, Close, 49.18


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-02-27, BUY CREATE, 49.18
    2012-02-28, BUY EXECUTED, Price: 58.44, Cost: 58.44, Comm 0.00
    2012-02-28, Close, 49.57
    2012-02-29, Close, 49.70
    2012-03-01, Close, 49.48
    2012-03-02, Close, 49.64
    2012-03-05, Close, 49.97


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-03-05, BUY CREATE, 49.97
    2012-03-06, BUY EXECUTED, Price: 59.04, Cost: 59.04, Comm 0.00
    2012-03-06, Close, 49.61
    2012-03-07, Close, 50.35
    2012-03-08, Close, 50.61
    2012-03-09, Close, 50.88
    2012-03-12, Close, 51.39


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-03-12, BUY CREATE, 51.39
    2012-03-13, BUY EXECUTED, Price: 60.93, Cost: 60.93, Comm 0.00
    2012-03-13, Close, 51.66
    2012-03-14, Close, 51.72
    2012-03-15, Close, 51.85
    2012-03-16, Close, 51.52
    2012-03-19, Close, 51.44


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-03-19, BUY CREATE, 51.44
    2012-03-20, BUY EXECUTED, Price: 60.33, Cost: 60.33, Comm 0.00
    2012-03-20, Close, 51.32
    2012-03-21, Close, 51.28
    2012-03-22, Close, 51.36
    2012-03-23, Close, 51.44
    2012-03-26, Close, 51.83


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-03-26, BUY CREATE, 51.83
    2012-03-27, BUY EXECUTED, Price: 61.35, Cost: 61.35, Comm 0.00
    2012-03-27, Close, 51.73
    2012-03-28, Close, 51.82
    2012-03-29, Close, 51.50
    2012-03-30, Close, 51.83
    2012-04-02, Close, 51.96


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-04-02, BUY CREATE, 51.96
    2012-04-03, BUY EXECUTED, Price: 61.14, Cost: 61.14, Comm 0.00
    2012-04-03, Close, 51.36
    2012-04-04, Close, 51.03
    2012-04-05, Close, 51.38
    2012-04-09, Close, 50.92
    2012-04-10, Close, 50.75


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-04-10, BUY CREATE, 50.75
    2012-04-11, BUY EXECUTED, Price: 60.29, Cost: 60.29, Comm 0.00
    2012-04-11, Close, 50.64
    2012-04-12, Close, 50.93
    2012-04-13, Close, 50.61
    2012-04-16, Close, 51.30
    2012-04-17, Close, 52.39


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-04-17, SELL CREATE, 52.39
    2012-04-18, SELL EXECUTED, Price: 61.55, Cost: 60.22, Comm 0.00
    2012-04-18, Close, 52.55
    2012-04-19, Close, 52.29
    2012-04-20, Close, 52.88
    2012-04-23, Close, 50.42
    2012-04-24, Close, 48.92


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-04-24, BUY CREATE, 48.92
    2012-04-25, BUY EXECUTED, Price: 57.91, Cost: 57.91, Comm 0.00
    2012-04-25, Close, 48.57
    2012-04-26, Close, 49.92
    2012-04-27, Close, 49.99
    2012-04-30, Close, 49.89
    2012-05-01, Close, 50.02


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-05-01, BUY CREATE, 50.02
    2012-05-02, BUY EXECUTED, Price: 58.96, Cost: 58.96, Comm 0.00
    2012-05-02, Close, 49.97
    2012-05-03, Close, 49.95
    2012-05-04, Close, 49.71
    2012-05-07, Close, 50.12
    2012-05-08, Close, 50.00


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-05-08, BUY CREATE, 50.00
    2012-05-09, BUY EXECUTED, Price: 58.48, Cost: 58.48, Comm 0.00
    2012-05-09, Close, 50.33
    2012-05-10, Close, 50.46
    2012-05-11, Close, 50.66
    2012-05-14, Close, 50.36
    2012-05-15, Close, 50.60


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-05-15, BUY CREATE, 50.60
    2012-05-16, BUY EXECUTED, Price: 59.53, Cost: 59.53, Comm 0.00
    2012-05-16, Close, 50.46
    2012-05-17, Close, 52.59
    2012-05-18, Close, 53.23
    2012-05-21, Close, 53.75
    2012-05-22, Close, 54.33


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-05-22, SELL CREATE, 54.33
    2012-05-23, SELL EXECUTED, Price: 63.39, Cost: 59.62, Comm 0.00
    2012-05-23, Close, 55.06
    2012-05-24, Close, 55.48
    2012-05-25, Close, 55.68
    2012-05-29, Close, 56.00
    2012-05-30, Close, 55.79


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-05-30, SELL CREATE, 55.79
    2012-05-31, SELL EXECUTED, Price: 65.40, Cost: 59.62, Comm 0.00
    2012-05-31, Close, 56.12
    2012-06-01, Close, 55.89
    2012-06-04, Close, 56.26
    2012-06-05, Close, 55.84
    2012-06-06, Close, 56.21


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-06-06, SELL CREATE, 56.21
    2012-06-07, SELL EXECUTED, Price: 66.08, Cost: 59.62, Comm 0.00
    2012-06-07, Close, 56.16
    2012-06-08, Close, 58.16
    2012-06-11, Close, 57.57
    2012-06-12, Close, 57.74
    2012-06-13, Close, 57.18


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-06-13, SELL CREATE, 57.18
    2012-06-14, SELL EXECUTED, Price: 67.10, Cost: 59.62, Comm 0.00
    2012-06-14, Close, 57.66
    2012-06-15, Close, 57.76
    2012-06-18, Close, 58.08
    2012-06-19, Close, 57.81
    2012-06-20, Close, 58.42


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-06-20, SELL CREATE, 58.42
    2012-06-21, SELL EXECUTED, Price: 68.49, Cost: 59.62, Comm 0.00
    2012-06-21, Close, 57.72
    2012-06-22, Close, 57.38
    2012-06-25, Close, 58.13
    2012-06-26, Close, 58.47
    2012-06-27, Close, 58.48


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-06-27, SELL CREATE, 58.48
    2012-06-28, SELL EXECUTED, Price: 67.92, Cost: 59.62, Comm 0.00
    2012-06-28, Close, 58.23
    2012-06-29, Close, 59.44
    2012-07-02, Close, 59.12
    2012-07-03, Close, 60.32
    2012-07-05, Close, 60.60


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-07-05, SELL CREATE, 60.60
    2012-07-06, SELL EXECUTED, Price: 70.73, Cost: 59.62, Comm 0.00
    2012-07-06, Close, 60.84
    2012-07-09, Close, 61.18
    2012-07-10, Close, 61.48
    2012-07-11, Close, 61.61
    2012-07-12, Close, 61.65


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-07-12, SELL CREATE, 61.65
    2012-07-13, SELL EXECUTED, Price: 72.27, Cost: 59.62, Comm 0.00
    2012-07-13, Close, 62.39
    2012-07-16, Close, 62.22
    2012-07-17, Close, 62.32
    2012-07-18, Close, 62.11
    2012-07-19, Close, 60.98


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-07-19, SELL CREATE, 60.98
    2012-07-20, SELL EXECUTED, Price: 71.46, Cost: 59.62, Comm 0.00
    2012-07-20, Close, 61.60
    2012-07-23, Close, 61.26
    2012-07-24, Close, 61.50
    2012-07-25, Close, 61.45
    2012-07-26, Close, 62.81


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-07-26, SELL CREATE, 62.81
    2012-07-27, SELL EXECUTED, Price: 73.84, Cost: 59.62, Comm 0.00
    2012-07-27, OPERATION PROFIT, GROSS 91.83, NET 91.83
    2012-07-27, Close, 63.53
    2012-07-30, Close, 63.92
    2012-07-31, Close, 63.46
    2012-08-01, Close, 62.77
    2012-08-02, Close, 63.13


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-08-03, Close, 63.56
    2012-08-06, Close, 63.33
    2012-08-07, Close, 63.08
    2012-08-08, Close, 63.70
    2012-08-09, Close, 63.30


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-08-10, Close, 63.16
    2012-08-13, Close, 62.92
    2012-08-14, Close, 63.44
    2012-08-15, Close, 63.82
    2012-08-16, Close, 61.84


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-08-16, BUY CREATE, 61.84
    2012-08-17, BUY EXECUTED, Price: 72.41, Cost: 72.41, Comm 0.00
    2012-08-17, Close, 61.71
    2012-08-20, Close, 61.97
    2012-08-21, Close, 61.23
    2012-08-22, Close, 61.52
    2012-08-23, Close, 61.34


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-08-23, BUY CREATE, 61.34
    2012-08-24, BUY EXECUTED, Price: 71.39, Cost: 71.39, Comm 0.00
    2012-08-24, Close, 61.81
    2012-08-27, Close, 62.14
    2012-08-28, Close, 62.07
    2012-08-29, Close, 62.38
    2012-08-30, Close, 61.93


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-08-30, BUY CREATE, 61.93
    2012-08-31, BUY EXECUTED, Price: 72.59, Cost: 72.59, Comm 0.00
    2012-08-31, Close, 62.23
    2012-09-04, Close, 63.01
    2012-09-05, Close, 63.04
    2012-09-06, Close, 64.12
    2012-09-07, Close, 63.28


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-09-07, BUY CREATE, 63.28
    2012-09-10, BUY EXECUTED, Price: 73.89, Cost: 73.89, Comm 0.00
    2012-09-10, Close, 63.01
    2012-09-11, Close, 63.48
    2012-09-12, Close, 63.49
    2012-09-13, Close, 64.41
    2012-09-14, Close, 63.86


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-09-14, SELL CREATE, 63.86
    2012-09-17, SELL EXECUTED, Price: 74.36, Cost: 72.57, Comm 0.00
    2012-09-17, Close, 63.42
    2012-09-18, Close, 63.39
    2012-09-19, Close, 63.75
    2012-09-20, Close, 64.07
    2012-09-21, Close, 63.82


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-09-21, SELL CREATE, 63.82
    2012-09-24, SELL EXECUTED, Price: 74.20, Cost: 72.57, Comm 0.00
    2012-09-24, Close, 64.06
    2012-09-25, Close, 63.65
    2012-09-26, Close, 63.59
    2012-09-27, Close, 63.41
    2012-09-28, Close, 63.26


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-09-28, BUY CREATE, 63.26
    2012-10-01, BUY EXECUTED, Price: 73.80, Cost: 73.80, Comm 0.00
    2012-10-01, Close, 63.47
    2012-10-02, Close, 63.22
    2012-10-03, Close, 63.60
    2012-10-04, Close, 64.05
    2012-10-05, Close, 64.40


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-10-05, SELL CREATE, 64.40
    2012-10-08, SELL EXECUTED, Price: 75.16, Cost: 72.98, Comm 0.00
    2012-10-08, Close, 64.50
    2012-10-09, Close, 63.55
    2012-10-10, Close, 64.65
    2012-10-11, Close, 64.30
    2012-10-12, Close, 64.98


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-10-12, SELL CREATE, 64.98
    2012-10-15, SELL EXECUTED, Price: 75.87, Cost: 72.98, Comm 0.00
    2012-10-15, Close, 66.13
    2012-10-16, Close, 65.92
    2012-10-17, Close, 66.03
    2012-10-18, Close, 65.62
    2012-10-19, Close, 64.82


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-10-19, BUY CREATE, 64.82
    2012-10-22, BUY EXECUTED, Price: 75.68, Cost: 75.68, Comm 0.00
    2012-10-22, Close, 64.84
    2012-10-23, Close, 64.08
    2012-10-24, Close, 64.13
    2012-10-25, Close, 64.56
    2012-10-26, Close, 64.38


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-10-26, BUY CREATE, 64.38
    2012-10-31, BUY EXECUTED, Price: 75.25, Cost: 75.25, Comm 0.00
    2012-10-31, Close, 64.30
    2012-11-01, Close, 62.96
    2012-11-02, Close, 62.38
    2012-11-05, Close, 62.69
    2012-11-06, Close, 63.22


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-11-06, BUY CREATE, 63.22
    2012-11-07, BUY EXECUTED, Price: 73.42, Cost: 73.42, Comm 0.00
    2012-11-07, Close, 62.67
    2012-11-08, Close, 62.13
    2012-11-09, Close, 61.98
    2012-11-12, Close, 62.13
    2012-11-13, Close, 61.55


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-11-13, BUY CREATE, 61.55
    2012-11-14, BUY EXECUTED, Price: 71.86, Cost: 71.86, Comm 0.00
    2012-11-14, Close, 61.12
    2012-11-15, Close, 58.90
    2012-11-16, Close, 58.31
    2012-11-19, Close, 59.16
    2012-11-20, Close, 59.14


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-11-20, BUY CREATE, 59.14
    2012-11-21, BUY EXECUTED, Price: 68.86, Cost: 68.86, Comm 0.00
    2012-11-21, Close, 59.05
    2012-11-23, Close, 60.17
    2012-11-26, Close, 59.92
    2012-11-27, Close, 59.57
    2012-11-28, Close, 60.48


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-11-28, BUY CREATE, 60.48
    2012-11-29, BUY EXECUTED, Price: 70.48, Cost: 70.48, Comm 0.00
    2012-11-29, Close, 60.71
    2012-11-30, Close, 61.73
    2012-12-03, Close, 61.15
    2012-12-04, Close, 61.82
    2012-12-05, Close, 61.76


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-12-05, BUY CREATE, 61.76
    2012-12-06, BUY EXECUTED, Price: 71.67, Cost: 71.67, Comm 0.00
    2012-12-06, Close, 61.70
    2012-12-07, Close, 62.31
    2012-12-10, Close, 62.19
    2012-12-11, Close, 61.10
    2012-12-12, Close, 59.42


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-12-12, BUY CREATE, 59.42
    2012-12-13, BUY EXECUTED, Price: 69.04, Cost: 69.04, Comm 0.00
    2012-12-13, Close, 59.51
    2012-12-14, Close, 59.26
    2012-12-17, Close, 59.64
    2012-12-18, Close, 59.90
    2012-12-19, Close, 59.06


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-12-19, BUY CREATE, 59.06
    2012-12-20, BUY EXECUTED, Price: 68.33, Cost: 68.33, Comm 0.00
    2012-12-20, Close, 59.47
    2012-12-21, Close, 59.17
    2012-12-24, Close, 59.10
    2012-12-26, Close, 58.60
    2012-12-27, Close, 58.77


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2012-12-27, BUY CREATE, 58.77
    2012-12-28, BUY EXECUTED, Price: 67.91, Cost: 67.91, Comm 0.00
    2012-12-28, Close, 58.27
    2012-12-31, Close, 58.81
    2013-01-02, Close, 59.68
    2013-01-03, Close, 59.30
    2013-01-04, Close, 59.52


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-01-04, BUY CREATE, 59.52
    2013-01-07, BUY EXECUTED, Price: 68.83, Cost: 68.83, Comm 0.00
    2013-01-07, Close, 58.95
    2013-01-08, Close, 59.12
    2013-01-09, Close, 59.10
    2013-01-10, Close, 58.92
    2013-01-11, Close, 59.15


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-01-11, BUY CREATE, 59.15
    2013-01-14, BUY EXECUTED, Price: 68.49, Cost: 68.49, Comm 0.00
    2013-01-14, Close, 58.87
    2013-01-15, Close, 59.45
    2013-01-16, Close, 59.65
    2013-01-17, Close, 59.34
    2013-01-18, Close, 59.64


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-01-18, BUY CREATE, 59.64
    2013-01-22, BUY EXECUTED, Price: 69.05, Cost: 69.05, Comm 0.00
    2013-01-22, Close, 59.97
    2013-01-23, Close, 59.89
    2013-01-24, Close, 60.15
    2013-01-25, Close, 59.47
    2013-01-28, Close, 59.77


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-01-28, BUY CREATE, 59.77
    2013-01-29, BUY EXECUTED, Price: 69.21, Cost: 69.21, Comm 0.00
    2013-01-29, Close, 60.24
    2013-01-30, Close, 60.12
    2013-01-31, Close, 60.29
    2013-02-01, Close, 60.76
    2013-02-04, Close, 60.02


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-02-04, BUY CREATE, 60.02
    2013-02-05, BUY EXECUTED, Price: 69.87, Cost: 69.87, Comm 0.00
    2013-02-05, Close, 61.00
    2013-02-06, Close, 61.46
    2013-02-07, Close, 61.39
    2013-02-08, Close, 61.61
    2013-02-11, Close, 61.54


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-02-11, SELL CREATE, 61.54
    2013-02-12, SELL EXECUTED, Price: 71.49, Cost: 70.68, Comm 0.00
    2013-02-12, Close, 61.54
    2013-02-13, Close, 61.53
    2013-02-14, Close, 61.04
    2013-02-15, Close, 59.73
    2013-02-19, Close, 59.27


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-02-19, BUY CREATE, 59.27
    2013-02-20, BUY EXECUTED, Price: 68.72, Cost: 68.72, Comm 0.00
    2013-02-20, Close, 59.65
    2013-02-21, Close, 60.56
    2013-02-22, Close, 60.68
    2013-02-25, Close, 60.71
    2013-02-26, Close, 61.29


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-02-26, SELL CREATE, 61.29
    2013-02-27, SELL EXECUTED, Price: 70.92, Cost: 70.56, Comm 0.00
    2013-02-27, Close, 61.76
    2013-02-28, Close, 61.01
    2013-03-01, Close, 61.83
    2013-03-04, Close, 63.14
    2013-03-05, Close, 63.54


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-03-05, SELL CREATE, 63.54
    2013-03-06, SELL EXECUTED, Price: 73.75, Cost: 70.56, Comm 0.00
    2013-03-06, Close, 63.25
    2013-03-07, Close, 63.20
    2013-03-08, Close, 63.35
    2013-03-11, Close, 63.31
    2013-03-12, Close, 63.85


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-03-12, SELL CREATE, 63.85
    2013-03-13, SELL EXECUTED, Price: 73.89, Cost: 70.56, Comm 0.00
    2013-03-13, Close, 63.89
    2013-03-14, Close, 63.52
    2013-03-15, Close, 62.89
    2013-03-18, Close, 62.68
    2013-03-19, Close, 62.85


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-03-19, BUY CREATE, 62.85
    2013-03-20, BUY EXECUTED, Price: 72.81, Cost: 72.81, Comm 0.00
    2013-03-20, Close, 63.32
    2013-03-21, Close, 63.44
    2013-03-22, Close, 64.44
    2013-03-25, Close, 64.93
    2013-03-26, Close, 64.86


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-03-26, SELL CREATE, 64.86
    2013-03-27, SELL EXECUTED, Price: 74.31, Cost: 70.72, Comm 0.00
    2013-03-27, Close, 64.87
    2013-03-28, Close, 64.91
    2013-04-01, Close, 65.43
    2013-04-02, Close, 65.95
    2013-04-03, Close, 65.93


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-04-03, SELL CREATE, 65.93
    2013-04-04, SELL EXECUTED, Price: 75.96, Cost: 70.72, Comm 0.00
    2013-04-04, Close, 66.10
    2013-04-05, Close, 66.27
    2013-04-08, Close, 67.05
    2013-04-09, Close, 67.77
    2013-04-10, Close, 67.12


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-04-10, SELL CREATE, 67.12
    2013-04-11, SELL EXECUTED, Price: 77.36, Cost: 70.72, Comm 0.00
    2013-04-11, Close, 67.48
    2013-04-12, Close, 68.15
    2013-04-15, Close, 68.07
    2013-04-16, Close, 68.25
    2013-04-17, Close, 68.11


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-04-17, SELL CREATE, 68.11
    2013-04-18, SELL EXECUTED, Price: 78.76, Cost: 70.72, Comm 0.00
    2013-04-18, Close, 66.93
    2013-04-19, Close, 67.91
    2013-04-22, Close, 67.64
    2013-04-23, Close, 68.61
    2013-04-24, Close, 67.69


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-04-24, SELL CREATE, 67.69
    2013-04-25, SELL EXECUTED, Price: 78.18, Cost: 70.72, Comm 0.00
    2013-04-25, Close, 68.23
    2013-04-26, Close, 68.57
    2013-04-29, Close, 68.00
    2013-04-30, Close, 67.42
    2013-05-01, Close, 67.72


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-05-01, SELL CREATE, 67.72
    2013-05-02, SELL EXECUTED, Price: 77.91, Cost: 70.72, Comm 0.00
    2013-05-02, Close, 68.06
    2013-05-03, Close, 68.75
    2013-05-06, Close, 68.38
    2013-05-07, Close, 68.38
    2013-05-08, Close, 68.29


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-05-08, SELL CREATE, 68.29
    2013-05-09, SELL EXECUTED, Price: 78.31, Cost: 70.72, Comm 0.00
    2013-05-09, Close, 68.42
    2013-05-10, Close, 68.85
    2013-05-13, Close, 68.51
    2013-05-14, Close, 68.75
    2013-05-15, Close, 69.69


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-05-15, SELL CREATE, 69.69
    2013-05-16, SELL EXECUTED, Price: 78.10, Cost: 70.72, Comm 0.00
    2013-05-16, Close, 68.51
    2013-05-17, Close, 67.96
    2013-05-20, Close, 67.55
    2013-05-21, Close, 67.54
    2013-05-22, Close, 67.22


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-05-22, BUY CREATE, 67.22
    2013-05-23, BUY EXECUTED, Price: 76.83, Cost: 76.83, Comm 0.00
    2013-05-23, Close, 66.61
    2013-05-24, Close, 67.47
    2013-05-28, Close, 67.48
    2013-05-29, Close, 66.52
    2013-05-30, Close, 66.00


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-05-30, BUY CREATE, 66.00
    2013-05-31, BUY EXECUTED, Price: 75.36, Cost: 75.36, Comm 0.00
    2013-05-31, Close, 65.31
    2013-06-03, Close, 66.05
    2013-06-04, Close, 66.27
    2013-06-05, Close, 65.67
    2013-06-06, Close, 66.00


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-06-06, BUY CREATE, 66.00
    2013-06-07, BUY EXECUTED, Price: 76.38, Cost: 76.38, Comm 0.00
    2013-06-07, Close, 66.61
    2013-06-10, Close, 66.11
    2013-06-11, Close, 65.67
    2013-06-12, Close, 65.31
    2013-06-13, Close, 65.45


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-06-13, BUY CREATE, 65.45
    2013-06-14, BUY EXECUTED, Price: 74.84, Cost: 74.84, Comm 0.00
    2013-06-14, Close, 65.34
    2013-06-17, Close, 65.41
    2013-06-18, Close, 66.09
    2013-06-19, Close, 64.98
    2013-06-20, Close, 63.73


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-06-20, BUY CREATE, 63.73
    2013-06-21, BUY EXECUTED, Price: 73.52, Cost: 73.52, Comm 0.00
    2013-06-21, Close, 64.15
    2013-06-24, Close, 64.75
    2013-06-25, Close, 64.90
    2013-06-26, Close, 65.46
    2013-06-27, Close, 65.68


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-06-27, BUY CREATE, 65.68
    2013-06-28, BUY EXECUTED, Price: 75.11, Cost: 75.11, Comm 0.00
    2013-06-28, Close, 65.01
    2013-07-01, Close, 65.09
    2013-07-02, Close, 65.20
    2013-07-03, Close, 65.24
    2013-07-05, Close, 65.63


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-07-05, BUY CREATE, 65.63
    2013-07-08, BUY EXECUTED, Price: 75.68, Cost: 75.68, Comm 0.00
    2013-07-08, Close, 66.94
    2013-07-09, Close, 67.22
    2013-07-10, Close, 67.00
    2013-07-11, Close, 67.75
    2013-07-12, Close, 67.75


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-07-12, SELL CREATE, 67.75
    2013-07-15, SELL EXECUTED, Price: 77.30, Cost: 73.23, Comm 0.00
    2013-07-15, Close, 67.22
    2013-07-16, Close, 67.52
    2013-07-17, Close, 67.37
    2013-07-18, Close, 67.49
    2013-07-19, Close, 68.14


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-07-19, SELL CREATE, 68.14
    2013-07-22, SELL EXECUTED, Price: 77.89, Cost: 73.23, Comm 0.00
    2013-07-22, Close, 67.96
    2013-07-23, Close, 68.55
    2013-07-24, Close, 68.27
    2013-07-25, Close, 68.08
    2013-07-26, Close, 68.07


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-07-26, SELL CREATE, 68.07
    2013-07-29, SELL EXECUTED, Price: 77.83, Cost: 73.23, Comm 0.00
    2013-07-29, Close, 68.06
    2013-07-30, Close, 67.97
    2013-07-31, Close, 68.02
    2013-08-01, Close, 68.26
    2013-08-02, Close, 68.72


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-08-02, SELL CREATE, 68.72
    2013-08-05, SELL EXECUTED, Price: 78.62, Cost: 73.23, Comm 0.00
    2013-08-05, Close, 68.74
    2013-08-06, Close, 67.96
    2013-08-07, Close, 67.93
    2013-08-08, Close, 67.82
    2013-08-09, Close, 67.52


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-08-09, SELL CREATE, 67.52
    2013-08-12, SELL EXECUTED, Price: 76.56, Cost: 73.23, Comm 0.00
    2013-08-12, Close, 67.67
    2013-08-13, Close, 67.48
    2013-08-14, Close, 67.08
    2013-08-15, Close, 65.33
    2013-08-16, Close, 65.07


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-08-16, BUY CREATE, 65.07
    2013-08-19, BUY EXECUTED, Price: 73.88, Cost: 73.88, Comm 0.00
    2013-08-19, Close, 64.60
    2013-08-20, Close, 64.29
    2013-08-21, Close, 64.58
    2013-08-22, Close, 64.50
    2013-08-23, Close, 64.48


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-08-23, BUY CREATE, 64.48
    2013-08-26, BUY EXECUTED, Price: 73.64, Cost: 73.64, Comm 0.00
    2013-08-26, Close, 64.12
    2013-08-27, Close, 63.97
    2013-08-28, Close, 63.55
    2013-08-29, Close, 63.59
    2013-08-30, Close, 64.07


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-08-30, BUY CREATE, 64.07
    2013-09-03, BUY EXECUTED, Price: 73.48, Cost: 73.48, Comm 0.00
    2013-09-03, Close, 63.81
    2013-09-04, Close, 64.01
    2013-09-05, Close, 63.80
    2013-09-06, Close, 63.73
    2013-09-09, Close, 64.54


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-09-09, BUY CREATE, 64.54
    2013-09-10, BUY EXECUTED, Price: 73.66, Cost: 73.66, Comm 0.00
    2013-09-10, Close, 64.94
    2013-09-11, Close, 65.01
    2013-09-12, Close, 64.89
    2013-09-13, Close, 65.29
    2013-09-16, Close, 65.66


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-09-16, BUY CREATE, 65.66
    2013-09-17, BUY EXECUTED, Price: 74.84, Cost: 74.84, Comm 0.00
    2013-09-17, Close, 65.98
    2013-09-18, Close, 67.10
    2013-09-19, Close, 66.91
    2013-09-20, Close, 66.58
    2013-09-23, Close, 67.10


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-09-23, SELL CREATE, 67.10
    2013-09-24, SELL EXECUTED, Price: 76.40, Cost: 73.49, Comm 0.00
    2013-09-24, Close, 66.51
    2013-09-25, Close, 65.54
    2013-09-26, Close, 65.51
    2013-09-27, Close, 65.29
    2013-09-30, Close, 64.94


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-09-30, BUY CREATE, 64.94
    2013-10-01, BUY EXECUTED, Price: 73.87, Cost: 73.87, Comm 0.00
    2013-10-01, Close, 64.61
    2013-10-02, Close, 64.72
    2013-10-03, Close, 64.23
    2013-10-04, Close, 63.92
    2013-10-07, Close, 63.10


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-10-07, BUY CREATE, 63.10
    2013-10-08, BUY EXECUTED, Price: 71.85, Cost: 71.85, Comm 0.00
    2013-10-08, Close, 64.00
    2013-10-09, Close, 64.09
    2013-10-10, Close, 65.66
    2013-10-11, Close, 65.69
    2013-10-14, Close, 65.57


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-10-14, BUY CREATE, 65.57
    2013-10-15, BUY EXECUTED, Price: 74.43, Cost: 74.43, Comm 0.00
    2013-10-15, Close, 65.30
    2013-10-16, Close, 66.38
    2013-10-17, Close, 66.53
    2013-10-18, Close, 66.47
    2013-10-21, Close, 65.98


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-10-21, BUY CREATE, 65.98
    2013-10-22, BUY EXECUTED, Price: 75.43, Cost: 75.43, Comm 0.00
    2013-10-22, Close, 67.01
    2013-10-23, Close, 66.64
    2013-10-24, Close, 67.10
    2013-10-25, Close, 66.80
    2013-10-28, Close, 67.73


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-10-28, SELL CREATE, 67.73
    2013-10-29, SELL EXECUTED, Price: 77.25, Cost: 73.59, Comm 0.00
    2013-10-29, Close, 67.66
    2013-10-30, Close, 67.53
    2013-10-31, Close, 67.38
    2013-11-01, Close, 67.67
    2013-11-04, Close, 67.89


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-11-04, SELL CREATE, 67.89
    2013-11-05, SELL EXECUTED, Price: 76.81, Cost: 73.59, Comm 0.00
    2013-11-05, Close, 67.97
    2013-11-06, Close, 68.62
    2013-11-07, Close, 68.05
    2013-11-08, Close, 68.45
    2013-11-11, Close, 69.37


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-11-11, SELL CREATE, 69.37
    2013-11-12, SELL EXECUTED, Price: 78.85, Cost: 73.59, Comm 0.00
    2013-11-12, Close, 69.11
    2013-11-13, Close, 69.27
    2013-11-14, Close, 69.43
    2013-11-15, Close, 69.55
    2013-11-18, Close, 69.55


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-11-18, SELL CREATE, 69.55
    2013-11-19, SELL EXECUTED, Price: 79.30, Cost: 73.59, Comm 0.00
    2013-11-19, Close, 69.58
    2013-11-20, Close, 69.27
    2013-11-21, Close, 69.24
    2013-11-22, Close, 70.07
    2013-11-25, Close, 70.62


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-11-25, SELL CREATE, 70.62
    2013-11-26, SELL EXECUTED, Price: 80.44, Cost: 73.59, Comm 0.00
    2013-11-26, Close, 70.84
    2013-11-27, Close, 71.05
    2013-11-29, Close, 71.13
    2013-12-02, Close, 71.21
    2013-12-03, Close, 71.30


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-12-03, SELL CREATE, 71.30
    2013-12-04, SELL EXECUTED, Price: 80.64, Cost: 73.59, Comm 0.00
    2013-12-04, Close, 70.84
    2013-12-05, Close, 70.15
    2013-12-06, Close, 70.59
    2013-12-09, Close, 70.60
    2013-12-10, Close, 69.83


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-12-10, SELL CREATE, 69.83
    2013-12-11, SELL EXECUTED, Price: 79.10, Cost: 73.59, Comm 0.00
    2013-12-11, Close, 69.84
    2013-12-12, Close, 69.32
    2013-12-13, Close, 68.95
    2013-12-16, Close, 68.65
    2013-12-17, Close, 68.22


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-12-17, BUY CREATE, 68.22
    2013-12-18, BUY EXECUTED, Price: 77.28, Cost: 77.28, Comm 0.00
    2013-12-18, Close, 68.83
    2013-12-19, Close, 68.21
    2013-12-20, Close, 68.38
    2013-12-23, Close, 68.77
    2013-12-24, Close, 68.89


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2013-12-24, BUY CREATE, 68.89
    2013-12-26, BUY EXECUTED, Price: 78.06, Cost: 78.06, Comm 0.00
    2013-12-26, Close, 69.23
    2013-12-27, Close, 69.30
    2013-12-30, Close, 69.44
    2013-12-31, Close, 69.49
    2014-01-02, Close, 69.68


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-01-02, SELL CREATE, 69.68
    2014-01-03, SELL EXECUTED, Price: 78.81, Cost: 74.33, Comm 0.00
    2014-01-03, Close, 69.45
    2014-01-06, Close, 69.07
    2014-01-07, Close, 69.28
    2014-01-08, Close, 68.73
    2014-01-09, Close, 68.96


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-01-09, BUY CREATE, 68.96
    2014-01-10, BUY EXECUTED, Price: 78.31, Cost: 78.31, Comm 0.00
    2014-01-10, Close, 68.92
    2014-01-13, Close, 68.43
    2014-01-14, Close, 68.85
    2014-01-15, Close, 68.58
    2014-01-16, Close, 67.79


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-01-16, BUY CREATE, 67.79
    2014-01-17, BUY EXECUTED, Price: 76.73, Cost: 76.73, Comm 0.00
    2014-01-17, Close, 67.28
    2014-01-21, Close, 66.97
    2014-01-22, Close, 66.54
    2014-01-23, Close, 66.20
    2014-01-24, Close, 65.72


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-01-24, BUY CREATE, 65.72
    2014-01-27, BUY EXECUTED, Price: 74.13, Cost: 74.13, Comm 0.00
    2014-01-27, Close, 65.48
    2014-01-28, Close, 65.94
    2014-01-29, Close, 65.44
    2014-01-30, Close, 66.01
    2014-01-31, Close, 65.95


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-01-31, BUY CREATE, 65.95
    2014-02-03, BUY EXECUTED, Price: 74.19, Cost: 74.19, Comm 0.00
    2014-02-03, Close, 64.17
    2014-02-04, Close, 64.23
    2014-02-05, Close, 64.35
    2014-02-06, Close, 64.31
    2014-02-07, Close, 65.13


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-02-07, BUY CREATE, 65.13
    2014-02-10, BUY EXECUTED, Price: 73.59, Cost: 73.59, Comm 0.00
    2014-02-10, Close, 65.14
    2014-02-11, Close, 66.06
    2014-02-12, Close, 66.20
    2014-02-13, Close, 66.55
    2014-02-14, Close, 66.93


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-02-14, BUY CREATE, 66.93
    2014-02-18, BUY EXECUTED, Price: 75.49, Cost: 75.49, Comm 0.00
    2014-02-18, Close, 66.52
    2014-02-19, Close, 66.10
    2014-02-20, Close, 64.92
    2014-02-21, Close, 64.57
    2014-02-24, Close, 64.77


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-02-24, BUY CREATE, 64.77
    2014-02-25, BUY EXECUTED, Price: 73.18, Cost: 73.18, Comm 0.00
    2014-02-25, Close, 64.77
    2014-02-26, Close, 66.04
    2014-02-27, Close, 65.84
    2014-02-28, Close, 65.97
    2014-03-03, Close, 65.45


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-03-03, BUY CREATE, 65.45
    2014-03-04, Order Canceled/Margin/Rejected
    2014-03-04, Close, 66.35
    2014-03-05, Close, 66.06
    2014-03-06, Close, 66.13
    2014-03-07, Close, 66.29
    2014-03-10, Close, 66.15


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-03-10, BUY CREATE, 66.15
    2014-03-11, Order Canceled/Margin/Rejected
    2014-03-11, Close, 66.59
    2014-03-12, Close, 67.13
    2014-03-13, Close, 66.60
    2014-03-14, Close, 66.02
    2014-03-17, Close, 66.37


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-03-17, BUY CREATE, 66.37
    2014-03-18, Order Canceled/Margin/Rejected
    2014-03-18, Close, 66.45
    2014-03-19, Close, 66.11
    2014-03-20, Close, 67.00
    2014-03-21, Close, 67.64
    2014-03-24, Close, 68.22


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-03-24, BUY CREATE, 68.22
    2014-03-25, Order Canceled/Margin/Rejected
    2014-03-25, Close, 68.32
    2014-03-26, Close, 67.75
    2014-03-27, Close, 67.67
    2014-03-28, Close, 67.56
    2014-03-31, Close, 67.93


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-03-31, BUY CREATE, 67.93
    2014-04-01, Order Canceled/Margin/Rejected
    2014-04-01, Close, 68.23
    2014-04-02, Close, 68.60
    2014-04-03, Close, 68.85
    2014-04-04, Close, 68.71
    2014-04-07, Close, 68.71


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-04-07, BUY CREATE, 68.71
    2014-04-08, Order Canceled/Margin/Rejected
    2014-04-08, Close, 69.49
    2014-04-09, Close, 69.30
    2014-04-10, Close, 68.34
    2014-04-11, Close, 67.99
    2014-04-14, Close, 68.77


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-04-14, BUY CREATE, 68.77
    2014-04-15, Order Canceled/Margin/Rejected
    2014-04-15, Close, 68.33
    2014-04-16, Close, 68.63
    2014-04-17, Close, 69.02
    2014-04-21, Close, 68.97
    2014-04-22, Close, 68.93


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-04-22, SELL CREATE, 68.93
    2014-04-23, SELL EXECUTED, Price: 77.82, Cost: 74.64, Comm 0.00
    2014-04-23, Close, 69.36
    2014-04-24, Close, 69.60
    2014-04-25, Close, 69.88
    2014-04-28, Close, 70.89
    2014-04-29, Close, 70.81


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-04-29, SELL CREATE, 70.81
    2014-04-30, SELL EXECUTED, Price: 79.59, Cost: 74.64, Comm 0.00
    2014-04-30, Close, 70.85
    2014-05-01, Close, 70.84
    2014-05-02, Close, 70.32
    2014-05-05, Close, 69.88
    2014-05-06, Close, 69.33


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-05-06, SELL CREATE, 69.33
    2014-05-07, SELL EXECUTED, Price: 77.85, Cost: 74.64, Comm 0.00
    2014-05-07, Close, 69.72
    2014-05-08, Close, 70.37
    2014-05-09, Close, 70.83
    2014-05-12, Close, 70.78
    2014-05-13, Close, 70.77


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-05-13, SELL CREATE, 70.77
    2014-05-14, SELL EXECUTED, Price: 79.05, Cost: 74.64, Comm 0.00
    2014-05-14, Close, 70.42
    2014-05-15, Close, 68.71
    2014-05-16, Close, 68.87
    2014-05-19, Close, 68.51
    2014-05-20, Close, 67.69


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-05-20, BUY CREATE, 67.69
    2014-05-21, BUY EXECUTED, Price: 75.94, Cost: 75.94, Comm 0.00
    2014-05-21, Close, 67.66
    2014-05-22, Close, 67.42
    2014-05-23, Close, 67.62
    2014-05-27, Close, 67.60
    2014-05-28, Close, 67.55


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-05-28, BUY CREATE, 67.55
    2014-05-29, BUY EXECUTED, Price: 75.67, Cost: 75.67, Comm 0.00
    2014-05-29, Close, 67.95
    2014-05-30, Close, 68.65
    2014-06-02, Close, 68.65
    2014-06-03, Close, 68.60
    2014-06-04, Close, 68.98


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-06-04, SELL CREATE, 68.98
    2014-06-05, SELL EXECUTED, Price: 77.05, Cost: 74.80, Comm 0.00
    2014-06-05, Close, 69.15
    2014-06-06, Close, 69.05
    2014-06-09, Close, 68.87
    2014-06-10, Close, 68.52
    2014-06-11, Close, 68.11


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-06-11, SELL CREATE, 68.11
    2014-06-12, SELL EXECUTED, Price: 76.05, Cost: 74.80, Comm 0.00
    2014-06-12, Close, 67.72
    2014-06-13, Close, 67.32
    2014-06-16, Close, 67.38
    2014-06-17, Close, 67.06
    2014-06-18, Close, 67.70


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-06-18, SELL CREATE, 67.70
    2014-06-19, SELL EXECUTED, Price: 75.88, Cost: 74.80, Comm 0.00
    2014-06-19, Close, 67.85
    2014-06-20, Close, 67.68
    2014-06-23, Close, 67.78
    2014-06-24, Close, 67.94
    2014-06-25, Close, 67.63


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-06-25, SELL CREATE, 67.63
    2014-06-26, SELL EXECUTED, Price: 75.52, Cost: 74.80, Comm 0.00
    2014-06-26, Close, 66.99
    2014-06-27, Close, 67.38
    2014-06-30, Close, 67.13
    2014-07-01, Close, 67.32
    2014-07-02, Close, 67.63


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-07-02, BUY CREATE, 67.63
    2014-07-03, BUY EXECUTED, Price: 75.61, Cost: 75.61, Comm 0.00
    2014-07-03, Close, 67.74
    2014-07-07, Close, 68.03
    2014-07-08, Close, 68.55
    2014-07-09, Close, 69.05
    2014-07-10, Close, 68.91


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-07-10, SELL CREATE, 68.91
    2014-07-11, SELL EXECUTED, Price: 76.87, Cost: 74.87, Comm 0.00
    2014-07-11, Close, 68.70
    2014-07-14, Close, 68.46
    2014-07-15, Close, 68.72
    2014-07-16, Close, 68.74
    2014-07-17, Close, 68.51


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-07-17, BUY CREATE, 68.51
    2014-07-18, BUY EXECUTED, Price: 76.62, Cost: 76.62, Comm 0.00
    2014-07-18, Close, 68.94
    2014-07-21, Close, 68.65
    2014-07-22, Close, 68.54
    2014-07-23, Close, 68.85
    2014-07-24, Close, 68.28


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-07-24, SELL CREATE, 68.28
    2014-07-25, SELL EXECUTED, Price: 76.19, Cost: 75.01, Comm 0.00
    2014-07-25, Close, 67.94
    2014-07-28, Close, 67.71
    2014-07-29, Close, 67.47
    2014-07-30, Close, 66.87
    2014-07-31, Close, 65.80


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-07-31, BUY CREATE, 65.80
    2014-08-01, BUY EXECUTED, Price: 73.32, Cost: 73.32, Comm 0.00
    2014-08-01, Close, 65.77
    2014-08-04, Close, 65.77
    2014-08-05, Close, 65.59
    2014-08-06, Close, 66.79
    2014-08-07, Close, 66.57


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-08-07, BUY CREATE, 66.57
    2014-08-08, BUY EXECUTED, Price: 74.06, Cost: 74.06, Comm 0.00
    2014-08-08, Close, 67.22
    2014-08-11, Close, 66.94
    2014-08-12, Close, 66.81
    2014-08-13, Close, 66.64
    2014-08-14, Close, 66.96


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-08-14, BUY CREATE, 66.96
    2014-08-15, BUY EXECUTED, Price: 74.65, Cost: 74.65, Comm 0.00
    2014-08-15, Close, 66.52
    2014-08-18, Close, 67.05
    2014-08-19, Close, 67.41
    2014-08-20, Close, 67.48
    2014-08-21, Close, 68.01


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-08-21, SELL CREATE, 68.01
    2014-08-22, SELL EXECUTED, Price: 75.78, Cost: 74.80, Comm 0.00
    2014-08-22, Close, 68.17
    2014-08-25, Close, 68.13
    2014-08-26, Close, 67.98
    2014-08-27, Close, 68.28
    2014-08-28, Close, 68.32


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-08-28, SELL CREATE, 68.32
    2014-08-29, SELL EXECUTED, Price: 75.75, Cost: 74.80, Comm 0.00
    2014-08-29, Close, 67.96
    2014-09-02, Close, 68.19
    2014-09-03, Close, 68.42
    2014-09-04, Close, 68.92
    2014-09-05, Close, 69.77


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-09-05, SELL CREATE, 69.77
    2014-09-08, SELL EXECUTED, Price: 77.13, Cost: 74.80, Comm 0.00
    2014-09-08, Close, 68.89
    2014-09-09, Close, 69.08
    2014-09-10, Close, 68.87
    2014-09-11, Close, 68.50
    2014-09-12, Close, 68.21


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-09-12, SELL CREATE, 68.21
    2014-09-15, SELL EXECUTED, Price: 75.78, Cost: 74.80, Comm 0.00
    2014-09-15, Close, 68.24
    2014-09-16, Close, 68.70
    2014-09-17, Close, 68.63
    2014-09-18, Close, 68.61
    2014-09-19, Close, 69.17


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-09-19, SELL CREATE, 69.17
    2014-09-22, SELL EXECUTED, Price: 76.79, Cost: 74.80, Comm 0.00
    2014-09-22, Close, 68.69
    2014-09-23, Close, 68.05
    2014-09-24, Close, 69.39
    2014-09-25, Close, 68.52
    2014-09-26, Close, 68.85


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-09-26, SELL CREATE, 68.85
    2014-09-29, SELL EXECUTED, Price: 76.06, Cost: 74.80, Comm 0.00
    2014-09-29, Close, 68.49
    2014-09-30, Close, 68.84
    2014-10-01, Close, 68.52
    2014-10-02, Close, 68.62
    2014-10-03, Close, 69.60


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-10-03, SELL CREATE, 69.60
    2014-10-06, SELL EXECUTED, Price: 77.05, Cost: 74.80, Comm 0.00
    2014-10-06, Close, 69.63
    2014-10-07, Close, 69.58
    2014-10-08, Close, 70.43
    2014-10-09, Close, 70.09
    2014-10-10, Close, 70.48


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-10-10, SELL CREATE, 70.48
    2014-10-13, SELL EXECUTED, Price: 78.03, Cost: 74.80, Comm 0.00
    2014-10-13, Close, 69.82
    2014-10-14, Close, 70.20
    2014-10-15, Close, 67.69
    2014-10-16, Close, 66.45
    2014-10-17, Close, 66.70


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-10-17, BUY CREATE, 66.70
    2014-10-20, BUY EXECUTED, Price: 74.14, Cost: 74.14, Comm 0.00
    2014-10-20, Close, 67.64
    2014-10-21, Close, 68.43
    2014-10-22, Close, 68.44
    2014-10-23, Close, 68.64
    2014-10-24, Close, 68.76


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-10-24, SELL CREATE, 68.76
    2014-10-27, SELL EXECUTED, Price: 76.33, Cost: 74.70, Comm 0.00
    2014-10-27, Close, 68.94
    2014-10-28, Close, 68.73
    2014-10-29, Close, 68.76
    2014-10-30, Close, 68.82
    2014-10-31, Close, 68.66


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-10-31, SELL CREATE, 68.66
    2014-11-03, SELL EXECUTED, Price: 76.35, Cost: 74.70, Comm 0.00
    2014-11-03, Close, 68.67
    2014-11-04, Close, 69.55
    2014-11-05, Close, 69.94
    2014-11-06, Close, 70.04
    2014-11-07, Close, 70.91


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-11-07, SELL CREATE, 70.91
    2014-11-10, SELL EXECUTED, Price: 78.60, Cost: 74.70, Comm 0.00
    2014-11-10, Close, 71.51
    2014-11-11, Close, 71.12
    2014-11-12, Close, 71.29
    2014-11-13, Close, 74.66
    2014-11-14, Close, 74.68


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-11-14, SELL CREATE, 74.68
    2014-11-17, SELL EXECUTED, Price: 82.58, Cost: 74.70, Comm 0.00
    2014-11-17, Close, 75.23
    2014-11-18, Close, 75.43
    2014-11-19, Close, 76.51
    2014-11-20, Close, 76.14
    2014-11-21, Close, 76.20


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-11-21, SELL CREATE, 76.20
    2014-11-24, SELL EXECUTED, Price: 84.85, Cost: 74.70, Comm 0.00
    2014-11-24, Close, 76.88
    2014-11-25, Close, 76.47
    2014-11-26, Close, 76.50
    2014-11-28, Close, 78.80
    2014-12-01, Close, 77.61


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-12-01, SELL CREATE, 77.61
    2014-12-02, SELL EXECUTED, Price: 86.27, Cost: 74.70, Comm 0.00
    2014-12-02, Close, 77.78
    2014-12-03, Close, 76.89
    2014-12-04, Close, 76.73
    2014-12-05, Close, 76.15
    2014-12-08, Close, 76.25


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-12-08, SELL CREATE, 76.25
    2014-12-09, SELL EXECUTED, Price: 83.65, Cost: 74.70, Comm 0.00
    2014-12-09, OPERATION PROFIT, GROSS 219.87, NET 219.87
    2014-12-09, Close, 75.64
    2014-12-10, Close, 75.11
    2014-12-11, Close, 75.88
    2014-12-12, Close, 75.87
    2014-12-15, Close, 75.98


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-12-16, Close, 75.10
    2014-12-17, Close, 76.25
    2014-12-18, Close, 77.79
    2014-12-19, Close, 77.09
    2014-12-22, Close, 78.19


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-12-23, Close, 78.45
    2014-12-24, Close, 78.24
    2014-12-26, Close, 78.67
    2014-12-29, Close, 78.43
    2014-12-30, Close, 78.56


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2014-12-31, Close, 77.74
    2015-01-02, Close, 77.76
    2015-01-05, Close, 77.53
    2015-01-06, Close, 78.13
    2015-01-07, Close, 80.20


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-01-08, Close, 81.89
    2015-01-09, Close, 80.88
    2015-01-12, Close, 81.49
    2015-01-13, Close, 80.84
    2015-01-14, Close, 78.40


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-01-15, Close, 79.10
    2015-01-16, Close, 78.55
    2015-01-20, Close, 78.47
    2015-01-21, Close, 78.43
    2015-01-22, Close, 79.93


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-01-23, Close, 80.12
    2015-01-26, Close, 80.23
    2015-01-27, Close, 79.23
    2015-01-28, Close, 78.59
    2015-01-29, Close, 79.40


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-01-30, Close, 76.92
    2015-02-02, Close, 77.59
    2015-02-03, Close, 78.02
    2015-02-04, Close, 78.44
    2015-02-05, Close, 79.01


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-02-06, Close, 79.05
    2015-02-09, Close, 77.77
    2015-02-10, Close, 79.02
    2015-02-11, Close, 78.16
    2015-02-12, Close, 77.75


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-02-13, Close, 77.68
    2015-02-17, Close, 77.81
    2015-02-18, Close, 78.11
    2015-02-19, Close, 75.60
    2015-02-20, Close, 76.31


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-02-23, Close, 76.58
    2015-02-24, Close, 76.55
    2015-02-25, Close, 75.65
    2015-02-26, Close, 75.86
    2015-02-27, Close, 75.97


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-03-02, Close, 76.00
    2015-03-03, Close, 75.47
    2015-03-04, Close, 74.75
    2015-03-05, Close, 75.65
    2015-03-06, Close, 74.76


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-03-09, Close, 75.02
    2015-03-10, Close, 74.29
    2015-03-11, Close, 73.48
    2015-03-12, Close, 74.58
    2015-03-13, Close, 74.58


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-03-16, Close, 75.85
    2015-03-17, Close, 75.24
    2015-03-18, Close, 75.16
    2015-03-19, Close, 74.24
    2015-03-20, Close, 75.80


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-03-23, Close, 75.87
    2015-03-24, Close, 75.63
    2015-03-25, Close, 74.05
    2015-03-26, Close, 74.57
    2015-03-27, Close, 74.08


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-03-27, BUY CREATE, 74.08
    2015-03-30, BUY EXECUTED, Price: 81.70, Cost: 81.70, Comm 0.00
    2015-03-30, Close, 75.16
    2015-03-31, Close, 74.90
    2015-04-01, Close, 73.50
    2015-04-02, Close, 73.52
    2015-04-06, Close, 73.75


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-04-06, BUY CREATE, 73.75
    2015-04-07, BUY EXECUTED, Price: 81.09, Cost: 81.09, Comm 0.00
    2015-04-07, Close, 73.31
    2015-04-08, Close, 73.79
    2015-04-09, Close, 73.62
    2015-04-10, Close, 73.44
    2015-04-13, Close, 73.12


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-04-13, BUY CREATE, 73.12
    2015-04-14, BUY EXECUTED, Price: 80.29, Cost: 80.29, Comm 0.00
    2015-04-14, Close, 72.99
    2015-04-15, Close, 72.61
    2015-04-16, Close, 72.16
    2015-04-17, Close, 70.92
    2015-04-20, Close, 71.16


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-04-20, BUY CREATE, 71.16
    2015-04-21, BUY EXECUTED, Price: 78.61, Cost: 78.61, Comm 0.00
    2015-04-21, Close, 71.06
    2015-04-22, Close, 71.42
    2015-04-23, Close, 72.10
    2015-04-24, Close, 72.71
    2015-04-27, Close, 72.28


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-04-27, BUY CREATE, 72.28
    2015-04-28, BUY EXECUTED, Price: 79.46, Cost: 79.46, Comm 0.00
    2015-04-28, Close, 72.03
    2015-04-29, Close, 70.92
    2015-04-30, Close, 71.08
    2015-05-01, Close, 71.58
    2015-05-04, Close, 72.10


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-05-04, BUY CREATE, 72.10
    2015-05-05, BUY EXECUTED, Price: 79.01, Cost: 79.01, Comm 0.00
    2015-05-05, Close, 71.15
    2015-05-06, Close, 71.16
    2015-05-07, Close, 71.51
    2015-05-08, Close, 71.96
    2015-05-11, Close, 71.57


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-05-11, BUY CREATE, 71.57
    2015-05-12, BUY EXECUTED, Price: 78.02, Cost: 78.02, Comm 0.00
    2015-05-12, Close, 72.36
    2015-05-13, Close, 71.63
    2015-05-14, Close, 72.14
    2015-05-15, Close, 72.62
    2015-05-18, Close, 73.24


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-05-18, BUY CREATE, 73.24
    2015-05-19, BUY EXECUTED, Price: 78.18, Cost: 78.18, Comm 0.00
    2015-05-19, Close, 70.04
    2015-05-20, Close, 69.55
    2015-05-21, Close, 69.75
    2015-05-22, Close, 69.52
    2015-05-26, Close, 68.64


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-05-26, BUY CREATE, 68.64
    2015-05-27, BUY EXECUTED, Price: 75.01, Cost: 75.01, Comm 0.00
    2015-05-27, Close, 68.90
    2015-05-28, Close, 68.58
    2015-05-29, Close, 68.06
    2015-06-01, Close, 68.48
    2015-06-02, Close, 68.30


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-06-02, BUY CREATE, 68.30
    2015-06-03, BUY EXECUTED, Price: 74.70, Cost: 74.70, Comm 0.00
    2015-06-03, Close, 68.63
    2015-06-04, Close, 67.95
    2015-06-05, Close, 66.95
    2015-06-08, Close, 66.54
    2015-06-09, Close, 66.41


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-06-09, BUY CREATE, 66.41
    2015-06-10, BUY EXECUTED, Price: 72.71, Cost: 72.71, Comm 0.00
    2015-06-10, Close, 66.83
    2015-06-11, Close, 66.84
    2015-06-12, Close, 66.37
    2015-06-15, Close, 65.92
    2015-06-16, Close, 66.30


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-06-16, BUY CREATE, 66.30
    2015-06-17, BUY EXECUTED, Price: 72.63, Cost: 72.63, Comm 0.00
    2015-06-17, Close, 66.65
    2015-06-18, Close, 66.88
    2015-06-19, Close, 66.66
    2015-06-22, Close, 66.70
    2015-06-23, Close, 66.50


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-06-23, BUY CREATE, 66.50
    2015-06-24, BUY EXECUTED, Price: 72.56, Cost: 72.56, Comm 0.00
    2015-06-24, Close, 66.33
    2015-06-25, Close, 65.85
    2015-06-26, Close, 66.09
    2015-06-29, Close, 65.45
    2015-06-30, Close, 65.00


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-06-30, BUY CREATE, 65.00
    2015-07-01, BUY EXECUTED, Price: 71.60, Cost: 71.60, Comm 0.00
    2015-07-01, Close, 65.87
    2015-07-02, Close, 65.85
    2015-07-06, Close, 66.47
    2015-07-07, Close, 67.62
    2015-07-08, Close, 66.95


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-07-08, BUY CREATE, 66.95
    2015-07-09, BUY EXECUTED, Price: 73.67, Cost: 73.67, Comm 0.00
    2015-07-09, Close, 66.70
    2015-07-10, Close, 67.01
    2015-07-13, Close, 67.70
    2015-07-14, Close, 67.62
    2015-07-15, Close, 67.49


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-07-15, BUY CREATE, 67.49
    2015-07-16, BUY EXECUTED, Price: 73.97, Cost: 73.97, Comm 0.00
    2015-07-16, Close, 67.66
    2015-07-17, Close, 67.25
    2015-07-20, Close, 66.99
    2015-07-21, Close, 66.66
    2015-07-22, Close, 67.04


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-07-22, BUY CREATE, 67.04
    2015-07-23, BUY EXECUTED, Price: 73.07, Cost: 73.07, Comm 0.00
    2015-07-23, Close, 66.45
    2015-07-24, Close, 65.60
    2015-07-27, Close, 65.41
    2015-07-28, Close, 66.07
    2015-07-29, Close, 66.19


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-07-29, BUY CREATE, 66.19
    2015-07-30, BUY EXECUTED, Price: 72.03, Cost: 72.03, Comm 0.00
    2015-07-30, Close, 66.13
    2015-07-31, Close, 65.96
    2015-08-03, Close, 66.15
    2015-08-04, Close, 66.21
    2015-08-05, Close, 67.82


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-08-05, BUY CREATE, 67.82
    2015-08-06, Order Canceled/Margin/Rejected
    2015-08-06, Close, 67.16
    2015-08-07, Close, 65.74
    2015-08-10, Close, 65.95
    2015-08-11, Close, 66.37
    2015-08-12, Close, 66.97


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-08-12, BUY CREATE, 66.97
    2015-08-13, Order Canceled/Margin/Rejected
    2015-08-13, Close, 66.53
    2015-08-14, Close, 66.78
    2015-08-17, Close, 66.35
    2015-08-18, Close, 64.11
    2015-08-19, Close, 63.27


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-08-19, BUY CREATE, 63.27
    2015-08-20, Order Canceled/Margin/Rejected
    2015-08-20, Close, 63.14
    2015-08-21, Close, 61.39
    2015-08-24, Close, 59.00
    2015-08-25, Close, 58.22
    2015-08-26, Close, 59.82


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-08-26, BUY CREATE, 59.82
    2015-08-27, Order Canceled/Margin/Rejected
    2015-08-27, Close, 60.97
    2015-08-28, Close, 59.92
    2015-08-31, Close, 59.72
    2015-09-01, Close, 58.88
    2015-09-02, Close, 59.46


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-09-02, BUY CREATE, 59.46
    2015-09-03, Order Canceled/Margin/Rejected
    2015-09-03, Close, 59.84
    2015-09-04, Close, 58.95
    2015-09-08, Close, 61.25
    2015-09-09, Close, 60.08
    2015-09-10, Close, 59.16


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-09-10, BUY CREATE, 59.16
    2015-09-11, Order Canceled/Margin/Rejected
    2015-09-11, Close, 59.65
    2015-09-14, Close, 59.31
    2015-09-15, Close, 59.34
    2015-09-16, Close, 59.69
    2015-09-17, Close, 59.48


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-09-17, BUY CREATE, 59.48
    2015-09-18, Order Canceled/Margin/Rejected
    2015-09-18, Close, 58.44
    2015-09-21, Close, 58.79
    2015-09-22, Close, 58.67
    2015-09-23, Close, 58.79
    2015-09-24, Close, 58.89


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-09-24, BUY CREATE, 58.89
    2015-09-25, Order Canceled/Margin/Rejected
    2015-09-25, Close, 58.85
    2015-09-28, Close, 58.74
    2015-09-29, Close, 58.85
    2015-09-30, Close, 59.82
    2015-10-01, Close, 59.30


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-10-01, BUY CREATE, 59.30
    2015-10-02, Order Canceled/Margin/Rejected
    2015-10-02, Close, 59.95
    2015-10-05, Close, 60.78
    2015-10-06, Close, 60.60
    2015-10-07, Close, 61.23
    2015-10-08, Close, 61.71


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-10-08, BUY CREATE, 61.71
    2015-10-09, Order Canceled/Margin/Rejected
    2015-10-09, Close, 61.53
    2015-10-12, Close, 61.75
    2015-10-13, Close, 61.57
    2015-10-14, Close, 55.39
    2015-10-15, Close, 54.74


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-10-15, BUY CREATE, 54.74
    2015-10-16, Order Canceled/Margin/Rejected
    2015-10-16, Close, 54.33
    2015-10-19, Close, 54.30
    2015-10-20, Close, 54.21
    2015-10-21, Close, 54.10
    2015-10-22, Close, 54.34


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-10-22, BUY CREATE, 54.34
    2015-10-23, Order Canceled/Margin/Rejected
    2015-10-23, Close, 53.79
    2015-10-26, Close, 53.53
    2015-10-27, Close, 53.03
    2015-10-28, Close, 53.18
    2015-10-29, Close, 53.48


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-10-29, BUY CREATE, 53.48
    2015-10-30, Order Canceled/Margin/Rejected
    2015-10-30, Close, 52.81
    2015-11-02, Close, 53.15
    2015-11-03, Close, 53.62
    2015-11-04, Close, 53.86
    2015-11-05, Close, 54.08


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-11-05, BUY CREATE, 54.08
    2015-11-06, Order Canceled/Margin/Rejected
    2015-11-06, Close, 54.23
    2015-11-09, Close, 53.97
    2015-11-10, Close, 54.14
    2015-11-11, Close, 53.13
    2015-11-12, Close, 52.54


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-11-12, BUY CREATE, 52.54
    2015-11-13, Order Canceled/Margin/Rejected
    2015-11-13, Close, 52.06
    2015-11-16, Close, 53.39
    2015-11-17, Close, 55.29
    2015-11-18, Close, 56.22
    2015-11-19, Close, 56.00


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-11-19, BUY CREATE, 56.00
    2015-11-20, Order Canceled/Margin/Rejected
    2015-11-20, Close, 55.42
    2015-11-23, Close, 55.60
    2015-11-24, Close, 55.29
    2015-11-25, Close, 55.58
    2015-11-27, Close, 55.26


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-11-27, BUY CREATE, 55.26
    2015-11-30, Order Canceled/Margin/Rejected
    2015-11-30, Close, 54.29
    2015-12-01, Close, 54.43
    2015-12-02, Close, 54.29
    2015-12-03, Close, 54.93
    2015-12-04, Close, 55.51


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-12-04, BUY CREATE, 55.51
    2015-12-07, Order Canceled/Margin/Rejected
    2015-12-07, Close, 56.29
    2015-12-08, Close, 55.46
    2015-12-09, Close, 55.01
    2015-12-10, Close, 55.41
    2015-12-11, Close, 55.23


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-12-11, BUY CREATE, 55.23
    2015-12-14, Order Canceled/Margin/Rejected
    2015-12-14, Close, 56.19
    2015-12-15, Close, 55.49
    2015-12-16, Close, 56.10
    2015-12-17, Close, 54.87
    2015-12-18, Close, 54.75


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-12-18, BUY CREATE, 54.75
    2015-12-21, Order Canceled/Margin/Rejected
    2015-12-21, Close, 55.40
    2015-12-22, Close, 56.33
    2015-12-23, Close, 56.84
    2015-12-24, Close, 56.59
    2015-12-28, Close, 56.52


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2015-12-28, BUY CREATE, 56.52
    2015-12-29, Order Canceled/Margin/Rejected
    2015-12-29, Close, 57.32
    2015-12-30, Close, 57.39
    2015-12-31, Close, 57.03
    2016-01-04, Close, 57.18
    2016-01-05, Close, 58.54


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-01-05, BUY CREATE, 58.54
    2016-01-06, Order Canceled/Margin/Rejected
    2016-01-06, Close, 59.13
    2016-01-07, Close, 60.50
    2016-01-08, Close, 59.12
    2016-01-11, Close, 59.75
    2016-01-12, Close, 59.19


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-01-12, BUY CREATE, 59.19
    2016-01-13, Order Canceled/Margin/Rejected
    2016-01-13, Close, 57.61
    2016-01-14, Close, 58.67
    2016-01-15, Close, 57.62
    2016-01-19, Close, 58.20
    2016-01-20, Close, 56.60


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-01-20, BUY CREATE, 56.60
    2016-01-21, Order Canceled/Margin/Rejected
    2016-01-21, Close, 57.57
    2016-01-22, Close, 58.33
    2016-01-25, Close, 59.03
    2016-01-26, Close, 59.54
    2016-01-27, Close, 59.50


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-01-27, BUY CREATE, 59.50
    2016-01-28, Order Canceled/Margin/Rejected
    2016-01-28, Close, 59.75
    2016-01-29, Close, 61.74
    2016-02-01, Close, 62.80
    2016-02-02, Close, 62.21
    2016-02-03, Close, 61.66


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-02-03, SELL CREATE, 61.66
    2016-02-04, SELL EXECUTED, Price: 65.76, Cost: 76.02, Comm 0.00
    2016-02-04, Close, 61.80
    2016-02-05, Close, 62.34
    2016-02-08, Close, 62.24
    2016-02-09, Close, 61.23
    2016-02-10, Close, 61.21


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-02-10, SELL CREATE, 61.21
    2016-02-11, SELL EXECUTED, Price: 65.02, Cost: 76.02, Comm 0.00
    2016-02-11, Close, 60.77
    2016-02-12, Close, 61.57
    2016-02-16, Close, 61.31
    2016-02-17, Close, 61.51
    2016-02-18, Close, 59.66


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-02-18, SELL CREATE, 59.66
    2016-02-19, SELL EXECUTED, Price: 63.89, Cost: 76.02, Comm 0.00
    2016-02-19, Close, 60.16
    2016-02-22, Close, 61.06
    2016-02-23, Close, 61.85
    2016-02-24, Close, 62.45
    2016-02-25, Close, 63.30


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-02-25, SELL CREATE, 63.30
    2016-02-26, SELL EXECUTED, Price: 68.08, Cost: 76.02, Comm 0.00
    2016-02-26, Close, 61.88
    2016-02-29, Close, 61.72
    2016-03-01, Close, 61.83
    2016-03-02, Close, 61.60
    2016-03-03, Close, 61.54


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-03-03, SELL CREATE, 61.54
    2016-03-04, SELL EXECUTED, Price: 66.14, Cost: 76.02, Comm 0.00
    2016-03-04, Close, 62.13
    2016-03-07, Close, 63.16
    2016-03-08, Close, 63.30
    2016-03-09, Close, 63.29
    2016-03-10, Close, 63.18


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-03-10, SELL CREATE, 63.18
    2016-03-11, SELL EXECUTED, Price: 67.77, Cost: 76.02, Comm 0.00
    2016-03-11, Close, 62.96
    2016-03-14, Close, 63.13
    2016-03-15, Close, 63.82
    2016-03-16, Close, 63.72
    2016-03-17, Close, 63.22


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-03-17, SELL CREATE, 63.22
    2016-03-18, SELL EXECUTED, Price: 67.48, Cost: 76.02, Comm 0.00
    2016-03-18, Close, 62.75
    2016-03-21, Close, 63.71
    2016-03-22, Close, 63.61
    2016-03-23, Close, 63.23
    2016-03-24, Close, 63.73


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-03-24, SELL CREATE, 63.73
    2016-03-28, SELL EXECUTED, Price: 67.93, Cost: 76.02, Comm 0.00
    2016-03-28, Close, 63.85
    2016-03-29, Close, 63.76
    2016-03-30, Close, 64.48
    2016-03-31, Close, 64.19
    2016-04-01, Close, 64.73


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-04-01, SELL CREATE, 64.73
    2016-04-04, SELL EXECUTED, Price: 69.00, Cost: 76.02, Comm 0.00
    2016-04-04, Close, 64.77
    2016-04-05, Close, 64.33
    2016-04-06, Close, 64.71
    2016-04-07, Close, 63.94
    2016-04-08, Close, 63.79


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-04-08, SELL CREATE, 63.79
    2016-04-11, SELL EXECUTED, Price: 68.01, Cost: 76.02, Comm 0.00
    2016-04-11, Close, 63.17
    2016-04-12, Close, 64.48
    2016-04-13, Close, 64.81
    2016-04-14, Close, 64.48
    2016-04-15, Close, 64.73


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-04-15, SELL CREATE, 64.73
    2016-04-18, SELL EXECUTED, Price: 69.05, Cost: 76.02, Comm 0.00
    2016-04-18, Close, 65.48
    2016-04-19, Close, 65.39
    2016-04-20, Close, 64.87
    2016-04-21, Close, 64.17
    2016-04-22, Close, 64.41


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-04-22, SELL CREATE, 64.41
    2016-04-25, SELL EXECUTED, Price: 68.55, Cost: 76.02, Comm 0.00
    2016-04-25, Close, 65.11
    2016-04-26, Close, 64.95
    2016-04-27, Close, 65.07
    2016-04-28, Close, 64.59
    2016-04-29, Close, 62.68


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-04-29, SELL CREATE, 62.68
    2016-05-02, SELL EXECUTED, Price: 66.62, Cost: 76.02, Comm 0.00
    2016-05-02, Close, 63.35
    2016-05-03, Close, 62.80
    2016-05-04, Close, 62.97
    2016-05-05, Close, 62.99
    2016-05-06, Close, 63.97


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-05-06, SELL CREATE, 63.97
    2016-05-09, SELL EXECUTED, Price: 68.25, Cost: 76.02, Comm 0.00
    2016-05-09, Close, 64.62
    2016-05-10, Close, 64.47
    2016-05-11, Close, 62.70
    2016-05-12, Close, 63.12
    2016-05-13, Close, 61.31


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-05-13, SELL CREATE, 61.31
    2016-05-16, SELL EXECUTED, Price: 64.86, Cost: 76.02, Comm 0.00
    2016-05-16, Close, 62.33
    2016-05-17, Close, 61.46
    2016-05-18, Close, 59.62
    2016-05-19, Close, 65.33
    2016-05-20, Close, 65.96


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-05-20, SELL CREATE, 65.96
    2016-05-23, SELL EXECUTED, Price: 69.60, Cost: 76.02, Comm 0.00
    2016-05-23, Close, 65.62
    2016-05-24, Close, 66.32
    2016-05-25, Close, 66.54
    2016-05-26, Close, 66.89
    2016-05-27, Close, 66.80


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-05-27, SELL CREATE, 66.80
    2016-05-31, SELL EXECUTED, Price: 70.58, Cost: 76.02, Comm 0.00
    2016-05-31, Close, 66.83
    2016-06-01, Close, 66.56
    2016-06-02, Close, 66.99
    2016-06-03, Close, 66.91
    2016-06-06, Close, 67.08


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-06-06, SELL CREATE, 67.08
    2016-06-07, SELL EXECUTED, Price: 71.10, Cost: 76.02, Comm 0.00
    2016-06-07, OPERATION PROFIT, GROSS -150.62, NET -150.62
    2016-06-07, Close, 67.06
    2016-06-08, Close, 67.30
    2016-06-09, Close, 67.12
    2016-06-10, Close, 67.17
    2016-06-13, Close, 66.59


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-06-14, Close, 66.99
    2016-06-15, Close, 67.15
    2016-06-16, Close, 67.32
    2016-06-17, Close, 66.99
    2016-06-20, Close, 67.13


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-06-21, Close, 67.47
    2016-06-22, Close, 67.74
    2016-06-23, Close, 68.07
    2016-06-24, Close, 67.94
    2016-06-27, Close, 67.51


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-06-28, Close, 67.51
    2016-06-29, Close, 68.41
    2016-06-30, Close, 68.94
    2016-07-01, Close, 68.74
    2016-07-05, Close, 69.05


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-07-06, Close, 69.70
    2016-07-07, Close, 69.42
    2016-07-08, Close, 69.71
    2016-07-11, Close, 69.92
    2016-07-12, Close, 69.18


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-07-13, Close, 69.51
    2016-07-14, Close, 69.58
    2016-07-15, Close, 69.55
    2016-07-18, Close, 69.71
    2016-07-19, Close, 69.54


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-07-20, Close, 69.67
    2016-07-21, Close, 69.41
    2016-07-22, Close, 69.44
    2016-07-25, Close, 69.63
    2016-07-26, Close, 69.61


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-07-27, Close, 69.22
    2016-07-28, Close, 69.15
    2016-07-29, Close, 68.89
    2016-08-01, Close, 69.66
    2016-08-02, Close, 69.04


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-08-03, Close, 68.86
    2016-08-04, Close, 69.20
    2016-08-05, Close, 69.64
    2016-08-08, Close, 69.24
    2016-08-09, Close, 69.43


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-08-10, Close, 70.30
    2016-08-11, Close, 70.15
    2016-08-12, Close, 70.24
    2016-08-15, Close, 69.70
    2016-08-16, Close, 69.29


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-08-17, Close, 69.33
    2016-08-18, Close, 70.63
    2016-08-19, Close, 69.21
    2016-08-22, Close, 69.11
    2016-08-23, Close, 68.41


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-08-24, Close, 68.66
    2016-08-25, Close, 67.70
    2016-08-26, Close, 67.63
    2016-08-29, Close, 67.87
    2016-08-30, Close, 67.79


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-08-31, Close, 67.91
    2016-09-01, Close, 69.24
    2016-09-02, Close, 68.92
    2016-09-06, Close, 69.39
    2016-09-07, Close, 68.50


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-09-08, Close, 68.28
    2016-09-09, Close, 66.83
    2016-09-12, Close, 68.39
    2016-09-13, Close, 67.93
    2016-09-14, Close, 67.99


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-09-15, Close, 68.82
    2016-09-16, Close, 69.27
    2016-09-19, Close, 68.53
    2016-09-20, Close, 68.41
    2016-09-21, Close, 68.62


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-09-22, Close, 68.70
    2016-09-23, Close, 68.78
    2016-09-26, Close, 68.08
    2016-09-27, Close, 68.76
    2016-09-28, Close, 68.24


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-09-29, Close, 67.24
    2016-09-30, Close, 68.56
    2016-10-03, Close, 68.45
    2016-10-04, Close, 68.21
    2016-10-05, Close, 68.13


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-10-06, Close, 65.93
    2016-10-07, Close, 65.31
    2016-10-10, Close, 64.62
    2016-10-11, Close, 64.06
    2016-10-12, Close, 64.13


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-10-13, Close, 64.86
    2016-10-14, Close, 65.07
    2016-10-17, Close, 64.85
    2016-10-18, Close, 65.47
    2016-10-19, Close, 65.49


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-10-20, Close, 65.33
    2016-10-21, Close, 64.96
    2016-10-24, Close, 65.77
    2016-10-25, Close, 65.93
    2016-10-26, Close, 66.15


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-10-27, Close, 66.38
    2016-10-28, Close, 66.53
    2016-10-31, Close, 66.56
    2016-11-01, Close, 65.88
    2016-11-02, Close, 66.02


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-11-03, Close, 66.19
    2016-11-04, Close, 65.74
    2016-11-07, Close, 66.33
    2016-11-08, Close, 66.34
    2016-11-09, Close, 67.59


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-11-10, Close, 67.86
    2016-11-11, Close, 67.71
    2016-11-14, Close, 67.01
    2016-11-15, Close, 67.89
    2016-11-16, Close, 67.86


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-11-17, Close, 65.77
    2016-11-18, Close, 65.15
    2016-11-21, Close, 65.94
    2016-11-22, Close, 66.66
    2016-11-23, Close, 67.33


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-11-25, Close, 67.71
    2016-11-28, Close, 67.67
    2016-11-29, Close, 67.84
    2016-11-30, Close, 66.95
    2016-12-01, Close, 67.18


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-12-02, Close, 67.38
    2016-12-05, Close, 66.48
    2016-12-06, Close, 66.88
    2016-12-07, Close, 67.59
    2016-12-08, Close, 67.34


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-12-09, Close, 67.09
    2016-12-12, Close, 68.62
    2016-12-13, Close, 68.74
    2016-12-14, Close, 68.30
    2016-12-15, Close, 68.05


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-12-16, Close, 67.96
    2016-12-19, Close, 68.53
    2016-12-20, Close, 68.76
    2016-12-21, Close, 68.20
    2016-12-22, Close, 66.63


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2016-12-23, Close, 66.58
    2016-12-27, Close, 66.73
    2016-12-28, Close, 66.36
    2016-12-29, Close, 66.31
    2016-12-30, Close, 66.18


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-01-03, Close, 65.73
    2017-01-04, Close, 66.12
    2017-01-05, Close, 66.26
    2017-01-06, Close, 65.35
    2017-01-09, Close, 65.78


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-01-10, Close, 65.32
    2017-01-11, Close, 65.61
    2017-01-12, Close, 65.07
    2017-01-13, Close, 64.27
    2017-01-17, Close, 65.51


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-01-18, Close, 65.21
    2017-01-19, Close, 64.74
    2017-01-20, Close, 64.32
    2017-01-23, Close, 63.81
    2017-01-24, Close, 64.53


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-01-24, BUY CREATE, 64.53
    2017-01-25, BUY EXECUTED, Price: 67.52, Cost: 67.52, Comm 0.00
    2017-01-25, Close, 64.04
    2017-01-26, Close, 63.89
    2017-01-27, Close, 62.86
    2017-01-30, Close, 63.59
    2017-01-31, Close, 63.90


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-01-31, BUY CREATE, 63.90
    2017-02-01, BUY EXECUTED, Price: 66.46, Cost: 66.46, Comm 0.00
    2017-02-01, Close, 63.41
    2017-02-02, Close, 63.86
    2017-02-03, Close, 63.67
    2017-02-06, Close, 63.57
    2017-02-07, Close, 64.04


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-02-07, BUY CREATE, 64.04
    2017-02-08, BUY EXECUTED, Price: 66.89, Cost: 66.89, Comm 0.00
    2017-02-08, Close, 64.92
    2017-02-09, Close, 66.14
    2017-02-10, Close, 65.12
    2017-02-13, Close, 64.88
    2017-02-14, Close, 65.73


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-02-14, SELL CREATE, 65.73
    2017-02-15, SELL EXECUTED, Price: 67.80, Cost: 66.96, Comm 0.00
    2017-02-15, Close, 65.76
    2017-02-16, Close, 65.94
    2017-02-17, Close, 66.41
    2017-02-21, Close, 68.41
    2017-02-22, Close, 68.65


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-02-22, SELL CREATE, 68.65
    2017-02-23, SELL EXECUTED, Price: 72.00, Cost: 66.96, Comm 0.00
    2017-02-23, Close, 68.27
    2017-02-24, Close, 69.31
    2017-02-27, Close, 68.68
    2017-02-28, Close, 67.91
    2017-03-01, Close, 67.45


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-03-01, SELL CREATE, 67.45
    2017-03-02, SELL EXECUTED, Price: 70.42, Cost: 66.96, Comm 0.00
    2017-03-02, OPERATION PROFIT, GROSS 9.35, NET 9.35
    2017-03-02, Close, 67.75
    2017-03-03, Close, 67.05
    2017-03-06, Close, 66.90
    2017-03-07, Close, 66.89
    2017-03-08, Close, 67.32


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-03-09, Close, 67.38
    2017-03-10, Close, 67.61
    2017-03-13, Close, 67.46
    2017-03-14, Close, 68.20
    2017-03-15, Close, 68.07


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-03-16, Close, 67.93
    2017-03-17, Close, 67.40
    2017-03-20, Close, 67.49
    2017-03-21, Close, 67.41
    2017-03-22, Close, 67.75


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-03-22, BUY CREATE, 67.75
    2017-03-23, BUY EXECUTED, Price: 70.20, Cost: 70.20, Comm 0.00
    2017-03-23, Close, 67.38
    2017-03-24, Close, 67.13
    2017-03-27, Close, 67.18
    2017-03-28, Close, 67.82
    2017-03-29, Close, 68.22


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-03-29, BUY CREATE, 68.22
    2017-03-30, BUY EXECUTED, Price: 70.69, Cost: 70.69, Comm 0.00
    2017-03-30, Close, 69.04
    2017-03-31, Close, 69.52
    2017-04-03, Close, 69.28
    2017-04-04, Close, 69.45
    2017-04-05, Close, 69.10


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-04-05, BUY CREATE, 69.10
    2017-04-06, BUY EXECUTED, Price: 71.70, Cost: 71.70, Comm 0.00
    2017-04-06, Close, 68.89
    2017-04-07, Close, 70.31
    2017-04-10, Close, 70.46
    2017-04-11, Close, 70.82
    2017-04-12, Close, 70.83


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-04-12, SELL CREATE, 70.83
    2017-04-13, SELL EXECUTED, Price: 73.37, Cost: 70.86, Comm 0.00
    2017-04-13, Close, 70.55
    2017-04-17, Close, 70.88
    2017-04-18, Close, 71.26
    2017-04-19, Close, 71.44
    2017-04-20, Close, 72.14


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-04-20, SELL CREATE, 72.14
    2017-04-21, SELL EXECUTED, Price: 74.74, Cost: 70.86, Comm 0.00
    2017-04-21, Close, 72.27
    2017-04-24, Close, 72.12
    2017-04-25, Close, 72.38
    2017-04-26, Close, 72.75
    2017-04-27, Close, 72.76


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-04-27, SELL CREATE, 72.76
    2017-04-28, SELL EXECUTED, Price: 75.24, Cost: 70.86, Comm 0.00
    2017-04-28, OPERATION PROFIT, GROSS 10.76, NET 10.76
    2017-04-28, Close, 72.51
    2017-05-01, Close, 72.55
    2017-05-02, Close, 72.83
    2017-05-03, Close, 73.07
    2017-05-04, Close, 73.63


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-05-05, Close, 73.78
    2017-05-08, Close, 73.41
    2017-05-09, Close, 73.99
    2017-05-10, Close, 74.47
    2017-05-11, Close, 73.91


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-05-12, Close, 73.51
    2017-05-15, Close, 74.07
    2017-05-16, Close, 72.92
    2017-05-17, Close, 72.93
    2017-05-18, Close, 75.28


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-05-19, Close, 76.48
    2017-05-22, Close, 76.26
    2017-05-23, Close, 76.21
    2017-05-24, Close, 75.88
    2017-05-25, Close, 76.03


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-05-26, Close, 75.86
    2017-05-30, Close, 75.88
    2017-05-31, Close, 76.31
    2017-06-01, Close, 77.49
    2017-06-02, Close, 77.30


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-06-05, Close, 77.92
    2017-06-06, Close, 76.63
    2017-06-07, Close, 76.85
    2017-06-08, Close, 76.63
    2017-06-09, Close, 77.11


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-06-12, Close, 76.93
    2017-06-13, Close, 77.21
    2017-06-14, Close, 77.57
    2017-06-15, Close, 76.61
    2017-06-16, Close, 73.05


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-06-19, Close, 73.30
    2017-06-20, Close, 73.34
    2017-06-21, Close, 74.02
    2017-06-22, Close, 73.32
    2017-06-23, Close, 72.66


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-06-26, Close, 73.30
    2017-06-27, Close, 73.80
    2017-06-28, Close, 74.28
    2017-06-29, Close, 73.72
    2017-06-30, Close, 73.48


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-07-03, Close, 73.17
    2017-07-05, Close, 73.13
    2017-07-06, Close, 73.27
    2017-07-07, Close, 73.14
    2017-07-10, Close, 71.10


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-07-10, BUY CREATE, 71.10
    2017-07-11, BUY EXECUTED, Price: 73.38, Cost: 73.38, Comm 0.00
    2017-07-11, Close, 71.33
    2017-07-12, Close, 71.79
    2017-07-13, Close, 72.87
    2017-07-14, Close, 74.12
    2017-07-17, Close, 74.15


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-07-17, BUY CREATE, 74.15
    2017-07-18, BUY EXECUTED, Price: 76.25, Cost: 76.25, Comm 0.00
    2017-07-18, Close, 73.98
    2017-07-19, Close, 73.66
    2017-07-20, Close, 73.81
    2017-07-21, Close, 73.93
    2017-07-24, Close, 74.65


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-07-24, BUY CREATE, 74.65
    2017-07-25, BUY EXECUTED, Price: 77.61, Cost: 77.61, Comm 0.00
    2017-07-25, Close, 76.23
    2017-07-26, Close, 76.60
    2017-07-27, Close, 77.46
    2017-07-28, Close, 77.49
    2017-07-31, Close, 77.66


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-07-31, SELL CREATE, 77.66
    2017-08-01, SELL EXECUTED, Price: 80.25, Cost: 75.75, Comm 0.00
    2017-08-01, Close, 78.16
    2017-08-02, Close, 78.19
    2017-08-03, Close, 78.52
    2017-08-04, Close, 78.14
    2017-08-07, Close, 78.91


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-08-07, SELL CREATE, 78.91
    2017-08-08, SELL EXECUTED, Price: 81.17, Cost: 75.75, Comm 0.00
    2017-08-08, Close, 79.21
    2017-08-09, Close, 79.73
    2017-08-10, Close, 78.80
    2017-08-11, Close, 78.55
    2017-08-14, Close, 78.84


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-08-14, SELL CREATE, 78.84
    2017-08-15, SELL EXECUTED, Price: 80.83, Cost: 75.75, Comm 0.00
    2017-08-15, OPERATION PROFIT, GROSS 15.01, NET 15.01
    2017-08-15, Close, 78.91
    2017-08-16, Close, 79.12
    2017-08-17, Close, 77.87
    2017-08-18, Close, 77.49
    2017-08-21, Close, 77.88


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-08-22, Close, 78.18
    2017-08-23, Close, 78.12
    2017-08-24, Close, 76.54
    2017-08-25, Close, 76.82
    2017-08-28, Close, 76.24


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-08-29, Close, 76.96
    2017-08-30, Close, 76.73
    2017-08-31, Close, 76.27
    2017-09-01, Close, 76.57
    2017-09-05, Close, 77.96


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-09-06, Close, 78.24
    2017-09-07, Close, 78.28
    2017-09-08, Close, 77.07
    2017-09-11, Close, 77.26
    2017-09-12, Close, 77.78


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-09-13, Close, 78.02
    2017-09-14, Close, 77.85
    2017-09-15, Close, 78.53
    2017-09-18, Close, 78.16
    2017-09-19, Close, 78.21


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-09-20, Close, 78.65
    2017-09-21, Close, 78.17
    2017-09-22, Close, 77.70
    2017-09-25, Close, 77.33
    2017-09-26, Close, 77.56


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-09-27, Close, 77.47
    2017-09-28, Close, 77.13
    2017-09-29, Close, 76.34
    2017-10-02, Close, 76.65
    2017-10-03, Close, 77.40


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-10-04, Close, 77.27
    2017-10-05, Close, 77.58
    2017-10-06, Close, 77.18
    2017-10-09, Close, 78.68
    2017-10-10, Close, 82.19


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-10-11, Close, 83.76
    2017-10-12, Close, 84.12
    2017-10-13, Close, 84.63
    2017-10-16, Close, 83.77
    2017-10-17, Close, 84.00


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-10-18, Close, 84.24
    2017-10-19, Close, 84.41
    2017-10-20, Close, 85.43
    2017-10-23, Close, 86.61
    2017-10-24, Close, 85.96


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-10-25, Close, 86.44
    2017-10-26, Close, 86.58
    2017-10-27, Close, 86.14
    2017-10-30, Close, 84.95
    2017-10-31, Close, 85.30


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-11-01, Close, 85.92
    2017-11-02, Close, 86.76
    2017-11-03, Close, 87.62
    2017-11-06, Close, 86.66
    2017-11-07, Close, 86.90


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-11-08, Close, 88.18
    2017-11-09, Close, 88.22
    2017-11-10, Close, 88.83
    2017-11-13, Close, 88.90
    2017-11-14, Close, 88.99


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-11-15, Close, 87.76
    2017-11-16, Close, 97.33
    2017-11-17, Close, 95.23
    2017-11-20, Close, 95.24
    2017-11-21, Close, 94.30


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-11-22, Close, 94.19
    2017-11-24, Close, 94.40
    2017-11-27, Close, 94.40
    2017-11-28, Close, 94.54
    2017-11-29, Close, 95.32


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-11-30, Close, 94.99
    2017-12-01, Close, 95.11
    2017-12-04, Close, 94.78
    2017-12-05, Close, 95.58
    2017-12-06, Close, 95.04


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-12-07, Close, 95.05
    2017-12-08, Close, 94.83
    2017-12-11, Close, 95.20
    2017-12-12, Close, 94.97
    2017-12-13, Close, 96.01


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-12-14, Close, 95.40
    2017-12-15, Close, 95.38
    2017-12-18, Close, 96.15
    2017-12-19, Close, 97.04
    2017-12-20, Close, 96.99


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-12-21, Close, 96.31
    2017-12-22, Close, 96.46
    2017-12-26, Close, 97.39
    2017-12-27, Close, 97.49
    2017-12-28, Close, 97.63


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2017-12-29, Close, 96.99
    2018-01-02, Close, 96.83
    2018-01-03, Close, 97.67
    2018-01-04, Close, 97.76
    2018-01-05, Close, 98.34


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-01-08, Close, 99.80
    2018-01-09, Close, 98.60
    2018-01-10, Close, 97.89
    2018-01-11, Close, 98.23
    2018-01-12, Close, 99.07


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-01-16, Close, 98.89
    2018-01-17, Close, 100.87
    2018-01-18, Close, 102.44
    2018-01-19, Close, 102.72
    2018-01-22, Close, 103.57


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-01-23, Close, 104.01
    2018-01-24, Close, 103.90
    2018-01-25, Close, 104.70
    2018-01-26, Close, 106.45
    2018-01-29, Close, 107.59


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-01-30, Close, 105.81
    2018-01-31, Close, 104.70
    2018-02-01, Close, 103.64
    2018-02-02, Close, 102.61
    2018-02-05, Close, 98.30


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-02-06, Close, 99.10
    2018-02-07, Close, 101.01
    2018-02-08, Close, 98.23
    2018-02-09, Close, 97.60
    2018-02-12, Close, 97.77


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-02-13, Close, 99.18
    2018-02-14, Close, 99.88
    2018-02-15, Close, 101.39
    2018-02-16, Close, 102.91
    2018-02-20, Close, 92.43


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-02-21, Close, 89.89
    2018-02-22, Close, 91.11
    2018-02-23, Close, 91.23
    2018-02-26, Close, 91.46
    2018-02-27, Close, 89.89


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-02-28, Close, 88.40
    2018-03-01, Close, 87.49
    2018-03-02, Close, 87.19
    2018-03-05, Close, 88.37
    2018-03-06, Close, 87.47


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-03-06, BUY CREATE, 87.47
    2018-03-07, BUY EXECUTED, Price: 87.98, Cost: 87.98, Comm 0.00
    2018-03-07, Close, 86.17
    2018-03-08, Close, 86.87
    2018-03-09, Close, 87.66
    2018-03-12, Close, 87.01
    2018-03-13, Close, 87.24


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-03-13, BUY CREATE, 87.24
    2018-03-14, BUY EXECUTED, Price: 88.51, Cost: 88.51, Comm 0.00
    2018-03-14, Close, 86.62
    2018-03-15, Close, 86.46
    2018-03-16, Close, 88.10
    2018-03-19, Close, 86.40
    2018-03-20, Close, 86.89


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-03-20, BUY CREATE, 86.89
    2018-03-21, BUY EXECUTED, Price: 87.89, Cost: 87.89, Comm 0.00
    2018-03-21, Close, 87.12
    2018-03-22, Close, 86.09
    2018-03-23, Close, 84.40
    2018-03-26, Close, 86.45
    2018-03-27, Close, 85.02


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-03-27, BUY CREATE, 85.02
    2018-03-28, BUY EXECUTED, Price: 86.26, Cost: 86.26, Comm 0.00
    2018-03-28, Close, 86.72
    2018-03-29, Close, 87.90
    2018-04-02, Close, 84.52
    2018-04-03, Close, 85.76
    2018-04-04, Close, 86.17


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-04-04, BUY CREATE, 86.17
    2018-04-05, BUY EXECUTED, Price: 87.60, Cost: 87.60, Comm 0.00
    2018-04-05, Close, 86.76
    2018-04-06, Close, 85.65
    2018-04-09, Close, 85.24
    2018-04-10, Close, 85.41
    2018-04-11, Close, 84.88


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-04-11, BUY CREATE, 84.88
    2018-04-12, BUY EXECUTED, Price: 86.19, Cost: 86.19, Comm 0.00
    2018-04-12, Close, 84.40
    2018-04-13, Close, 84.99
    2018-04-16, Close, 85.80
    2018-04-17, Close, 86.85
    2018-04-18, Close, 86.52


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-04-18, BUY CREATE, 86.52
    2018-04-19, BUY EXECUTED, Price: 87.41, Cost: 87.41, Comm 0.00
    2018-04-19, Close, 86.84
    2018-04-20, Close, 85.94
    2018-04-23, Close, 85.07
    2018-04-24, Close, 85.49
    2018-04-25, Close, 86.12


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-04-25, BUY CREATE, 86.12
    2018-04-26, BUY EXECUTED, Price: 87.17, Cost: 87.17, Comm 0.00
    2018-04-26, Close, 86.88
    2018-04-27, Close, 86.24
    2018-04-30, Close, 87.40
    2018-05-01, Close, 86.36
    2018-05-02, Close, 85.30


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-05-02, BUY CREATE, 85.30
    2018-05-03, BUY EXECUTED, Price: 86.19, Cost: 86.19, Comm 0.00
    2018-05-03, Close, 85.20
    2018-05-04, Close, 86.48
    2018-05-07, Close, 84.44
    2018-05-08, Close, 84.71
    2018-05-09, Close, 82.06


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-05-09, BUY CREATE, 82.06
    2018-05-10, BUY EXECUTED, Price: 82.64, Cost: 82.64, Comm 0.00
    2018-05-10, Close, 82.21
    2018-05-11, Close, 82.90
    2018-05-14, Close, 83.90
    2018-05-15, Close, 84.03
    2018-05-16, Close, 85.63


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-05-16, BUY CREATE, 85.63
    2018-05-17, BUY EXECUTED, Price: 87.04, Cost: 87.04, Comm 0.00
    2018-05-17, Close, 84.00
    2018-05-18, Close, 83.16
    2018-05-21, Close, 84.02
    2018-05-22, Close, 82.89
    2018-05-23, Close, 82.53


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-05-23, BUY CREATE, 82.53
    2018-05-24, BUY EXECUTED, Price: 83.00, Cost: 83.00, Comm 0.00
    2018-05-24, Close, 82.37
    2018-05-25, Close, 81.98
    2018-05-29, Close, 81.92
    2018-05-30, Close, 83.63
    2018-05-31, Close, 82.06


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-05-31, BUY CREATE, 82.06
    2018-06-01, BUY EXECUTED, Price: 83.04, Cost: 83.04, Comm 0.00
    2018-06-01, Close, 82.51
    2018-06-04, Close, 84.93
    2018-06-05, Close, 84.13
    2018-06-06, Close, 84.07
    2018-06-07, Close, 84.46


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-06-07, BUY CREATE, 84.46
    2018-06-08, BUY EXECUTED, Price: 84.78, Cost: 84.78, Comm 0.00
    2018-06-08, Close, 83.87
    2018-06-11, Close, 83.81
    2018-06-12, Close, 83.61
    2018-06-13, Close, 83.60
    2018-06-14, Close, 83.31


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-06-14, BUY CREATE, 83.31
    2018-06-15, Order Canceled/Margin/Rejected
    2018-06-15, Close, 83.22
    2018-06-18, Close, 82.52
    2018-06-19, Close, 83.13
    2018-06-20, Close, 83.13
    2018-06-21, Close, 83.72


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-06-21, BUY CREATE, 83.72
    2018-06-22, Order Canceled/Margin/Rejected
    2018-06-22, Close, 84.33
    2018-06-25, Close, 85.97
    2018-06-26, Close, 85.48
    2018-06-27, Close, 86.39
    2018-06-28, Close, 85.36


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-06-28, BUY CREATE, 85.36
    2018-06-29, Order Canceled/Margin/Rejected
    2018-06-29, Close, 85.16
    2018-07-02, Close, 83.51
    2018-07-03, Close, 83.95
    2018-07-05, Close, 84.08
    2018-07-06, Close, 84.02


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-07-06, BUY CREATE, 84.02
    2018-07-09, Order Canceled/Margin/Rejected
    2018-07-09, Close, 85.43
    2018-07-10, Close, 86.71
    2018-07-11, Close, 86.03
    2018-07-12, Close, 86.02
    2018-07-13, Close, 87.19


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-07-13, BUY CREATE, 87.19
    2018-07-16, Order Canceled/Margin/Rejected
    2018-07-16, Close, 87.13
    2018-07-17, Close, 87.68
    2018-07-18, Close, 87.56
    2018-07-19, Close, 87.21
    2018-07-20, Close, 87.55


    INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.


    2018-07-20, BUY CREATE, 87.55
    2018-07-23, Order Canceled/Margin/Rejected
    2018-07-23, Close, 87.12
    2018-07-24, Close, 87.45
    2018-07-25, Close, 87.39
    2018-07-26, Close, 87.72
    2018-07-27, Close, 87.62
    2018-07-27, BUY CREATE, 87.62
    2018-07-30, Order Canceled/Margin/Rejected
    2018-07-30, Close, 88.37
    2018-07-31, Close, 88.71
    Final Portfolio Value: 1301.42


### Conclusion

Wow, we have an ending porfolio value of \$1301.42! I'll have to admit that I'm a bit surprised. Although our gains probably didn't beat inflation or compensate for brokerage fees, with a lot of further investigation and tweaking, this algorithm might be viable.

But if we step back, I think this simple example is important for a greater reason. With algo-trading, we can see the power of using data to your advantage. Instead of taking financial risk and using an algorithm right away, you can look back in time and see how it would've preformed. Today's market is of course always different than yesterday's, but backtrading can give you meaningful insight into whether or not you should go forward with using an algorithm.

The next step will be to move to paper trading in order to see how well your algorithm would perform if you were to deploy it. Part 3?

### Disclaimer

The above references an opinion and is for information purposes only.  It is not intended to be investment advice.  Seek a duly licensed professional for investment advice.