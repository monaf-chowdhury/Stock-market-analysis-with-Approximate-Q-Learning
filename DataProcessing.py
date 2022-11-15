# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 02:22:13 2022

@author: monaf
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime

np.set_printoptions(suppress=True) # this stops making any scientific values in 'e' format 

gp = pd.read_csv("GRAE Historical Data.csv")
gp.name= 'GP'
bx = pd.read_csv("BEXI Historical Data.csv")
bx.name = 'BX'
sq = pd.read_csv("SQPH Historical Data.csv")
sq.name='SQ'

# moving average convergence divergence 
def calculating_macd(stock):
    slow = stock['Price'].ewm(span = 26,adjust = False ).mean()
    fast = stock['Price'].ewm(span=12,adjust = False).mean()
    macd = fast - slow 
    signal = macd.ewm(span=9,adjust= False).mean()                 # exponentially ma of macd
    bins = macd - signal                                           # histogram to reveal difference between MACD and Signal
    stock[stock.name+'_MACD'] = macd
    return stock

# Relative strength index
def calculating_rsi(stock,time=14):
    price = stock['Price']
    ret = price.diff()
    up = []
    down = []
    for i in range(len(ret)):
        if ret[i] < 0 :
            up.append(0)
            down.append(ret[i])
        else:
            up.append(ret[i])
            down.append(0)
    up_series = pd.Series(up)
    down_series = pd.Series(down).abs()    
    up_ewm = up_series.ewm(com = time -1,adjust = False ).mean()
    down_ewm = down_series.ewm(com = time -1,adjust = False).mean()
    
    rs = up_ewm/down_ewm
    rsi = 100 - (100/(1+rs))
    #rsi_df = pd.DataFrame(rsi).rename(columns = {0:'rsi'}).set_index(price.index)
    stock[stock.name +'_RSI'] = rsi
    return stock


# Commodity Channel Index 
def calculating_cci(stock,time=14):
    typical_price = ( stock['High'] + stock['Low'] + stock['Price'] ) / 3
    sma = typical_price.rolling(time).mean()  #simple moving average
    mad = typical_price.rolling(time).apply(lambda x: pd.Series(x).mad()) #mean absolute deviation
    cci = (typical_price-sma) / (0.015*mad)
    
    stock[stock.name+'_CCI'] = cci
    return stock 

'''
https://medium.com/codex/algorithmic-trading-with-average-directional-index-in-python-2b5a20ecf06a 
for theoretical explanation 
'''

# Average directional Index 
# Time frame 28 days 
def calculating_adx(stock, time=28):
    high = stock['High']
    low = stock['Low']
    price = stock['Price']
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - price.shift(1)))
    tr3 = pd.DataFrame(abs(low - price.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.rolling(time).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha = 1/time).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha = 1/time).mean() / atr)) 
    
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (time - 1)) + dx) / time
    adx_smooth = adx.ewm(alpha = 1/time).mean()
    
    #stock['adx_plus_di'] = plus_di
    #stock['adx_minus_di'] = minus_di
    stock[stock.name+'_ADX'] = adx_smooth
    return stock

###################### Processing the datasets ######################################


def setting_companies(stock):
    name = stock.name 
    stock = stock.iloc[::-1]
    stock.reset_index(inplace=True,drop=True)
    stock["Date"] = pd.to_datetime(stock["Date"])
    stock.fillna(method='ffill',inplace=True)
    stock.reset_index(drop=True, inplace=True)
    stock.name = name
    
    # Now calculating indices
    calculating_macd(stock)
    calculating_rsi(stock)
    calculating_cci(stock)
    calculating_adx(stock)
    
    stock.drop(["Open", "High", "Low", "Change %"], axis=1, inplace=True)
    
    stock[name+"_RSI"].fillna(method='backfill', inplace=True)
    stock[name+"_CCI"].fillna(method='backfill', inplace=True)
    stock[name+"_ADX"].fillna(method='backfill', inplace=True)
    
    # Renaming the columns
    stock.rename(columns={"Price":name+"_Price","Open":name+"_Open","High":name+"_High","Low":name+"_Low","Vol.":name+"_Vol"},inplace=True)
    
    # Taking care of volume
    stock[name+'_Vol'] = stock[name+'_Vol'].astype('str')
    values = {'K':1e3,'M':1e6}
    stock[name+'_Vol'] = stock[name+'_Vol'].apply(lambda x:float(x[:-1])*values[x[-1:]])
    
    return stock

bx = setting_companies(bx)
gp = setting_companies(gp)
sq = setting_companies(sq)

# Merging the 3 stocks 
stock = pd.merge(pd.merge(bx,gp,on='Date'),sq,on='Date')
stock.to_csv("Final Stock Data.csv",index=False)


# No need for modifications as of 25th September, 2022 