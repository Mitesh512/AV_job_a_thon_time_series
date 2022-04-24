#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 10:53:50 2022

@author: mitesh.gupta
"""

import pandas as pd
import numpy as np
import datetime


cols_for_model = ['hour', 'month', 'week', 'year', 'day_of_week',
       'day_of_year', 'quarter', 'is_weekend', 'is_weekday', 'days_in_month']

def get_date_feats(df):
    df = df.copy()
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.week
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = np.where(df['day_of_week'].isin([5,6]),1,0)
    df['is_weekday'] = np.where(df['day_of_week'].isin([0,1,2,3,4]),1,0)
    df['days_in_month'] = df['date'].dt.days_in_month
    
    return df

def get_lag_feats(df,num_of_lags=4):
    df = df.copy()
    for lag in range(1,num_of_lags+1):
        df["demand_lag_0" + str(lag)] = df['demand'].shift(lag).fillna(method = 'bfill')
    return df

def feat_engineer(df):
    df = df.copy()
    df = get_date_feats(df)
#     df = get_lag_feats(df,4)
    return df


def add_hour(dt,hr):
    return dt  + datetime.timedelta(hours = hr)



