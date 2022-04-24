#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 10:54:45 2022

@author: mitesh.gupta
"""

import plotly.express as px
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def get_daily_sum_demand(df):
    # daily demand summing up all the hours in the day
    fig = px.line(df.groupby(['date'])['demand'].sum().reset_index(), x="date", y="demand")
    fig.show()

def get_daily_mean_demand(df):
    # mean daily demand 
    fig = px.line(df.groupby(['date'])['demand'].mean().reset_index(), x="date", y="demand")
    fig.show()


def get_hourly_demand(df,type = "mean"):
    
    if type == "mean":
        fig = px.line(df.groupby(['hour'])['demand'].mean().reset_index(), x="hour", y="demand")
    
    if type == "max":
        fig = px.line(df.groupby(['hour'])['demand'].max().reset_index(), x="hour", y="demand")
    
    if type == "min":
        fig = px.line(df.groupby(['hour'])['demand'].min().reset_index(), x="hour", y="demand")
    
    fig.show()


def perform_dft(df):
    df = df.copy()
    print("Observations of Dickey-fuller test")
    df = adfuller(df['demand'],autolag='AIC')
    dfoutput=pd.Series(df[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
    for key,value in df[4].items():
        dfoutput['critical value (%s)'%key]= value
    print(dfoutput)
    
