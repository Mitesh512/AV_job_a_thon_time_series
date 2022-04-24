#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 10:49:50 2022

@author: mitesh.gupta
"""
# Model
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from glmnet import ElasticNet as glm_elastic
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

import xgboost as xgb
from fbprophet import Prophet

import math
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.callbacks import EarlyStopping
from keras.optimizers import adam_v2


import feat_engg



def lr_glmnet(train_df,test_df):
    
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    X_train = train_df[feat_engg.cols_for_model]
    test_df = test_df[feat_engg.cols_for_model]
    y_train = train_df["demand"]
    
    # Standard Scaling
    st = StandardScaler()
    st.fit(X_train)
    X_train = st.transform(X_train)
    test_df = st.transform(test_df)


    # apply model
    glm = glm_elastic(alpha = 0.7)
    glm.fit(X_train, y_train)
    
    test_df = pd.DataFrame(test_df,columns = feat_engg.cols_for_model )
    demand_pred = glm.predict(test_df)
    
    return glm, demand_pred
    
    
    



def train_with_prophet_model(fb_train_df,fb_test_df):
    fb_train_df = fb_train_df.copy()
    fb_test_df = fb_test_df.copy()
    
    # model
    prophet_model = Prophet()
    prophet_model.fit(fb_train_df)
    
   # make prediction
    y_pred_full = prophet_model.predict(prophet_model.make_future_dataframe(
    periods=11000,freq = 'H'))[['ds','yhat_lower', 'yhat','yhat_upper']]
    
    
    # choose prediction for test time period
    demand_pred_with_final_model = y_pred_full[y_pred_full['ds'].isin(fb_test_df['date'].values)]['yhat']
    
    return demand_pred_with_final_model.values


def xgb_model(train_df,test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    X_train = train_df[feat_engg.cols_for_model]
    test_df = test_df[feat_engg.cols_for_model]
    y_train = train_df["demand"]
    
    # Standard Scaling
    st = StandardScaler()
    st.fit(X_train)
    X_train = st.transform(X_train)
    test_df = st.transform(test_df)

    xgm = xgb.XGBRegressor(n_estimators=50,subsample=0.8,
                           colsample_bytree= 0.5,
                        gamma = 0.0,learning_rate = 0.1,
                        max_depth = 3,
                        min_child_weight = 1)
    xgm.fit(X_train, y_train)
    

    test_df = pd.DataFrame(test_df,columns = feat_engg.cols_for_model )
    demand_pred = xgm.predict(test_df)
        
    return xgm,  demand_pred
    
    
    
    

    