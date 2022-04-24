#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 10:55:27 2022

@author: mitesh.gupta
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import feat_engg
import model


## reading data
train_df = pd.read_csv("train_E1GspfA.csv")
test_df = pd.read_csv("test_6QvDdzb.csv")


train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

# ---------------------- feature engineering -----------------
## feat_engg for fb_prophet model
fb_train_df = train_df.copy()
fb_test_df = test_df.copy()

fb_train_df['date'] = fb_train_df.apply(lambda row: feat_engg.add_hour(row['date'],row['hour']),axis=1)
fb_test_df['date'] = fb_test_df.apply(lambda row: feat_engg.add_hour(row['date'],row['hour']),axis=1)
fb_train_df = fb_train_df[['date','demand']]
fb_train_df.columns = ['ds','y'] 



## applying feature engineering for other models
train_df = feat_engg.feat_engineer(train_df)
test_df = feat_engg.feat_engineer(test_df)



# ---------------------------- Modeling -------------
# 1 Linear Regression Model glmnet_mdoel
glm , glm_demand_pred = model.lr_glmnet(train_df,test_df)

test_df['demand_glmnet'] = glm_demand_pred

# 2 FB Prophet model
prophet_demand_pred = model.train_with_prophet_model(fb_train_df,fb_test_df)
test_df['demand_prophet'] = prophet_demand_pred

# xg boost model
xgm , xg_demand_pred = model.xgb_model(train_df,test_df)
test_df['demand_xgb'] = xg_demand_pred

test_df['mean_demand_glm_prophet'] = (test_df['demand_glmnet'] + test_df['demand_prophet']) / 2
test_df['mean_demand_glm_xgb'] = (test_df['demand_glmnet'] + test_df['demand_xgb']) / 2
test_df['mean_demand_prophet_xgb'] = (test_df['demand_prophet'] + test_df['demand_xgb']) / 2
test_df['mean_demand_prophet_xgb_prophet_glm'] = (test_df['demand_prophet'] + test_df['demand_xgb'] + test_df['demand_glmnet']) / 3

demand_cols = [col for col in test_df.columns if 'demand' in col]

for col in demand_cols:
    sub_df = pd.read_csv("sample_4E0BhPN.csv")
    sub_df['demand'] = test_df[col]
    
    sub_df.to_csv('predictions/sub_df_' + col + '.csv',index = False)



