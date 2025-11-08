#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.callbacks import EarlyStopping


# In[2]:


df = pd.read_stata("/mnt/home/goldberg/rd1150/df_ml.dta")


# In[3]:


# One-hot encode style, dropping the first category to avoid dummy trap
df = pd.get_dummies(df, columns=['style'], drop_first=True)


# In[4]:


#arrange the data
df['caldt'] = pd.to_datetime(df['caldt'])
df = df.sort_values(by=['a1', 'caldt'])
regressors = [col for col in df.columns if col not in ['a1','caldt','real_alpha_net']]
df[regressors] = df.groupby('a1')[regressors].shift(1)
df = df.dropna(subset=regressors)
print(df.head())


# In[5]:


df = df.dropna()
df = df[df["sortino_ratio"] <= 10]
df = df[df["sharpe_ratio"] <= 5.5]
df['a1'].nunique()


# In[6]:


# Normalize features
feature_cols = [col for col in df.columns if col not in ["a1", "caldt", "real_alpha_net"]]
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])


# In[7]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ols_preds = []
performance = []
ols_importance = []

# Define start and end year for prediction
first_year = df['caldt'].dt.year.min()  # e.g., 1993
start_year = first_year + 10            # first prediction year = 2003
end_year = df['caldt'].dt.year.max()    # last year in your data

# ---- Rolling yearly prediction ----
for target_year in range(start_year, end_year + 1):
    # Training: previous 10 years
    train_mask = df['caldt'].dt.year.between(target_year - 10, target_year - 1)
    test_mask = df['caldt'].dt.year == target_year

    train_data = df[train_mask].dropna(subset=feature_cols + ['real_alpha_net'])
    test_data = df[test_mask].dropna(subset=feature_cols + ['real_alpha_net'])

    if train_data.empty or test_data.empty:
        continue

    X_train = train_data[feature_cols]
    y_train = train_data['real_alpha_net']
    X_test = test_data[feature_cols]
    y_test = test_data['real_alpha_net']
    fund_test = test_data['a1']

    # Fit OLS model
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    preds = ols.predict(X_test)

    # Store predictions per month
    preds_df = pd.DataFrame({
        'a1': fund_test,
        'caldt': test_data['caldt'],
        'actual_value': y_test,
        'predicted_value': preds,
        'model': 'OLS'
    })
    ols_preds.append(preds_df)

    # Store yearly performance (aggregate over months of target year)
    performance.append({
        'model': 'OLS',
        'year': target_year,
        'r2': r2_score(y_test, preds),
        'rmse': mean_squared_error(y_test, preds, squared=False),
        'mae': mean_absolute_error(y_test, preds)
    })

    # Store coefficient importance per year
    for i, col in enumerate(feature_cols):
        ols_importance.append({
            'model': 'OLS',
            'predictor': col,
            'importance': ols.coef_[i],
            'year': target_year
        })

# ---- Combine results into DataFrames ----
ols_preds_df = pd.concat(ols_preds, ignore_index=True)
ols_perf_df = pd.DataFrame(performance)
ols_imp_df = pd.DataFrame(ols_importance)

# ---- Export to CSV ----
ols_preds_df.to_csv("/mnt/home/goldberg/rd1150/ols_predictions.csv", index=False)
ols_perf_df.to_csv("/mnt/home/goldberg/rd1150/ols_performance.csv", index=False)
ols_imp_df.to_csv("/mnt/home/goldberg/rd1150/ols_feature_importance.csv", index=False)


# In[8]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# In[ ]:


rf_preds = []
performance = []
rf_importance = []

first_year = df['caldt'].dt.year.min()
start_year = first_year + 10
end_year = df['caldt'].dt.year.max()

# ---- Rolling yearly Random Forest predictions ----
for target_year in range(start_year, end_year + 1):
    train_mask = df['caldt'].dt.year.between(target_year - 10, target_year - 1)
    test_mask = df['caldt'].dt.year == target_year

    train_data = df[train_mask].dropna(subset=feature_cols + ['real_alpha_net'])
    test_data = df[test_mask].dropna(subset=feature_cols + ['real_alpha_net'])

    if train_data.empty or test_data.empty:
        continue

    X_train = train_data[feature_cols]
    y_train = train_data['real_alpha_net']
    X_test = test_data[feature_cols]
    y_test = test_data['real_alpha_net']
    fund_test = test_data['a1']

    # ---- Fit Random Forest ----
    rf = RandomForestRegressor(
        n_estimators=500,        # number of trees
        max_depth=None,          # grow trees fully
        random_state=42,
        n_jobs=-1                # use all cores
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    # ---- Store predictions ----
    preds_df = pd.DataFrame({
        'a1': fund_test,
        'caldt': test_data['caldt'],
        'actual_value': y_test,
        'predicted_value': preds,
        'model': 'RF'
    })
    rf_preds.append(preds_df)

    # ---- Store yearly performance ----
    performance.append({
        'model': 'RF',
        'year': target_year,
        'r2': r2_score(y_test, preds),
        'rmse': mean_squared_error(y_test, preds, squared=False),
        'mae': mean_absolute_error(y_test, preds)
    })

    # ---- Store feature importance per year ----
    for i, col in enumerate(feature_cols):
        rf_importance.append({
            'model': 'RF',
            'predictor': col,
            'importance': rf.feature_importances_[i],
            'year': target_year
        })

# ---- Combine results ----
rf_preds_df = pd.concat(rf_preds, ignore_index=True)
rf_perf_df = pd.DataFrame(performance)
rf_imp_df = pd.DataFrame(rf_importance)

# ---- Export to CSV ----
rf_preds_df.to_csv("/mnt/home/goldberg/rd1150/rf_predictions.csv", index=False)
rf_perf_df.to_csv("/mnt/home/goldberg/rd1150/rf_performance.csv", index=False)
rf_imp_df.to_csv("/mnt/home/goldberg/rd1150/rf_feature_importance.csv", index=False)


# In[ ]:




