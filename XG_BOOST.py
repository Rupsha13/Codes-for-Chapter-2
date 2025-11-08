
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


df=pd.read_stata("/mnt/home/goldberg/rd1150/df_ml.dta")
df = pd.get_dummies(df, columns=['style'], drop_first=True)
df['caldt'] = pd.to_datetime(df['caldt'])
df = df.sort_values(by=['a1', 'caldt'])
regressors = [col for col in df.columns if col not in ['a1','caldt','real_alpha_net']]
df[regressors] = df.groupby('a1')[regressors].shift(1)
df = df.dropna(subset=regressors)
df = df.dropna()
df = df[df["sortino_ratio"] <= 10]
df = df[df["sharpe_ratio"] <= 5.5]
df['a1'].nunique()
feature_cols = [col for col in df.columns if col not in ["a1", "caldt", "real_alpha_net"]]
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])


xgb_preds = []
performance = []
xgb_importance = []

first_year = df['caldt'].dt.year.min()
start_year = first_year + 10
end_year = df['caldt'].dt.year.max()

# ---- Rolling yearly XGBoost predictions ----
for target_year in range(start_year, end_year + 1):
    # define rolling 10-year training window
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

    # ---- Fit XGBoost ----
    xgb = XGBRegressor(
        n_estimators=500,         # number of boosting rounds
        learning_rate=0.05,       # smaller = more conservative
        max_depth=6,              # tree depth
        subsample=0.8,            # sample fraction per tree
        colsample_bytree=0.8,     # feature fraction per tree
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror'
    )

    xgb = XGBRegressor(eval_metric='rmse')
    xgb.fit(X_train, y_train, verbose=False)
    preds = xgb.predict(X_test)

    # ---- Store predictions ----
    preds_df = pd.DataFrame({
        'a1': fund_test,
        'caldt': test_data['caldt'],
        'actual_value': y_test,
        'predicted_value': preds,
        'model': 'XGB'
    })
    xgb_preds.append(preds_df)

    # ---- Store yearly performance ----
    performance.append({
        'model': 'XGB',
        'year': target_year,
        'r2': r2_score(y_test, preds),
        'rmse': mean_squared_error(y_test, preds, squared=False),
        'mae': mean_absolute_error(y_test, preds)
    })

    # ---- Store feature importance per year ----
    importance_dict = xgb.get_booster().get_score(importance_type='gain')
    for col in feature_cols:
        xgb_importance.append({
            'model': 'XGB',
            'predictor': col,
            'importance': importance_dict.get(col, 0),
            'year': target_year
        })

# ---- Combine results ----
xgb_preds_df = pd.concat(xgb_preds, ignore_index=True)
xgb_perf_df = pd.DataFrame(performance)
xgb_imp_df = pd.DataFrame(xgb_importance)

# ---- Export to CSV ----
xgb_preds_df.to_csv("/mnt/home/goldberg/rd1150/xg_predictions.csv", index=False)
xgb_perf_df.to_csv("/mnt/home/goldberg/rd1150/xg_performance.csv", index=False)
xgb_imp_df.to_csv("/mnt/home/goldberg/rd1150/xg_feature_importance.csv", index=False)

