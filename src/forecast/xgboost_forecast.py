import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    y_true_safe = np.where(y_true == 0, np.finfo(float).eps, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

def run_forecast(df):
    df = df.copy()
    max_lag = 5
    for lag in range(1, max_lag + 1):
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)

    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df.dropna(inplace=True)

    test_size = 30
    tscv = TimeSeriesSplit(n_splits=5)
    X = df.drop('Close', axis=1)
    y = df['Close']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }
    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_scaled, y)
    best_model = grid_search.best_estimator_

    X_train, X_test = X_scaled[:-test_size], X_scaled[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)

    y_test_np = y_test.values
    rmse = np.sqrt(mean_squared_error(y_test_np, preds))
    mae = mean_absolute_error(y_test_np, preds)
    r2 = r2_score(y_test_np, preds)
    mape = mean_absolute_percentage_error(y_test_np, preds)

    last_row_scaled = scaler.transform(X.iloc[[-1]])
    future_pred = best_model.predict(last_row_scaled)[0]

    metrics = {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

    return best_model, scaler, future_pred, preds, y_test, metrics
