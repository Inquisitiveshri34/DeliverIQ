import pickle, json, time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

NUM_FEATURES = [
    'distance_km', 'num_lanes', 'num_signals',
    'avg_speed_kmph', 'temperature_c', 'visibility_km',
    'hour', 'day_of_week'
]
BOOL_FEATURES = ['is_rush_hour', 'is_weekend']
ORD_FEATURES  = ['traffic_level']
CAT_FEATURES  = ['road_type', 'weather_type']   # zone_type dropped in cleaning

def baseline_predictor(y_train, y_test):
    y_pred = np.full(len(y_test), y_train.mean())
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'Baseline  MAE={mae:.3f}  RMSE={rmse:.3f}  R²≈0')
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': 0.0}

def build_pipeline(model):
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), NUM_FEATURES),
        ('bool', 'passthrough', BOOL_FEATURES),
        ('ord', OrdinalEncoder(categories=[['low','medium','high']]), ORD_FEATURES),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), CAT_FEATURES),
    ], remainder='drop')
    return Pipeline([('preprocessor', preprocessor), ('model', model)])

def train_model(pipeline, X_train, y_train):
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    print(f'Training time: {time.time()-t0:.1f}s')
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    mask = y_test > 0
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    print(f'MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}  MAPE={mape:.2f}%')
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

def tune_model(pipeline, param_grid, X_train, y_train, method='grid', n_iter=50):
    common = dict(cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
    if method == 'grid':
        search = GridSearchCV(pipeline, param_grid, **common)
    else:
        search = RandomizedSearchCV(pipeline, param_grid,
                                    n_iter=n_iter, random_state=42, **common)
    search.fit(X_train, y_train)
    print(f'Best params: {search.best_params_}')
    print(f'Best CV MAE: {-search.best_score_:.3f} min')
    return search.best_estimator_

def save_model(pipeline, metrics, feature_names, importances, out_dir='models'):
    import os; os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/best_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    with open(f'{out_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({'feature': feature_names, 'importance': importances})\
      .sort_values('importance', ascending=False)\
      .to_csv(f'{out_dir}/feature_importance.csv', index=False)
    print('Saved: best_model.pkl, metrics.json, feature_importance.csv')