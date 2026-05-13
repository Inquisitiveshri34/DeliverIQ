


def extract_time_features(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week_num'] = df['timestamp'].dt.dayofweek  # Mon=0
    df['is_weekend'] = df['day_of_week_num'].isin([5, 6]).astype(int)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 17, 18]).astype(int)
    return df

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
import numpy as np

def encode_categoricals(df, fit=False, encoders=None):
    df = df.copy()
    
    if fit:
        encoders = {}
        
        # Ordinal encode traffic_level
        ord_enc = OrdinalEncoder(categories=[['low', 'medium', 'high']])
        df['traffic_level'] = ord_enc.fit_transform(df[['traffic_level']])
        encoders['ordinal'] = ord_enc
        
        # One-hot encode nominal categoricals
        ohe = OneHotEncoder(drop='first', sparse_output=False)
        cat_cols = ['road_type', 'zone_type', 'weather_type']
        ohe_array = ohe.fit_transform(df[cat_cols])
        ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out(cat_cols), index=df.index)
        df = df.drop(columns=cat_cols)
        df = pd.concat([df, ohe_df], axis=1)
        encoders['onehot'] = ohe
        
        return df, encoders
    
    else:
        # Apply pre-fitted encoders
        df['traffic_level'] = encoders['ordinal'].transform(df[['traffic_level']])
        
        cat_cols = ['road_type', 'zone_type', 'weather_type']
        ohe_array = encoders['onehot'].transform(df[cat_cols])
        ohe_df = pd.DataFrame(ohe_array, columns=encoders['onehot'].get_feature_names_out(cat_cols), index=df.index)
        df = df.drop(columns=cat_cols)
        df = pd.concat([df, ohe_df], axis=1)
        
        return df, encoders
    
from sklearn.preprocessing import StandardScaler

def scale_numerics(df, scaler=None, fit=False):
    df = df.copy()
    num_cols = [
        'distance_km', 'num_lanes', 'num_signals',
        'avg_speed_kmph', 'temperature_c', 'visibility_km',
        'hour', 'day_of_week_num'
    ]
    
    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    
    return df, scaler

from sklearn.model_selection import train_test_split

def build_feature_matrix(df):
    df, _ = extract_time_features(df) if 'timestamp' in df.columns else (df, None)
    
    drop_cols = ['trip_id', 'road_id', 'traffic_id', 'weather_id',
                 'timestamp', 'timestamp_w', 'log_travel_time', 'travel_time_min']
    
    y = df['travel_time_min']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f'Train: {X_train.shape}, Test: {X_test.shape}')
    assert len(set(X_train.index) & set(X_test.index)) == 0, 'Index overlap!'
    return X_train, X_test, y_train, y_test


