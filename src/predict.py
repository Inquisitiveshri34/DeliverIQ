import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

def load_model(path='models/best_model.pkl'):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    assert isinstance(model, Pipeline), 'Loaded object is not an sklearn Pipeline'
    print(f'Model loaded from {path}')
    return model

def build_input_df(user_inputs: dict) -> pd.DataFrame:
    """
    Converts a flat dict of user inputs into a single-row DataFrame
    that matches the training feature schema exactly.
    """
    hour = user_inputs['hour']
    day  = user_inputs.get('day_of_week', 0)
    
    row = {
        'distance_km':    user_inputs['distance_km'],
        'num_lanes':      user_inputs['num_lanes'],
        'num_signals':    user_inputs['num_signals'],
        'avg_speed_kmph': user_inputs['avg_speed_kmph'],
        'temperature_c':  user_inputs.get('temperature_c', 28.0),
        'visibility_km':  user_inputs['visibility_km'],
        'hour':           hour,
        'day_of_week':    day,
        'is_rush_hour':   int(hour in [7, 8, 17, 18]),
        'is_weekend':     int(day >= 5),
        'traffic_level':  user_inputs['traffic_level'],
        'road_type':      user_inputs['road_type'],
        'weather_type':   user_inputs['weather_type'],
    }
    return pd.DataFrame([row])

def predict(model: Pipeline, input_df: pd.DataFrame,
            use_log_transform: bool = False) -> float:
    pred = model.predict(input_df)[0]
    if use_log_transform:
        pred = np.expm1(pred)
    return float(pred)