# app.py  —  DeliverIQ · Urban Delivery Route Optimiser
# Streamlit inference interface
# Flow: Input form → predict() → display

import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 0.  Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeliverIQ — Travel Time Predictor",
    page_icon="🚚",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────
# 1.  Load artifacts  (cached — runs once at startup)
# ─────────────────────────────────────────────────────────────
MODEL_PATH      = Path("models/best_model.pkl")
METRICS_PATH    = Path("models/metrics.json")
FEAT_IMP_PATH   = Path("models/feature_importance.csv")


@st.cache_resource(show_spinner="Loading model…")
def load_model(path: Path):
    """Load the fitted sklearn Pipeline from disk."""
    if not path.exists():
        st.error(
            f"❌ `{path}` not found. Run NB-04 to completion first.",
            icon="🚨",
        )
        st.stop()
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    from sklearn.pipeline import Pipeline
    assert isinstance(pipeline, Pipeline), "Loaded object is not an sklearn Pipeline."
    return pipeline


@st.cache_data(show_spinner=False)
def load_metrics(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_feature_importance(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


model      = load_model(MODEL_PATH)
metrics    = load_metrics(METRICS_PATH)
feat_imp   = load_feature_importance(FEAT_IMP_PATH)

USE_LOG_TRANSFORM: bool = metrics.get("log_transform", True)


# ─────────────────────────────────────────────────────────────
# 2.  Helper — build_input_df  (mirrors src/predict.py)
# ─────────────────────────────────────────────────────────────
# Lookup tables used to derive hidden numeric features from
# the user's categorical selections (per src_module_reference)
_AVG_SPEED = {"low": 20.0, "medium": 35.0, "high": 60.0}
_VISIBILITY = {"clear": 10.0, "rain": 4.0, "fog": 1.5}


def build_input_df(user_inputs: dict) -> pd.DataFrame:
    """
    Converts the Streamlit form dict to a single-row DataFrame
    with column names exactly matching the training feature schema.

    Derived columns:
        avg_speed_kmph  ← traffic_level lookup
        visibility_km   ← weather_type lookup
        is_rush_hour    ← hour in {7, 8, 17, 18}
        is_weekend      ← day_of_week >= 5
    """
    hour        = user_inputs["hour"]
    dow         = user_inputs["day_of_week"]
    traffic_lvl = user_inputs["traffic_level"]
    weather_t   = user_inputs["weather_type"]

    row = {
        # — numeric features —
        "distance_km":    user_inputs["distance_km"],
        "num_lanes":      user_inputs["num_lanes"],
        "num_signals":    user_inputs["num_signals"],
        "avg_speed_kmph": _AVG_SPEED[traffic_lvl],
        "temperature_c":  user_inputs.get("temperature_c", 28.0),
        "visibility_km":  _VISIBILITY[weather_t],
        "hour":           hour,
        "day_of_week":    dow,
        # — boolean features —
        "is_rush_hour":   int(hour in (7, 8, 17, 18)),
        "is_weekend":     int(dow >= 5),
        # — ordinal feature —
        "traffic_level":  traffic_lvl,
        # — categorical features —
        "road_type":      user_inputs["road_type"],
        "weather_type":   weather_t,
    }
    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────
# 3.  Helper — predict  (mirrors src/predict.py)
# ─────────────────────────────────────────────────────────────
def predict(pipeline, input_df: pd.DataFrame, use_log_transform: bool = False) -> float:
    """
    Run inference.  If use_log_transform=True, the model was trained
    on log1p(travel_time_min); apply np.expm1() to back-transform.
    Clamps result to 0.0 to guard against negative predictions.
    """
    raw = pipeline.predict(input_df)[0]
    result = float(np.expm1(raw)) if use_log_transform else float(raw)
    return max(0.0, result)


# ─────────────────────────────────────────────────────────────
# 4.  Sidebar — model card & metrics
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/delivery-truck.png", width=64)
    st.title("DeliverIQ")
    st.caption("Urban Delivery Route Optimiser — ML Sprint 5")
    st.divider()

    st.subheader("📊 Model Card")
    st.markdown(f"**Model:** `{metrics.get('model_name', 'RandomForestRegressor')}`")
    st.markdown(f"**Train rows:** {metrics.get('training_rows', 0):,}")
    st.markdown(f"**Test rows:** {metrics.get('test_rows', 0):,}")
    st.markdown(f"**Log-transform:** `{'Yes' if USE_LOG_TRANSFORM else 'No'}`")

    st.divider()
    st.subheader("📈 Test-Set Metrics")

    col_m1, col_m2 = st.columns(2)
    col_m1.metric("MAE",  f"{metrics.get('MAE',  0):.2f} min")
    col_m2.metric("RMSE", f"{metrics.get('RMSE', 0):.2f} min")
    col_m1.metric("R²",   f"{metrics.get('R2',   0):.4f}")
    col_m2.metric("MAPE", f"{metrics.get('MAPE', 0):.2f}%")

    st.divider()
    st.subheader("🏆 Baseline Comparison")
    baseline_mae   = metrics.get("baseline_MAE", 0)
    improvement    = metrics.get("improvement_pct", 0)
    st.metric(
        "Baseline MAE",
        f"{baseline_mae:.2f} min",
        delta=f"−{improvement:.1f}% vs baseline",
        delta_color="inverse",
    )

    st.divider()
    st.subheader("⚙️ Best Params")
    best_params = metrics.get("best_params", {})
    if best_params:
        params_df = pd.DataFrame(
            best_params.items(), columns=["Parameter", "Value"]
        )
        st.dataframe(params_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No hyperparameter info saved.")


# ─────────────────────────────────────────────────────────────
# 5.  Main — title
# ─────────────────────────────────────────────────────────────
st.title("🚚 DeliverIQ — Travel Time Predictor")
st.markdown(
    "Fill in the route details below and click **Predict Travel Time** "
    "to get an estimated delivery duration in minutes."
)
st.divider()


# ─────────────────────────────────────────────────────────────
# 6.  Input form
# ─────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("🗺️ Route Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        distance_km = st.slider(
            "Distance (km)",
            min_value=1.0, max_value=100.0, value=15.0, step=0.5,
            help="Origin-to-destination straight-line distance.",
        )
        num_lanes = st.slider(
            "Number of Lanes",
            min_value=1, max_value=8, value=2,
            help="Lane count on the primary road segment.",
        )
        num_signals = st.slider(
            "Number of Signals",
            min_value=0, max_value=30, value=5,
            help="Traffic signals along the route.",
        )

    with col2:
        road_type = st.selectbox(
            "Road Type",
            options=["highway", "urban", "residential"],
            index=1,
            help="Dominant road type on the route.",
        )
        traffic_level = st.selectbox(
            "Traffic Level",
            options=["low", "medium", "high"],
            index=1,
            help=(
                "Observed congestion level.\n\n"
                "Avg speed derived: low=60 km/h · medium=35 km/h · high=20 km/h"
            ),
        )
        weather_type = st.selectbox(
            "Weather Condition",
            options=["clear", "rain", "fog"],
            index=0,
            help=(
                "Current weather along the route.\n\n"
                "Visibility derived: clear=10 km · rain=4 km · fog=1.5 km"
            ),
        )

    with col3:
        hour = st.slider(
            "Departure Hour (24h)",
            min_value=0, max_value=23, value=9,
            help="Hour of departure (0 = midnight, 17 = 5 PM).",
        )
        day_of_week = st.selectbox(
            "Day of Week",
            options=["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"],
            index=0,
            help="Mon–Fri = weekday (0–4) · Sat–Sun = weekend (5–6).",
        )
        temperature_c = st.slider(
            "Temperature (°C)",
            min_value=-5.0, max_value=50.0, value=28.0, step=0.5,
            help="Ambient temperature. Default 28°C (typical Delhi).",
        )

    st.divider()

    # Derived info badges (read-only, shown before submit)
    dow_num = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(day_of_week)
    is_rush  = hour in (7, 8, 17, 18)
    is_wkend = dow_num >= 5
    avg_spd  = _AVG_SPEED[traffic_level]
    vis_km   = _VISIBILITY[weather_type]

    badge_cols = st.columns(4)
    badge_cols[0].info(f"🕐 {'Rush hour' if is_rush else 'Off-peak'}")
    badge_cols[1].info(f"📅 {'Weekend' if is_wkend else 'Weekday'}")
    badge_cols[2].info(f"🚗 Avg speed: **{avg_spd:.0f} km/h**")
    badge_cols[3].info(f"👁️ Visibility: **{vis_km:.1f} km**")

    submitted = st.form_submit_button(
        "🔮 Predict Travel Time", use_container_width=True, type="primary"
    )


# ─────────────────────────────────────────────────────────────
# 7.  Prediction & display
# ─────────────────────────────────────────────────────────────
if submitted:
    user_inputs = {
        "distance_km":   distance_km,
        "num_lanes":     num_lanes,
        "num_signals":   num_signals,
        "hour":          hour,
        "day_of_week":   dow_num,
        "traffic_level": traffic_level,
        "road_type":     road_type,
        "weather_type":  weather_type,
        "temperature_c": temperature_c,
    }

    input_df  = build_input_df(user_inputs)
    pred_time = predict(model, input_df, use_log_transform=USE_LOG_TRANSFORM)

    st.divider()
    st.subheader("🎯 Prediction Result")

    res_col1, res_col2, res_col3 = st.columns([2, 1, 1])

    with res_col1:
        st.success(
            f"### ⏱️ Estimated Travel Time: **{pred_time:.1f} minutes**",
            icon="✅",
        )
        hrs, mins = divmod(int(round(pred_time)), 60)
        if hrs > 0:
            st.caption(f"≈ {hrs} hr {mins} min")

    with res_col2:
        st.metric("MAE uncertainty (±)", f"{metrics.get('MAE', 0):.2f} min")

    with res_col3:
        st.metric("Avg speed assumed", f"{avg_spd:.0f} km/h")

    # — Input echo table —
    with st.expander("📋 Input Summary", expanded=False):
        echo_df = input_df.T.rename(columns={0: "Value"})
        echo_df.index.name = "Feature"
        st.dataframe(echo_df, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────────────────
    # 8.  Feature importance chart  (Sprint 5.33)
    # ─────────────────────────────────────────────────────────
    with st.expander("📊 Feature Importance (Top 15)", expanded=True):
        top15 = feat_imp.head(15).copy()
        top15["importance_pct"] = top15["importance"] * 100

        fig = px.bar(
            top15.sort_values("importance_pct"),
            x="importance_pct",
            y="feature",
            orientation="h",
            labels={"importance_pct": "Importance (%)", "feature": "Feature"},
            title="Random Forest — Feature Importances (Top 15)",
            color="importance_pct",
            color_continuous_scale="Blues",
            text=top15.sort_values("importance_pct")["importance_pct"].apply(
                lambda v: f"{v:.1f}%"
            ),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            coloraxis_showscale=False,
            yaxis_title="",
            xaxis_title="Mean Decrease in Impurity (%)",
            height=520,
            margin=dict(l=10, r=40, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "`distance_km` dominates at ~65% — consistent with how `travel_time_min` "
            "was generated. `num_signals` and `avg_speed_kmph` are next most informative."
        )