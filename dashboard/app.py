"""
UAE Sandstorm Prediction System — Streamlit Dashboard
Purple–Pink Geospatial Intelligence Theme
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

from src.model.inference import SandstormPredictor
from src.utils.helpers import load_config, risk_level_from_score

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAE Sandstorm Intelligence System",
    page_icon="🌪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Theme ──────────────────────────────────────────────────────────────
THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'Share Tech Mono', monospace;
        background-color: #0D0010;
        color: #E8D5F5;
    }

    body, .main, .block-container {
    color: #FFFFFF !important;
    }

    .main {
        background-color: #0D0010;
    }

    .block-container {
        padding-top: 1.5rem;
    }

    h1, h2, h3 {
        color: #C8A2C8;
        letter-spacing: 0.08em;
    }

    .stButton > button {
        background: linear-gradient(135deg, #7F00FF, #FF007F);
        color: white;
        border: none;
        border-radius: 4px;
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 0.05em;
        padding: 0.5rem 1.5rem;
        transition: opacity 0.2s;
    }

    .stButton > button:hover {
        opacity: 0.85;
    }

    .stSlider > div > div {
        background: #3D1A5E;
    }

    .metric-card {
        background: linear-gradient(160deg, #1A0028, #2E003E);
        border: 1px solid #7F00FF;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 0 18px rgba(127, 0, 255, 0.25);
    }

    .risk-low {
        color: #C8A2C8;
        font-size: 2rem;
        font-weight: bold;
        text-shadow: 0 0 12px #C8A2C8;
    }

    .risk-medium {
        color: #FF4DCC;
        font-size: 2rem;
        font-weight: bold;
        text-shadow: 0 0 12px #FF4DCC;
    }

    .risk-high {
        color: #FF007F;
        font-size: 2rem;
        font-weight: bold;
        text-shadow: 0 0 18px #FF007F;
        animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    .stSidebar {
        background-color: #110018;
        border-right: 1px solid #3D1A5E;
    }

    .stNumberInput input, .stSelectbox select {
        background-color: #1A0028;
        color: #E8D5F5;
        border: 1px solid #7F00FF;
        border-radius: 4px;
    }

    hr {
        border-color: #3D1A5E;
    }
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#C8A2C8; letter-spacing:0.12em;'>
    ◈ UAE SANDSTORM INTELLIGENCE SYSTEM ◈
</h1>
<p style='text-align:center; color:#8E4585; font-size:0.85rem; letter-spacing:0.1em;'>
    XGBoost · NASA MERRA-2 · NOAA ISD · Real-Time Risk Assessment
</p>
<hr/>
""", unsafe_allow_html=True)


# ── Model loader ───────────────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    model_path = Path("models/sandstorm_xgb.pkl")
    if not model_path.exists():
        return None
    return SandstormPredictor.from_saved(model_path)


predictor = load_predictor()


# ── Sidebar input form ─────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙ Meteorological Input")
st.sidebar.markdown("---")

wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 40.0, 6.0, 0.1)
wind_u = st.sidebar.slider("Wind U10M (m/s)", -20.0, 20.0, 3.0, 0.1)
wind_v = st.sidebar.slider("Wind V10M (m/s)", -20.0, 20.0, 1.5, 0.1)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 35.0, 0.5)
temperature = st.sidebar.slider("Temperature (°C)", 10.0, 55.0, 30.0, 0.5)
pressure = st.sidebar.slider("Surface Pressure (Pa)", 95000.0, 105000.0, 101325.0, 100.0)
aod = st.sidebar.slider("Aerosol Optical Depth", 0.0, 3.0, 0.15, 0.01)
visibility = st.sidebar.slider("Visibility (m)", 0.0, 10000.0, 5000.0, 100.0)

st.sidebar.markdown("---")
st.sidebar.markdown("### Lag Features")
wind_lag_1h = st.sidebar.number_input("Wind Speed Lag 1h", value=5.5)
wind_lag_3h = st.sidebar.number_input("Wind Speed Lag 3h", value=5.0)
wind_lag_6h = st.sidebar.number_input("Wind Speed Lag 6h", value=4.5)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("▶ ANALYZE RISK")


# ── Derived features ───────────────────────────────────────────────────────
def build_feature_dict():
    wind_dir = (np.degrees(np.arctan2(-wind_u, -wind_v)) % 360)
    humidity_pressure_ratio = humidity / (pressure / 1000.0)
    aod_wind = aod * wind_speed

    from datetime import datetime
    now = datetime.utcnow()
    month_sin = np.sin(2 * np.pi * now.month / 12)
    month_cos = np.cos(2 * np.pi * now.month / 12)
    doy = now.timetuple().tm_yday
    doy_sin = np.sin(2 * np.pi * doy / 365)
    doy_cos = np.cos(2 * np.pi * doy / 365)
    hour_sin = np.sin(2 * np.pi * now.hour / 24)
    hour_cos = np.cos(2 * np.pi * now.hour / 24)

    return {
        "wind_speed": wind_speed,
        "wind_direction": wind_dir,
        "wind_u10m": wind_u,
        "wind_v10m": wind_v,
        "temperature_2m": temperature,
        "humidity": humidity,
        "surface_pressure": pressure,
        "aerosol_optical_depth": aod,
        "visibility_m": visibility,
        "wind_speed_lag_1h": wind_lag_1h,
        "wind_speed_lag_3h": wind_lag_3h,
        "wind_speed_lag_6h": wind_lag_6h,
        "humidity_lag_1h": humidity,
        "humidity_lag_3h": humidity,
        "humidity_lag_6h": humidity,
        "aerosol_optical_depth_lag_1h": aod,
        "aerosol_optical_depth_lag_3h": aod,
        "aerosol_optical_depth_lag_6h": aod,
        "wind_speed_roll_3h": (wind_speed + wind_lag_1h + wind_lag_3h) / 3,
        "wind_speed_roll_6h": wind_speed * 0.5 + wind_lag_6h * 0.5,
        "wind_speed_roll_12h": wind_speed,
        "humidity_roll_3h": humidity,
        "humidity_roll_6h": humidity,
        "humidity_roll_12h": humidity,
        "humidity_pressure_ratio": humidity_pressure_ratio,
        "aod_wind_interaction": aod_wind,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
    }


# ── Main panel ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])

if run_btn:
    if predictor is None:
        st.error("Model not loaded. Run `python scripts/train.py` first.")
    else:
        features = build_feature_dict()
        result = predictor.predict(features)

        level_class = {
            "LOW": "risk-low",
            "MEDIUM": "risk-medium",
            "HIGH": "risk-high",
        }[result.risk_level]

        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='color:#8E4585; font-size:0.75rem; letter-spacing:0.1em;'>RISK SCORE</div>
                <div style='color:#7F00FF; font-size:2.5rem; font-weight:bold;'>{result.risk_score:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='color:#8E4585; font-size:0.75rem; letter-spacing:0.1em;'>RISK LEVEL</div>
                <div class='{level_class}'>{result.risk_level}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='color:#8E4585; font-size:0.75rem; letter-spacing:0.1em;'>INPUT WIND SPEED</div>
                <div style='color:#FF4DCC; font-size:2.5rem; font-weight:bold;'>{wind_speed:.1f} m/s</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Gauge chart ──
        st.markdown("---")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result.risk_score,
            number={"font": {"color": "#E8D5F5", "family": "monospace"}, "suffix": ""},
            gauge={
                "axis": {"range": [0, 1], "tickcolor": "#C8A2C8"},
                "bar": {"color": "#7F00FF"},
                "bgcolor": "#1A0028",
                "bordercolor": "#3D1A5E",
                "steps": [
                    {"range": [0, 0.30], "color": "#2E003E"},
                    {"range": [0.30, 0.60], "color": "#4A006E"},
                    {"range": [0.60, 1.0], "color": "#7F00FF"},
                ],
                "threshold": {
                    "line": {"color": "#FF007F", "width": 3},
                    "thickness": 0.8,
                    "value": result.risk_score,
                },
            },
            title={"text": "Sandstorm Risk Score", "font": {"color": "#C8A2C8", "family": "monospace"}},
        ))

        fig_gauge.update_layout(
            paper_bgcolor="#0D0010",
            font={"color": "#E8D5F5", "family": "monospace"},
            height=320,
        )

        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Feature importance chart ──
        st.markdown("### Feature Contribution")
        top_features = sorted(
            result.feature_vector.items(), key=lambda x: abs(x[1]), reverse=True
        )[:12]

        feat_names = [f[0] for f in top_features]
        feat_vals = [f[1] for f in top_features]

        colors = ["#7F00FF" if v >= 0 else "#FF007F" for v in feat_vals]

        fig_feat = go.Figure(go.Bar(
            x=feat_vals,
            y=feat_names,
            orientation="h",
            marker=dict(color=colors),
        ))
        fig_feat.update_layout(
            paper_bgcolor="#0D0010",
            plot_bgcolor="#110018",
            font={"color": "#E8D5F5", "family": "monospace"},
            xaxis=dict(gridcolor="#3D1A5E"),
            yaxis=dict(gridcolor="#3D1A5E"),
            height=400,
            title="Input Feature Values (Top 12 by magnitude)",
            title_font={"color": "#C8A2C8"},
        )
        st.plotly_chart(fig_feat, use_container_width=True)

else:
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div style='color:#8E4585; font-size:0.75rem; letter-spacing:0.1em;'>RISK SCORE</div>
            <div style='color:#3D1A5E; font-size:2.5rem;'>—</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div style='color:#8E4585; font-size:0.75rem; letter-spacing:0.1em;'>RISK LEVEL</div>
            <div style='color:#3D1A5E; font-size:2rem;'>STANDBY</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div style='color:#8E4585; font-size:0.75rem; letter-spacing:0.1em;'>SYSTEM STATUS</div>
            <div style='color:#8E4585; font-size:1.2rem;'>AWAITING INPUT</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <br/>
    <p style='text-align:center; color:#3D1A5E; font-size:0.9rem;'>
        Configure meteorological parameters in the sidebar and click ANALYZE RISK.
    </p>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<hr/>
<p style='text-align:center; color:#3D1A5E; font-size:0.75rem; letter-spacing:0.08em;'>
    UAE SANDSTORM INTELLIGENCE SYSTEM · XGBoost · NASA MERRA-2 · NCMS
</p>
""", unsafe_allow_html=True)