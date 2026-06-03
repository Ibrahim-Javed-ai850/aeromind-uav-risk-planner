import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from model import train_model, predict_risk
from briefing import generate_mission_briefing
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AeroMind",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #0a0e1a;
    color: #e0e8ff;
}

h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: #4a7ab5;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #e0e8ff;
}

.status-APPROVED {
    background: linear-gradient(135deg, #0d2b1a, #0d3b22);
    border: 1px solid #1a7a40;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #4ade80;
}

.status-CAUTION {
    background: linear-gradient(135deg, #2b2000, #3b2f00);
    border: 1px solid #a07000;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #facc15;
}

.status-ABORT {
    background: linear-gradient(135deg, #2b0d0d, #3b1010);
    border: 1px solid #7a1a1a;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #f87171;
}

.briefing-box {
    background: linear-gradient(135deg, #0d1627, #111e35);
    border: 1px solid #1e3a5f;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.7;
    color: #a8c4e8;
    white-space: pre-wrap;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #3b82f6;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.4rem;
}

.confidence-bar-bg {
    background: #1a2235;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
    margin-top: 4px;
}

.stSlider > div > div > div > div {
    background: #3b82f6 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load / train model ────────────────────────────────────────────────────────
MODEL_PATH = "uav_risk_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return train_model(MODEL_PATH)

model, scaler, label_encoder = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1rem 0 0.5rem 0;">
    <div style="font-family: 'Space Mono', monospace; font-size: 0.7rem; letter-spacing: 0.3em;
                color: #3b82f6; text-transform: uppercase; margin-bottom: 0.3rem;">
        ◈ AEROMIND v2.0 — ML-POWERED
    </div>
    <h1 style="font-size: 2.8rem; margin: 0; background: linear-gradient(90deg, #e0e8ff, #3b82f6);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.1;">
        UAV Mission Planner
    </h1>
    <div style="color: #4a7ab5; font-size: 0.95rem; margin-top: 0.4rem;">
        XGBoost risk classification · LLM mission briefings · Real-time feasibility analysis
    </div>
</div>
<hr style="border-color: #1e3a5f; margin: 1rem 0 1.5rem 0;">
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.6], gap="large")

with left_col:
    st.markdown('<div class="section-header">Mission Parameters</div>', unsafe_allow_html=True)

    battery  = st.slider("🔋 Battery Level (%)",        0,   100, 85)
    wind     = st.slider("💨 Wind Speed (km/h)",         0,    50,  8)
    payload  = st.slider("📦 Payload Weight (kg)",     0.0,  5.0, 1.0, step=0.1)
    distance = st.slider("📍 Mission Distance (km)",   0.0, 20.0, 4.0, step=0.5)
    altitude = st.slider("✈️  Flight Altitude (m)",    10,  400, 120)
    temp     = st.slider("🌡️  Ambient Temp (°C)",      -10,   45,  22)

    st.markdown('<div class="section-header" style="margin-top:1.2rem;">Environment</div>', unsafe_allow_html=True)

    visibility = st.selectbox("👁️  Visibility", ["Clear", "Light Haze", "Heavy Fog", "Rain"])
    time_of_day = st.selectbox("🕐 Time of Day", ["Day", "Dusk/Dawn", "Night"])

    run_analysis = st.button("▶ RUN MISSION ANALYSIS", use_container_width=True, type="primary")

# ── Analysis ──────────────────────────────────────────────────────────────────
visibility_map  = {"Clear": 0, "Light Haze": 1, "Heavy Fog": 3, "Rain": 4}
time_map        = {"Day": 0, "Dusk/Dawn": 1, "Night": 2}

features = np.array([[
    battery, wind, payload, distance, altitude, temp,
    visibility_map[visibility], time_map[time_of_day]
]])

risk_label, risk_proba, mission_status, flight_time, battery_required, battery_remaining = \
    predict_risk(model, scaler, label_encoder, features,
                 battery, wind, payload, distance)

with right_col:
    # ── Status banner ──────────────────────────────────────────────────────
    status_key = mission_status.replace("🟢 ", "").replace("🟡 ", "").replace("🔴 ", "")
    status_class = {"APPROVED": "APPROVED", "CAUTION": "CAUTION", "ABORT": "ABORT"}.get(status_key, "ABORT")

    st.markdown(f"""
    <div class="status-{status_class}">
        <div style="font-family: 'Space Mono', monospace; font-size: 0.65rem;
                    letter-spacing: 0.2em; opacity: 0.7; margin-bottom: 0.2rem;">MISSION STATUS</div>
        <div style="font-size: 1.5rem; font-weight: 700;">{mission_status}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics row ────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    def metric_card(col, label, value):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>""", unsafe_allow_html=True)

    metric_card(m1, "Risk Level", risk_label)
    metric_card(m2, "Flight Time", f"{flight_time:.0f}m")
    metric_card(m3, "Batt. Req.", f"{battery_required:.0f}%")
    metric_card(m4, "Batt. Left", f"{max(battery_remaining, 0):.0f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confidence chart ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Confidence</div>', unsafe_allow_html=True)

    classes = label_encoder.classes_
    colors  = {"LOW": "#4ade80", "MEDIUM": "#facc15", "HIGH": "#f87171"}

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=classes,
        y=risk_proba * 100,
        marker_color=[colors.get(c, "#3b82f6") for c in classes],
        marker_line_width=0,
        text=[f"{p*100:.1f}%" for p in risk_proba],
        textposition="outside",
        textfont=dict(family="Space Mono", color="#a8c4e8", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono", color="#4a7ab5", size=10),
        margin=dict(l=0, r=0, t=10, b=0),
        height=180,
        xaxis=dict(showgrid=False, tickfont=dict(color="#4a7ab5")),
        yaxis=dict(showgrid=True, gridcolor="#1e3a5f", range=[0, 110],
                   ticksuffix="%", tickfont=dict(color="#4a7ab5")),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Risk factor radar ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Risk Factor Breakdown</div>', unsafe_allow_html=True)

    categories = ["Battery", "Wind", "Payload", "Distance", "Altitude", "Visibility"]
    raw_vals   = [
        max(0, (100 - battery) / 100),
        wind / 50,
        payload / 5,
        distance / 20,
        altitude / 400,
        visibility_map[visibility] / 4,
    ]
    norm_vals  = [v * 100 for v in raw_vals]

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=norm_vals + [norm_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(59, 130, 246, 0.15)",
        line=dict(color="#3b82f6", width=2),
        name="Risk Factors",
    ))
    radar_fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1e3a5f",
                            tickfont=dict(color="#4a7ab5", size=8), showticklabels=False),
            angularaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#a8c4e8", size=10)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=20),
        height=240,
        showlegend=False,
    )
    st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})

# ── AI Briefing ────────────────────────────────────────────────────────────────
st.markdown('<hr style="border-color: #1e3a5f; margin: 1rem 0;">', unsafe_allow_html=True)
st.markdown('<div class="section-header">◈ AI Mission Briefing</div>', unsafe_allow_html=True)

if run_analysis or "briefing" not in st.session_state:
    params = {
        "battery": battery, "wind": wind, "payload": payload,
        "distance": distance, "altitude": altitude, "temp": temp,
        "visibility": visibility, "time_of_day": time_of_day,
        "risk_level": risk_label, "mission_status": mission_status,
        "flight_time": flight_time, "battery_required": battery_required,
        "battery_remaining": battery_remaining,
        "model_confidence": float(max(risk_proba)) * 100,
    }
    with st.spinner("Generating mission briefing..."):
        briefing = generate_mission_briefing(params)
    st.session_state["briefing"] = briefing
else:
    briefing = st.session_state.get("briefing", "Click 'Run Mission Analysis' to generate a briefing.")

st.markdown(f'<div class="briefing-box">{briefing}</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top: 2rem; font-family: 'Space Mono', monospace;
            font-size: 0.65rem; color: #2a4a6b; letter-spacing: 0.1em;">
    AEROMIND v2.0 · XGBoost Classifier · Anthropic Claude API · Built with Streamlit
</div>
""", unsafe_allow_html=True)