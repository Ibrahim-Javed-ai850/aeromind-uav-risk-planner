import streamlit as st

st.title("🚀 AeroMind")
st.subheader("AI-Powered UAV Mission Planning & Risk Assessment")

battery = st.slider("Battery Level (%)", 0, 100, 100)
wind = st.slider("Wind Speed (km/h)", 0, 50, 0)
payload = st.slider("Payload Weight (kg)", 0.0, 5.0, 0.0)
distance = st.slider("Mission Distance (km)", 0.0, 20.0, 0.0)

risk_score = 0

if battery < 40:
    risk_score += 2

if wind > 25:
    risk_score += 2

if payload > 2:
    risk_score += 1

if distance > 5:
    risk_score += 1

if risk_score >= 4:
    risk_level = "🔴 HIGH"
elif risk_score >= 2:
    risk_level = "🟡 MEDIUM"
else:
    risk_level = "🟢 LOW"

st.header(f"Risk Level: {risk_level}")
