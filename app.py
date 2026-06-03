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

if risk_level == "🔴 HIGH":
    recommendation = "Mission not recommended. Reduce risk factors before flight."
elif risk_level == "🟡 MEDIUM":
    recommendation = "Proceed with caution. Review wind speed, payload, battery, and distance before launch."
else:
    recommendation = "Mission approved. Current conditions appear suitable for flight."

st.header(f"Risk Level: {risk_level}")
st.subheader("Mission Recommendation")
st.write(recommendation)

# Version 0.3: Flight Time + Battery Estimate

base_flight_time = 30  # minutes

battery_factor = battery / 100
payload_penalty = payload * 2
wind_penalty = wind * 0.3
distance_penalty = distance * 1.5

estimated_flight_time = (base_flight_time * battery_factor) - payload_penalty - wind_penalty - distance_penalty

if estimated_flight_time < 0:
    estimated_flight_time = 0

battery_required = (distance * 8) + (payload * 5) + (wind * 0.7)

st.subheader("Flight Time Estimate")
st.write(f"Estimated Flight Time: {estimated_flight_time:.1f} minutes")

st.subheader("Battery Estimate")
st.write(f"Estimated Battery Required: {battery_required:.1f}%")
# Mission Feasibility Status

battery_remaining = battery - battery_required

if estimated_flight_time <= 5 or battery_remaining < 10 or risk_level == "🔴 HIGH":
    mission_status = "🔴 ABORT"
    feasibility_message = "Mission is not recommended. Conditions may create unsafe flight risk."
elif estimated_flight_time <= 12 or battery_remaining < 25 or risk_level == "🟡 MEDIUM":
    mission_status = "🟡 CAUTION"
    feasibility_message = "Mission may proceed with caution. Review battery, wind, payload, and distance before launch."
else:
    mission_status = "🟢 APPROVED"
    feasibility_message = "Mission approved. Current conditions appear suitable for flight."

st.subheader("Mission Feasibility Status")

if "APPROVED" in mission_status:
    st.success(mission_status)
elif "CAUTION" in mission_status:
    st.warning(mission_status)
else:
    st.error(mission_status)
st.write(feasibility_message)

st.subheader("Estimated Battery Remaining")
st.write(f"Estimated Battery Remaining After Mission: {battery_remaining:.1f}%")
battery_required = (distance * 8) + (payload * 5) + (wind * 0.7)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Risk Score", risk_score)

with col2:
    st.metric("Flight Time", f"{estimated_flight_time:.1f} min")

with col3:
    st.metric("Battery Required", f"{battery_required:.1f}%")
