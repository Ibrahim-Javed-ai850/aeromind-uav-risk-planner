# 🚀 AeroMind

## AI-Powered UAV Mission Planning & Risk Assessment Platform

AeroMind is an aerospace-focused ML system that assists drone operators in evaluating mission safety before takeoff. The system combines a trained XGBoost classifier with LLM-generated mission briefings to provide real-time risk assessment, flight time estimation, and intelligent decision support for UAV operations.

---

## Features

### Current Release — v2.0

✅ **XGBoost ML Risk Classifier** — LOW / MEDIUM / HIGH prediction trained on 12,000 synthetic UAV missions (95.9% test accuracy)

✅ **LLM Mission Briefings** — Anthropic Claude API generates professional ops-center style mission reports from structured telemetry

✅ **Physics-Based Flight Estimates** — Flight time, battery required, battery remaining

✅ **Interactive Risk Radar Chart** — Real-time visual breakdown of 6 contributing risk factors

✅ **Model Confidence Visualization** — Per-class probability bar chart from XGBoost output

✅ **Mission Feasibility Analysis** — APPROVED / CAUTION / ABORT status with rationale

### Mission Inputs

- Battery Level (%)
- Wind Speed (km/h)
- Payload Weight (kg)
- Mission Distance (km)
- Flight Altitude (m)
- Ambient Temperature (°C)
- Visibility Conditions (Clear / Light Haze / Heavy Fog / Rain)
- Time of Day (Day / Dusk/Dawn / Night)

### Mission Outputs

- ML Risk Level (LOW / MEDIUM / HIGH) with confidence scores
- Mission Status (APPROVED / CAUTION / ABORT)
- Estimated Flight Time (minutes)
- Estimated Battery Required (%)
- Estimated Battery Remaining (%)
- Risk Factor Radar Chart
- AI-Generated Mission Briefing (Anthropic Claude API)

---

## Tech Stack

**Core ML**
- Python
- XGBoost — gradient boosted classifier
- scikit-learn — preprocessing, train/test split, metrics
- NumPy — synthetic data generation and feature engineering

**AI / LLM**
- Anthropic Claude API (`claude-sonnet-4`) — natural language mission briefings

**Visualization & Frontend**
- Streamlit — interactive web UI
- Plotly — radar chart, confidence bar chart

**Infrastructure**
- joblib — model serialization and caching
- Git / GitHub

---

## ML Architecture

The risk classifier is trained on **12,000 synthetically generated UAV mission records** using a physics-informed labelling function that incorporates:

- Individual parameter thresholds (battery, wind, payload, distance, altitude, temperature, visibility, time of day)
- **Interaction terms** — e.g., low battery × long distance, high wind × heavy payload, night flight × low visibility

The XGBoost model achieves **95.9% test accuracy** across three classes (LOW / MEDIUM / HIGH) on a held-out 15% split.

```
Features:  battery, wind, payload, distance, altitude, temp, visibility, time_of_day
Model:     XGBClassifier (300 estimators, max_depth=6, lr=0.08)
Training:  12,000 samples | Test: 1,800 samples
Accuracy:  95.94%
```

---

## Project Roadmap

### v1.0 — Rule-Based Engine
- UAV risk scoring via if/else logic
- Mission recommendation system

### v1.3 — Enhanced Logic & UI
- Improved risk thresholds
- Flight time and battery estimation
- Mission feasibility status

### v2.0 — ML + LLM Integration ✅ Current
- XGBoost classifier (12k training samples, 95.9% accuracy)
- Physics-informed synthetic data pipeline with interaction terms
- Anthropic Claude API mission briefings
- Plotly radar + confidence visualizations
- 8 input parameters including altitude, temperature, visibility, time of day

### v2.5 — Planned
- Real-time weather API integration (OpenWeatherMap)
- Location-based mission planning

### v3.0 — Planned
- Historical flight database and mission logging
- Route optimization (A* waypoint planner)
- Predictive mission failure detection

---

## Applications

- UAV Mission Planning & Pre-flight Safety Analysis
- Aerospace Operations Decision Support
- Autonomous Systems Research
- AI/ML Applications in Aerospace Engineering

---

## Author

**Ibrahim Javed**  
Toronto Metropolitan University (TMU)  
Bachelor of Engineering — Aerospace Engineering

---
AeroMind v2.0 | XGBoost ML | Anthropic Claude API | Streamlit | Plotly