"""
model.py — XGBoost UAV risk classifier
Trains on 12,000 synthetic UAV mission records.
Features: battery, wind, payload, distance, altitude, temp, visibility, time_of_day
Labels:   LOW / MEDIUM / HIGH
"""

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    XGB_AVAILABLE = False


# ── Synthetic data generation ─────────────────────────────────────────────────

def _label(battery, wind, payload, distance, altitude, temp, visibility, time_of_day):
    """
    Physics-informed labelling function.
    visibility: 0=Clear, 1=Light Haze, 3=Heavy Fog, 4=Rain
    time_of_day: 0=Day, 1=Dusk/Dawn, 2=Night
    """
    score = 0.0

    # Battery: critical below 30%, moderate 30-50%
    if battery < 20:
        score += 3.5
    elif battery < 35:
        score += 2.5
    elif battery < 50:
        score += 1.0

    # Wind: dangerous above 30 km/h
    if wind > 35:
        score += 3.0
    elif wind > 25:
        score += 2.0
    elif wind > 15:
        score += 0.8

    # Payload: heavier = more battery drain + stability risk
    if payload > 3.5:
        score += 2.0
    elif payload > 2.0:
        score += 1.0

    # Distance: range risk
    if distance > 15:
        score += 2.0
    elif distance > 8:
        score += 1.0

    # Altitude: higher = more wind exposure + comms risk
    if altitude > 300:
        score += 1.5
    elif altitude > 200:
        score += 0.5

    # Temperature extremes
    if temp < 0 or temp > 40:
        score += 1.5
    elif temp < 5 or temp > 35:
        score += 0.5

    # Visibility degradation
    score += visibility * 0.8

    # Night flying penalty
    score += time_of_day * 0.7

    # Combined penalties (interaction terms)
    if battery < 40 and distance > 10:
        score += 1.5
    if wind > 20 and payload > 2:
        score += 1.0
    if time_of_day == 2 and visibility >= 1:
        score += 1.0

    if score >= 5.5:
        return "HIGH"
    elif score >= 2.5:
        return "MEDIUM"
    else:
        return "LOW"


def generate_training_data(n=12000, seed=42):
    rng = np.random.default_rng(seed)

    battery    = rng.integers(5, 101, n).astype(float)
    wind       = rng.integers(0, 51, n).astype(float)
    payload    = rng.uniform(0, 5, n)
    distance   = rng.uniform(0, 20, n)
    altitude   = rng.integers(10, 401, n).astype(float)
    temp       = rng.uniform(-10, 45, n)
    visibility = rng.choice([0, 1, 3, 4], n, p=[0.55, 0.25, 0.12, 0.08])
    time_of_day = rng.choice([0, 1, 2], n, p=[0.60, 0.20, 0.20])

    labels = [
        _label(battery[i], wind[i], payload[i], distance[i],
               altitude[i], temp[i], visibility[i], time_of_day[i])
        for i in range(n)
    ]

    X = np.column_stack([battery, wind, payload, distance,
                         altitude, temp, visibility, time_of_day])
    return X, np.array(labels)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(save_path="uav_risk_model.pkl"):
    print("Training UAV risk classifier on 12,000 synthetic missions...")
    X, y = generate_training_data(12000)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.15, random_state=42, stratify=y_enc
    )

    if XGB_AVAILABLE:
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
    else:
        # Fallback to RandomForest if XGBoost not installed
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    acc = (y_pred == y_test).mean()
    print(f"Test Accuracy: {acc * 100:.2f}%")

    joblib.dump((clf, scaler, le), save_path)
    print(f"Model saved to {save_path}")
    return clf, scaler, le


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_risk(model, scaler, label_encoder, features,
                 battery, wind, payload, distance):
    """
    Returns: risk_label, risk_proba, mission_status,
             flight_time, battery_required, battery_remaining
    """
    X_scaled = scaler.transform(features)
    pred_idx  = model.predict(X_scaled)[0]
    risk_proba = model.predict_proba(X_scaled)[0]
    risk_label = label_encoder.inverse_transform([pred_idx])[0]

    # Physics-based estimates
    battery_factor  = battery / 100
    payload_penalty = payload * 2
    wind_penalty    = wind * 0.3
    dist_penalty    = distance * 1.5
    flight_time     = max(0, (30 * battery_factor) - payload_penalty - wind_penalty - dist_penalty)

    battery_required  = (distance * 8) + (payload * 5) + (wind * 0.7)
    battery_remaining = battery - battery_required

    # Mission status logic
    if (flight_time <= 5 or battery_remaining < 10 or risk_label == "HIGH"):
        mission_status = "🔴 ABORT"
    elif (flight_time <= 12 or battery_remaining < 25 or risk_label == "MEDIUM"):
        mission_status = "🟡 CAUTION"
    else:
        mission_status = "🟢 APPROVED"

    return risk_label, risk_proba, mission_status, flight_time, battery_required, battery_remaining


if __name__ == "__main__":
    train_model()