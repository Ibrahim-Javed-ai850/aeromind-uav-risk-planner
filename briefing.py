"""
briefing.py — LLM-generated mission briefings via Anthropic Claude API
"""

import anthropic
import os


def generate_mission_briefing(params: dict) -> str:
    """
    Calls the Anthropic API to generate a concise, professional UAV mission briefing.
    Falls back to a rule-based briefing if the API call fails (e.g., no key set).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        return _fallback_briefing(params)

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are an autonomous UAV mission control AI. Generate a concise, professional mission briefing based on the following telemetry and risk assessment data.

MISSION PARAMETERS:
- Battery Level: {params['battery']}%
- Wind Speed: {params['wind']} km/h
- Payload: {params['payload']} kg
- Distance: {params['distance']} km
- Altitude: {params['altitude']} m
- Temperature: {params['temp']}°C
- Visibility: {params['visibility']}
- Time of Day: {params['time_of_day']}

ML RISK ASSESSMENT (XGBoost classifier, 12k training samples):
- Risk Level: {params['risk_level']}
- Mission Status: {params['mission_status']}
- Model Confidence: {params['model_confidence']:.1f}%
- Estimated Flight Time: {params['flight_time']:.1f} min
- Battery Required: {params['battery_required']:.1f}%
- Estimated Battery Remaining: {params['battery_remaining']:.1f}%

Write a 4-6 sentence mission briefing in the style of a professional aviation operations center. 
Be specific about the key risk factors. 
If mission status is ABORT, explain what changes would make the mission viable.
If CAUTION, list the 2 most critical parameters to monitor.
If APPROVED, confirm readiness and note any advisories.
End with a single actionable recommendation.
Do not use bullet points. Write in flowing prose."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return _fallback_briefing(params) + f"\n\n[Note: Live AI briefing unavailable — {str(e)}]"


def _fallback_briefing(params: dict) -> str:
    """Rule-based briefing when API is unavailable."""
    status = params["mission_status"]
    risk   = params["risk_level"]
    batt   = params["battery"]
    wind   = params["wind"]
    dist   = params["distance"]
    ft     = params["flight_time"]
    br     = params["battery_remaining"]

    lines = [f"MISSION BRIEFING — Status: {status} | Risk: {risk}\n"]

    if "ABORT" in status:
        lines.append(
            f"Mission is not recommended for launch. Critical risk factors have been identified: "
            f"battery at {batt}% against a required draw of {params['battery_required']:.0f}%, "
            f"with only {max(br, 0):.0f}% projected remaining post-mission. "
            f"Wind conditions at {wind} km/h may compound handling difficulty under payload. "
            f"Recommendation: Charge battery above 70%, reduce distance below {dist*0.6:.1f} km, "
            f"or reduce payload weight before re-evaluation."
        )
    elif "CAUTION" in status:
        lines.append(
            f"Mission may proceed under monitored conditions. Estimated flight time of {ft:.1f} minutes "
            f"is within operational range, but margins are tighter than optimal. "
            f"Primary concerns: battery trajectory ({batt}% current, {max(br, 0):.0f}% projected remaining) "
            f"and wind speed ({wind} km/h) which may affect stability at {params['altitude']}m altitude. "
            f"Recommendation: Assign a ground observer and maintain direct line-of-sight throughout the mission."
        )
    else:
        lines.append(
            f"Mission is cleared for launch. All parameters fall within safe operational thresholds. "
            f"Battery reserves are adequate at {batt}% with an estimated {max(br, 0):.0f}% remaining post-mission. "
            f"Wind speed of {wind} km/h presents minimal risk at planned altitude of {params['altitude']}m. "
            f"Estimated flight duration: {ft:.1f} minutes. "
            f"Recommendation: Proceed as planned. Conduct pre-flight checklist and confirm GPS lock before takeoff."
        )

    return "\n".join(lines)