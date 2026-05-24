import os
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

# -------------------------
# Load API key from .env
# -------------------------
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="Shrimp Pond AI Copilot", layout="wide")

SYSTEM_PROMPT = """
You are an advisor for a Vietnamese shrimp farming cooperative.

Your job is to explain pond disease-risk predictions in simple language for farm operators.

Given a pond's risk data, produce:
1. A short explanation of the current risk level.
2. The main reason for the risk.
3. A practical recommended action.

Rules:
- Use simple language.
- Be specific and actionable.
- Do not exaggerate certainty.
- Do not claim the shrimp are definitely diseased.
- Keep the response to 3 sentences.
- Write the response in Vietnamese.
"""

# -------------------------
# Live Weather API Function
# -------------------------
@st.cache_data(ttl=1800)
def fetch_live_weather(lat, lon):
    """
    Fetch live rainfall and temperature from Open-Meteo.
    Refreshes every 30 minutes.
    """

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,precipitation,rain"
        "&timezone=auto"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})

        return {
            "temperature": current.get("temperature_2m"),
            "rainfall": current.get("rain", current.get("precipitation", 0)),
            "time": current.get("time")
        }

    except Exception as e:
        st.warning(f"Could not fetch live weather data: {e}")

        return {
            "temperature": None,
            "rainfall": None,
            "time": None
        }


# -------------------------
# App Title
# -------------------------
st.title("Shrimp Pond AI Copilot")
st.write(
    "This dashboard shows pond risk levels, live rainfall and temperature data, "
    "then gives simple recommendations for shrimp pond operators."
)

# -------------------------
# Load CSV Data
# -------------------------
df = pd.read_csv("model_input.csv")

# -------------------------
# Fetch Live Weather Data
# -------------------------
# Example location: Can Tho, Vietnam
# You can change this later to the actual shrimp farm location.
lat = 10.0452
lon = 105.7469

live_weather = fetch_live_weather(lat, lon)

live_temperature = live_weather["temperature"]
live_rainfall = live_weather["rainfall"]
live_time = live_weather["time"]

# -------------------------
# Live Environmental Data
# -------------------------
st.subheader("Live Environmental Data")

col1, col2 = st.columns(2)

with col1:
    if live_rainfall is not None:
        st.metric("Rainfall", f"{live_rainfall} mm")
    else:
        st.metric("Rainfall", "Unavailable")

with col2:
    if live_temperature is not None:
        st.metric("Temperature", f"{live_temperature} °C")
    else:
        st.metric("Temperature", "Unavailable")

if live_time:
    st.caption(f"Weather data last updated: {live_time}")

# -------------------------
# Live Weather Risk Logic
# -------------------------
live_risk_score = 0
live_risk_reasons = []

if live_rainfall is not None and live_rainfall > 10:
    live_risk_score += 2
    live_risk_reasons.append("Heavy rainfall may suddenly change pond water conditions.")

if live_temperature is not None and live_temperature > 32:
    live_risk_score += 1
    live_risk_reasons.append("High temperature may stress shrimp.")

if live_risk_score >= 3:
    live_risk_level = "High"
elif live_risk_score >= 1:
    live_risk_level = "Medium"
else:
    live_risk_level = "Low"

st.subheader("Live Weather-Based Risk")

col1, col2 = st.columns(2)

with col1:
    st.metric("Live Risk Level", live_risk_level)

with col2:
    st.metric("Live Risk Score", live_risk_score)

if live_risk_reasons:
    st.write("Live weather risk factors:")
    for reason in live_risk_reasons:
        st.write("-", reason)
else:
    st.write("No major rainfall or temperature risk detected.")

# -------------------------
# Screen 1: Pond Risk Table
# -------------------------
st.subheader("Screen 1: Pond Risk Table")

def highlight_risk(row):
    risk = row["risk_score"]

    if risk >= 0.7:
        return ["background-color: #7f1d1d; color: white;"] * len(row)   # dark red
    elif risk >= 0.4:
        return ["background-color: #92400e; color: white;"] * len(row)   # dark orange
    else:
        return ["background-color: #14532d; color: white;"] * len(row)   # dark green

display_df = df[["pond_id", "date", "risk_score", "hotspot", "top_driver", "action_flag"]]

styled_df = display_df.style.apply(highlight_risk, axis=1)

st.dataframe(styled_df, use_container_width=True)

# -------------------------
# Select Pond
# -------------------------
st.subheader("Select a Pond")

selected_row = st.selectbox(
    "Choose a pond row:",
    range(len(df)),
    format_func=lambda i: f"{df.iloc[i]['pond_id']} | {df.iloc[i]['date']} | risk={df.iloc[i]['risk_score']}"
)

pond_data = df.iloc[selected_row]

# -------------------------
# Screen 2: Pond Detail Card
# -------------------------
st.subheader("Screen 2: Pond Detail Card")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Pond ID", pond_data["pond_id"])

with col2:
    st.metric("Dataset Risk Score", pond_data["risk_score"])

with col3:
    st.metric("Hotspot", pond_data["hotspot"])

st.write("Date:", pond_data["date"])
st.write("Top driver:", pond_data["top_driver"])
st.write("Action flag:", pond_data["action_flag"])

# -------------------------
# Gemini AI Copilot Explanation
# -------------------------
def get_copilot_explanation(pond_data):
    pond_dict = pond_data.to_dict()

    prompt = f"""
{SYSTEM_PROMPT}

Here is the selected pond dataset information:
{pond_dict}

Here is the live environmental data:
- Rainfall: {live_rainfall} mm
- Temperature: {live_temperature} °C
- Live weather risk level: {live_risk_level}
- Live weather risk score: {live_risk_score}
- Live weather risk reasons: {live_risk_reasons}

Write the 3-sentence recommendation now.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

st.subheader("AI Copilot Explanation")

if st.button("Explain this pond"):
    with st.spinner("Copilot is analysing this pond..."):
        explanation = get_copilot_explanation(pond_data)
        st.info(explanation)

# -------------------------
# Screen 3: Replan Alert
# -------------------------
st.subheader("Screen 3: Replan Alert")

if st.button("Replan & Refresh"):
    st.cache_data.clear()
    st.success("New live weather data requested. Please wait while the dashboard refreshes.")
    st.rerun()