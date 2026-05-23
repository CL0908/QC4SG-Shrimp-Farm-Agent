import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
"""

st.title("Shrimp Pond AI Copilot")
st.write("This dashboard shows pond risk levels and gives simple recommendations.")

# Load your CSV data
df = pd.read_csv("model_input.csv")

# -------------------------
# Screen 1: Pond Risk Table
# -------------------------
st.subheader("Screen 1: Pond Risk Table")

st.dataframe(
    df[["pond_id", "date", "risk_score", "hotspot", "top_driver", "action_flag"]],
    width="stretch"
)

# -------------------------
# Add this below your table
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
    st.metric("Risk Score", pond_data["risk_score"])

with col3:
    st.metric("Hotspot", pond_data["hotspot"])

st.write("Date:", pond_data["date"])
st.write("Top driver:", pond_data["top_driver"])
st.write("Action flag:", pond_data["action_flag"])

# -------------------------
# AI Copilot Explanation
# -------------------------
def get_copilot_explanation(pond_data):
    pond_dict = pond_data.to_dict()

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=SYSTEM_PROMPT,
        input=f"Here is the pond risk data: {pond_dict}"
    )

    return response.output_text

st.subheader("AI Copilot Explanation")

if st.button("Explain this pond"):
    with st.spinner("Copilot is analysing this pond..."):
        explanation = get_copilot_explanation(pond_data)
        st.info(explanation)

# -------------------------
# Screen 3: Replan Alert
# -------------------------
st.subheader("Screen 3: Replan Alert")

if st.button("Replan & Refresh Data"):
    st.success("New data received. Risk table refreshed. Please review updated pond priorities.")