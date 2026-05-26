import os
import datetime
import requests

# ── Data & science ────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Streamlit + visualisation ─────────────────────────────
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go

# ── Gemini AI (Google Generative AI) ──────────────────────
import google.generativeai as genai

# ── Load secrets from .env before anything else ───────────
from dotenv import load_dotenv
load_dotenv()

# ── Page config  (MUST be the very first Streamlit call) ──
st.set_page_config(
    page_title="QuteShrimp",
    layout="wide",
    page_icon="🦐",
)

# ── Check the API key is present; stop early if missing ───
_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    st.error("No GEMINI_API_KEY found. Check that .env exists and contains the key.")
    st.stop()

# Tell the Gemini library which key to use
genai.configure(api_key=_api_key)


# ============================================================
# ── GLOBAL CSS  (Apple-minimal light theme)
#
# Design principles:
#  • Near-white canvas (#FBFBFD), white cards, no heavy borders
#  • Cards: white bg, gentle shadow, 16 px rounded corners
#  • Typography: system-font stack (-apple-system → "SF Pro Display")
#  • Accent: #0A84FF (Apple blue)  |  risk colours muted/pastel
#  • Section labels: 11 px small-caps, tracked, muted gray
#
# BUG FIX — title clipped by Streamlit toolbar:
#   .block-container { padding-top: 3rem !important; }
# ============================================================
st.markdown("""
<style>

/* ── System-font stack — mirrors what Apple products use ──────── */
html, body, [class*="css"], .stApp, button, input, select, textarea {
    font-family: -apple-system, "SF Pro Display", "SF Pro Text",
                 "Segoe UI", Helvetica, Arial, sans-serif !important;
}

/* ── Main canvas ───────────────────────────────────────────────── */
.stApp { background-color: #FBFBFD; }

/* ── BUG FIX #1: push content below the Streamlit toolbar so the
   title is never clipped and the logo never overlaps heading text ── */
.block-container {
    padding-top:    3rem !important;
    padding-bottom: 2.5rem;
    max-width:      100%;
}

/* ── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #F5F5F7;
    border-right:     1px solid #E5E5EA;
}

/* ── White card panel ───────────────────────────────────────────── */
/* Usage: <div class="card">…content…</div>                         */
/* Gives each section a floating white panel with a soft shadow.    */
.card {
    background:    #FFFFFF;
    border-radius: 16px;
    padding:       20px 22px;
    margin-bottom: 16px;
    box-shadow:    0 1px 3px rgba(0,0,0,0.06),
                   0 8px 24px rgba(0,0,0,0.04);
}

/* ── Metric cards (top row) ─────────────────────────────────────── */
.metric-card {
    background:    #FFFFFF;
    border-radius: 16px;
    padding:       20px 14px;
    text-align:    center;
    min-height:    108px;
    box-shadow:    0 1px 3px rgba(0,0,0,0.06),
                   0 8px 24px rgba(0,0,0,0.04);
}
/* Uppercase label above the number */
.metric-card .mc-label {
    font-size:      11px;
    font-weight:    500;
    color:          #8E8E93;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom:  8px;
}
/* The headline number */
.metric-card .mc-value {
    font-size:   28px;
    font-weight: 700;
    color:       #1D1D1F;
    line-height: 1.1;
}
/* Teal/blue sub-label beneath the number */
.metric-card .mc-sub {
    font-size:   11px;
    font-weight: 500;
    color:       #0A84FF;
    margin-top:  4px;
}

/* ── Section label — Apple small-caps style ─────────────────────── */
/* No border, no teal, just tracked uppercase gray text.             */
.section-header {
    font-size:      11px;
    font-weight:    600;
    color:          #8E8E93;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    margin-bottom:  14px;
}

/* ── Title bar ──────────────────────────────────────────────────── */
.title-bar {
    display:         flex;
    align-items:     flex-start;
    justify-content: space-between;
    margin-bottom:   28px;
    padding-bottom:  20px;
    border-bottom:   1px solid #E5E5EA;
}
/* App name — large, tight tracking */
.title-main {
    font-size:      28px;
    font-weight:    700;
    color:          #1D1D1F;
    letter-spacing: -0.02em;
    line-height:    1.2;
}
/* Subtitle line — muted gray, comfortable reading size */
.title-sub  { font-size: 13px; color: #8E8E93; margin-top: 4px; }
/* Date badge on the right — accent blue */
.title-date { font-size: 13px; color: #0A84FF; text-align: right; font-weight: 500; }

/* ── Pond overview HTML table ───────────────────────────────────── */
/* BUG FIX #2: table uses full width; pill labels are shortened      */
/* so the Action column never truncates.                             */
.pond-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.pond-table th {
    text-align:     left;
    padding:        8px 10px;
    color:          #8E8E93;
    font-size:      10.5px;
    font-weight:    600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom:  1px solid #F2F2F7;
    white-space:    nowrap;
}
.pond-table td {
    padding:        7px 10px;
    border-bottom:  1px solid #F2F2F7;
    color:          #1D1D1F;
}
.pond-table tr:hover td { background: #F5F5F7; }

/* ── Risk-level text colours (muted, suitable for white background) */
.risk-high    { color: #DC2626; font-weight: 600; }
.risk-medium  { color: #EA580C; font-weight: 600; }
.risk-low     { color: #16A34A; font-weight: 600; }
.risk-verylow { color: #0891B2; font-weight: 600; }

/* ── Action pills — shortened labels, pastel light palette ─────── */
/* BUG FIX #2: short labels (Harvest / Monitor / Routine / Clear)   */
/* prevent the Action column from overflowing its cell.             */
.pill {
    display:        inline-block;
    border-radius:  999px;
    padding:        3px 10px;
    font-size:      11px;
    font-weight:    600;
    white-space:    nowrap;
    letter-spacing: 0.01em;
}
.pill-red    { background: #FEF2F2; color: #DC2626; }   /* Harvest */
.pill-amber  { background: #FFF7ED; color: #C2410C; }   /* Monitor */
.pill-blue   { background: #EFF6FF; color: #1D4ED8; }   /* Routine */
.pill-green  { background: #F0FDF4; color: #15803D; }   /* Clear   */

/* ── Status grid tile ───────────────────────────────────────────── */
.grid-tile {
    border-radius: 12px;
    padding:       14px 6px;
    text-align:    center;
    margin-bottom: 8px;
}
/* Pond number label — small tracked caps */
.grid-tile .gt-id {
    font-size:      11px;
    font-weight:    600;
    color:          #8E8E93;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
/* Risk score — large and bold */
.grid-tile .gt-score { font-size: 22px; font-weight: 700; line-height: 1.3; }

/* ── Plotly chart containers — add card shadow ─────────────────── */
[data-testid="stPlotlyChart"] {
    background:    #FFFFFF;
    border-radius: 16px;
    overflow:      hidden;
    box-shadow:    0 1px 3px rgba(0,0,0,0.06),
                   0 8px 24px rgba(0,0,0,0.04);
}

/* ── pydeck map container — rounded corners + shadow ────────────── */
[data-testid="stDeckGlJsonChart"] {
    border-radius: 16px;
    overflow:      hidden;
    box-shadow:    0 1px 3px rgba(0,0,0,0.06),
                   0 8px 24px rgba(0,0,0,0.04);
}

/* ── Row spacer helper ──────────────────────────────────────────── */
.row-gap { margin-bottom: 28px; }

</style>
""", unsafe_allow_html=True)


# ============================================================
# ── CONSTANTS & HELPER FUNCTIONS
# ============================================================

# GPS coordinates (lat, lon) for each of the 8 shrimp ponds.
# Hardcoded because the CSVs do not include geometry columns.
POND_COORDS = {
    "POND_001": (9.18, 105.15),
    "POND_002": (9.05, 105.05),
    "POND_003": (9.29, 105.72),
    "POND_004": (9.22, 105.55),
    "POND_005": (9.60, 106.00),
    "POND_006": (9.45, 105.95),
    "POND_007": (9.95, 105.10),
    "POND_008": (8.95, 105.10),
}


def risk_color_rgb(score):
    """Return an [R, G, B, A] list for pydeck based on the 4-band risk scale."""
    if score >= 0.75:   return [220,  38,  38, 220]   # red    — High
    elif score >= 0.50: return [249, 115,  22, 220]   # orange — Medium
    elif score >= 0.25: return [ 52, 211, 153, 220]   # green  — Low
    else:               return [110, 231, 183, 220]   # mint   — Very Low


def risk_label(score):
    """Return a human-readable risk level string."""
    if score >= 0.75:   return "High"
    elif score >= 0.50: return "Medium"
    elif score >= 0.25: return "Low"
    else:               return "Very Low"


def risk_hex_bg(score):
    """Return a pastel background colour for the status-grid tile (light theme)."""
    if score >= 0.75:   return "#FEF2F2"   # pastel red
    elif score >= 0.50: return "#FFF7ED"   # pastel orange
    elif score >= 0.25: return "#F0FDF4"   # pastel green
    else:               return "#ECFEFF"   # pastel cyan


def risk_css_class(score):
    """Return the CSS class name that colours risk-level text."""
    if score >= 0.75:   return "risk-high"
    elif score >= 0.50: return "risk-medium"
    elif score >= 0.25: return "risk-low"
    else:               return "risk-verylow"


def action_pill_html(flag):
    """
    Return an HTML pill badge for a given action_flag string.
    BUG FIX #2: labels shortened to Harvest / Monitor / Routine / Clear
    so the Action column never truncates.
    """
    flag_upper = str(flag).upper()
    if "HARVEST" in flag_upper:
        return '<span class="pill pill-red">Harvest</span>'
    elif "MONITOR" in flag_upper or "PREPARE" in flag_upper:
        return '<span class="pill pill-amber">Monitor</span>'
    elif "ROUTINE" in flag_upper:
        return '<span class="pill pill-blue">Routine</span>'
    else:
        return '<span class="pill pill-green">Clear</span>'


# ============================================================
# ── GEMINI AI COPILOT
# These functions are preserved exactly from the original app.
# The try/except fallback chain ensures the app never crashes
# if the Gemini API is unreachable.
# ============================================================

# System prompt tells Gemini how to behave as a shrimp-farm advisor
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


def rule_based_explanation(pond_data):
    """
    Rule-based fallback for a SINGLE pond.
    Called when ALL Gemini models have failed so the UI still shows something useful.
    """
    score = pond_data["risk_score"]
    if score >= 0.7:    level = "High"
    elif score >= 0.4:  level = "Moderate"
    else:               level = "Low"
    return (
        f"This pond is at {level} risk (score {score:.2f}). "
        f"Main driver: {pond_data['top_driver']}. "
        f"Recommended action: {pond_data['action_flag']}."
    )


def get_copilot_explanation(pond_data):
    """
    Ask Gemini for a 3-sentence risk explanation for a SINGLE pond.
    Tries multiple model names in order (most preferred first).
    Falls back to rule_based_explanation if all models fail.
    """
    pond_dict = pond_data.to_dict()
    prompt = f"""{SYSTEM_PROMPT}

Here is the pond risk data:
{pond_dict}

Write the 3-sentence recommendation now."""

    # Try Gemini models in priority order
    # (gemini-flash-lite-latest confirmed working with this key)
    for model_name in ["gemini-flash-lite-latest", "gemini-2.5-flash-lite", "gemini-2.5-flash"]:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.warning(f"Gemini failed ({model_name}): {e}")

    # All models failed — use rule-based fallback so the UI never breaks
    return rule_based_explanation(pond_data)


def get_top3_attention(top3_df):
    """
    Ask Gemini which ponds need immediate attention.
    Passes the top-3 highest-risk ponds as context.
    Falls back to a formatted rule-based summary if all models fail.
    """
    # Convert top-3 rows to a list of dicts for the prompt
    summary = top3_df[["pond_id", "risk_score", "top_driver", "action_flag"]].to_dict("records")

    prompt = f"""{SYSTEM_PROMPT}

A cooperative manager asked: "Which ponds need immediate attention and why?"

Here are the top-3 highest-risk ponds right now:
{summary}

Answer in 4–5 sentences, naming each pond and its key risk driver."""

    for model_name in ["gemini-flash-lite-latest", "gemini-2.5-flash-lite", "gemini-2.5-flash"]:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.warning(f"Gemini failed ({model_name}): {e}")

    # Rule-based fallback — build a simple bullet list for each top-3 pond
    lines = []
    for _, r in top3_df.iterrows():
        lines.append(
            f"• {r['pond_id']}: risk {r['risk_score']:.2f}, "
            f"driver {r['top_driver']}, action {r['action_flag']}."
        )
    return "Gemini unavailable — rule-based summary:\n" + "\n".join(lines)


# ── Live weather  (cached 30 min — free Open-Meteo API, no key needed) ──
@st.cache_data(ttl=1800)
def fetch_live_weather(lat, lon):
    """
    Fetch current weather conditions from the free Open-Meteo API.
    Returns a dict with temperature, humidity, precipitation, and wind speed.
    Returns None silently if the request fails (widget is decorative).
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
        f"&timezone=Asia%2FBangkok"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        c = r.json()["current"]
        return {
            "temperature_c":    c["temperature_2m"],
            "humidity_pct":     c["relative_humidity_2m"],
            "precipitation_mm": c["precipitation"],
            "wind_speed_kmh":   c["wind_speed_10m"],
        }
    except Exception:
        return None


# ============================================================
# ── DATA LOADING
# @st.cache_data means Streamlit will NOT re-read the CSV on
# every button click — it re-uses the cached DataFrame until
# the cache is explicitly cleared (e.g. by the Replan button).
# ============================================================

@st.cache_data
def load_model_output():
    """Load the ML pipeline predictions (risk scores, action flags, etc.)."""
    return pd.read_csv("model_output.csv")


@st.cache_data
def load_pond_features():
    """Load the environmental feature data (drought, temp, DO, etc.) for the radar chart."""
    return pd.read_csv("pond_features.csv")


# Attempt to load both files; show a clear error if either is missing
try:
    df      = load_model_output()    # columns: pond_id, date, risk_score, hotspot, top_driver, action_flag
    feat_df = load_pond_features()   # columns: pond_id, date, drought_index, water_index, temp_c, do_mgl, storm_exposure, …
except FileNotFoundError as e:
    st.error(f"Data file missing: {e}. Run the pipeline scripts first to generate the CSVs.")
    st.stop()

# ── Derived DataFrames ──────────────────────────────────────────────────

# The most recent date available = the "current dekad"
latest_date = df["date"].max()

# One row per pond — keep only the latest date, then sort highest risk first.
latest_df = (
    df.sort_values("date")
    .groupby("pond_id", as_index=False)
    .last()
    .sort_values("risk_score", ascending=False)
    .reset_index(drop=True)
)

# Latest environmental features — used only for the radar chart
latest_feat = (
    feat_df.sort_values("date")
    .groupby("pond_id", as_index=False)
    .last()
    .reset_index(drop=True)
)

today_str = datetime.date.today().strftime("%d %b %Y")

st.markdown(f"""
<div class="title-bar">
  <div>
    <div class="title-main">🦐&nbsp; QuteShrimp Dashboard</div>
    <div class="title-sub">Mekong Delta Shrimp Cooperatives · Risk Intelligence</div>
  </div>
  <div class="title-date">
    📅 {today_str}<br>
    <span style="color:#8E8E93; font-size:11px; font-weight:400;">
      Latest data: {latest_date}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    # App identity in sidebar header
    st.markdown("## 🦐 QuteShrimp")
    st.markdown("---")

    # ── Model status (decorative — hardcoded) ─────────────────────
    st.markdown("### ⚙️ Model Status")
    st.markdown("""
    <div class="card" style="font-size:13px; line-height:2em;">
      🟢 <b>QRC Classifier:</b> Online<br>
      🟢 <b>RF Regressor:</b> Online<br>
      🕒 <b>Last training:</b> 22 May 2026
    </div>
    """, unsafe_allow_html=True)

    # ── Data sources (decorative — hardcoded) ─────────────────────
    st.markdown("### 🗂️ Data Sources")
    st.markdown("""
    <div class="card" style="font-size:13px; line-height:2em;">
      ✅ GDO FAPAR (2025)<br>
      ✅ Salinity / NDWI (2023)<br>
      ✅ IBTrACS Storms (1980–2022)
    </div>
    """, unsafe_allow_html=True)

    # ── Pond selector ─────────────────────────────────────────────
    # Choosing a pond here updates the Copilot panel in the main area.
    st.markdown("### 🎯 Select Pond")
    pond_ids_list = latest_df["pond_id"].tolist()   # already sorted by risk desc
    selected_pond_id = st.selectbox(
        "Pond",
        pond_ids_list,
        label_visibility="collapsed",
    )
    # Pull out the full row for the selected pond so we can pass it to Gemini
    pond_data = latest_df[latest_df["pond_id"] == selected_pond_id].iloc[0]

    st.markdown("---")

    # ── Replan / Re-run button ────────────────────────────────────
    # Clears both @st.cache_data caches so both CSVs are re-read on next render.
    if st.button("🔄 Re-run / Replan", use_container_width=True):
        load_model_output.clear()
        load_pond_features.clear()
        st.success("Caches cleared — reloading data…")
        st.rerun()


# ============================================================
# ── TOP METRIC ROW  (5 summary cards)
# ============================================================

n_hotspot   = int(latest_df["hotspot"].sum())
pct_hotspot = round(n_hotspot / len(latest_df) * 100)
n_high_risk = int((latest_df["risk_score"] >= 0.75).sum())
avg_risk    = latest_df["risk_score"].mean()
n_alerts    = int(
    latest_df["action_flag"].isin(["HARVEST_NOW", "MONITOR_CLOSELY"]).sum()
)

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="mc-label">⚠️ At-Risk Ponds</div>
      <div class="mc-value">{n_hotspot} / 8</div>
      <div class="mc-sub">{pct_hotspot}% of fleet</div>
    </div>""", unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="mc-label">🔴 High Risk</div>
      <div class="mc-value">{n_high_risk}</div>
      <div class="mc-sub">score ≥ 0.75</div>
    </div>""", unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="mc-label">📊 Avg Risk Score</div>
      <div class="mc-value">{avg_risk:.2f}</div>
      <div class="mc-sub">current dekad</div>
    </div>""", unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
      <div class="mc-label">🌧️ Weather Outlook</div>
      <div class="mc-value" style="font-size:22px">Rainy</div>
      <div class="mc-sub">Mekong Delta</div>
    </div>""", unsafe_allow_html=True)

with m5:
    st.markdown(f"""
    <div class="metric-card">
      <div class="mc-label">🚨 Active Alerts</div>
      <div class="mc-value">{n_alerts}</div>
      <div class="mc-sub">Harvest + Monitor</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="row-gap"></div>', unsafe_allow_html=True)


col_map, col_table, col_copilot = st.columns([1.1, 1.2, 0.9])


with col_map:
    st.markdown('<div class="section-header">🗺&nbsp; Pond Risk Map</div>', unsafe_allow_html=True)

    # Build one dict per pond for pydeck
    map_rows = []
    for _, row in latest_df.iterrows():
        coords = POND_COORDS.get(row["pond_id"])
        if not coords:
            continue
        # Extract just the integer pond number: POND_001 → "1"
        pond_num = str(int(row["pond_id"].split("_")[1]))
        map_rows.append({
            "pond_id":    row["pond_id"],
            "pond_num":   pond_num,             # short label shown on map
            "lat":        coords[0],
            "lon":        coords[1],
            "risk_score": float(row["risk_score"]),
            "action":     row["action_flag"],
            "color":      risk_color_rgb(row["risk_score"]),
        })
    map_df = pd.DataFrame(map_rows)

    # Layer 1: coloured circles — slightly larger radius for a light basemap
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=6000,
        pickable=True,
        auto_highlight=True,
        highlight_color=[10, 132, 255, 80],   # Apple-blue highlight
    )

    # Layer 2: pond numbers only (BUG FIX #3 — smaller font, dark text for light map)
    text_layer = pdk.Layer(
        "TextLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_text="pond_num",
        get_size=12,                           # smaller than before (was 14)
        get_color=[30, 30, 30, 255],           # near-black text — readable on light map
        get_anchor="'middle'",
        get_alignment_baseline="'center'",
        get_pixel_offset=[0, 0],               # centred inside the circle
    )

    view_state = pdk.ViewState(latitude=9.35, longitude=105.50, zoom=7)

    st.pydeck_chart(
        pdk.Deck(
            layers=[scatter_layer, text_layer],
            initial_view_state=view_state,
            # Carto Positron: very light, clean, no Mapbox token needed
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            # Tooltip shows full ID + score (BUG FIX #3 requirement)
            tooltip={"text": "{pond_id}\nRisk: {risk_score}\nAction: {action}"},
        ),
        use_container_width=True,
        height=340,
    )

    # Colour legend
    st.markdown("""
    <div style="display:flex; gap:14px; font-size:11px; color:#8E8E93;
                margin-top:8px; flex-wrap:wrap;">
      <span>🔴 High ≥0.75</span>
      <span>🟠 Medium 0.50–0.75</span>
      <span>🟢 Low 0.25–0.50</span>
      <span>🩵 Very Low &lt;0.25</span>
    </div>
    """, unsafe_allow_html=True)

with col_table:
    # Build HTML rows — colour per cell, pills for action flags
    rows_html = ""
    for _, row in latest_df.iterrows():
        lvl      = risk_label(row["risk_score"])
        css      = risk_css_class(row["risk_score"])
        hs       = "Yes" if row["hotspot"] else "No"
        hs_color = "#DC2626" if row["hotspot"] else "#8E8E93"
        pill     = action_pill_html(row["action_flag"])

        rows_html += f"""
        <tr>
          <td><b style="color:#1D1D1F">{row['pond_id']}</b></td>
          <td style="font-variant-numeric:tabular-nums; color:#1D1D1F">
            {row['risk_score']:.2f}</td>
          <td><span class="{css}">{lvl}</span></td>
          <td style="color:{hs_color}">{hs}</td>
          <td style="color:#8E8E93; font-size:11.5px">{row['top_driver']}</td>
          <td>{pill}</td>
        </tr>"""

    # Entire table lives inside a white card panel
    st.markdown(f"""
    <div class="card" style="padding:14px 16px;">
      <div class="section-header" style="margin-bottom:12px;">
        📋&nbsp; Pond Overview — Current Dekad
      </div>
      <table class="pond-table">
        <thead><tr>
          <th>Pond</th>
          <th>Score</th>
          <th>Level</th>
          <th>Hotspot</th>
          <th>Top Driver</th>
          <th>Action</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)


with col_copilot:
    st.markdown('<div class="section-header">🤖&nbsp; AI Copilot</div>', unsafe_allow_html=True)

    # Summary card for the selected pond
    st.markdown(f"""
    <div class="card" style="padding:14px 16px; margin-bottom:12px;">
      <div style="font-size:11px; color:#8E8E93; text-transform:uppercase;
                  letter-spacing:0.08em; font-weight:500; margin-bottom:6px;">
        Selected Pond
      </div>
      <div style="font-size:20px; font-weight:700; color:#1D1D1F;">
        {selected_pond_id}
      </div>
      <div style="font-size:12px; color:#6E6E73; margin-top:4px;">
        Risk {pond_data['risk_score']:.2f}
        &nbsp;·&nbsp;
        <span class="{risk_css_class(pond_data['risk_score'])}">
          {risk_label(pond_data['risk_score'])}
        </span>
        &nbsp;|&nbsp; {pond_data['action_flag']}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Button 1: explain the single pond chosen in the sidebar
    if st.button("💬 Explain this pond", use_container_width=True):
        with st.spinner("Copilot is analysing this pond…"):
            explanation = get_copilot_explanation(pond_data)
            st.info(explanation)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Button 2: fleet-level question — feeds the top-3 risk ponds to Gemini
    if st.button("🚨 Which ponds need immediate attention?", use_container_width=True):
        top3 = latest_df.head(3)
        with st.spinner("Copilot is reviewing the fleet…"):
            fleet_answer = get_top3_attention(top3)
            st.warning(fleet_answer)

    # Live weather widget for the selected pond (Open-Meteo, 30-min cache)
    coords = POND_COORDS.get(selected_pond_id)
    if coords:
        weather = fetch_live_weather(*coords)
        if weather:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="card" style="padding:12px 16px; font-size:12px;">
              <div style="font-size:11px; color:#8E8E93; text-transform:uppercase;
                          letter-spacing:0.08em; font-weight:500; margin-bottom:8px;">
                🌦 Live Weather
              </div>
              <div style="color:#1D1D1F; line-height:2em;">
                🌡️ {weather['temperature_c']}°C
                &nbsp;&nbsp;
                💧 {weather['humidity_pct']}% RH<br>
                🌧️ {weather['precipitation_mm']} mm
                &nbsp;&nbsp;
                💨 {weather['wind_speed_kmh']} km/h
              </div>
            </div>
            """, unsafe_allow_html=True)


# BUG FIX #4: consistent gap before the bottom charts row
st.markdown('<div class="row-gap"></div>', unsafe_allow_html=True)



col_trend, col_radar, col_grid = st.columns([1.1, 1.1, 0.8])

with col_trend:
    st.markdown('<div class="section-header">📈&nbsp; Risk Score Trend — All Ponds</div>',
                unsafe_allow_html=True)

    fig_trend = go.Figure()

    for pond_id in sorted(df["pond_id"].unique()):
        pond_ts = (
            df[df["pond_id"] == pond_id]
            .sort_values("date")[["date", "risk_score"]]
        )
        fig_trend.add_trace(go.Scatter(
            x=pond_ts["date"],
            y=pond_ts["risk_score"],
            mode="lines",
            name=pond_id.replace("POND_", "P"),
            line=dict(width=1.8),
        ))

    # Light-theme chart layout
    fig_trend.update_layout(
        template="simple_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            color="#1D1D1F", size=11,
            family="-apple-system, 'SF Pro Display', 'Segoe UI', Helvetica, sans-serif",
        ),
        legend=dict(
            orientation="v", x=1.01, y=1,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#8E8E93"),
        ),
        xaxis=dict(
            showgrid=False,
            linecolor="#E5E5EA",
            tickfont=dict(color="#8E8E93", size=10),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#F2F2F7", gridwidth=1,
            range=[0, 1],
            linecolor="#E5E5EA",
            tickfont=dict(color="#8E8E93", size=10),
            title=dict(text="Risk Score", font=dict(color="#8E8E93", size=11)),
        ),
        margin=dict(l=8, r=8, t=16, b=30),
        height=300,
    )
    st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})


with col_radar:
    st.markdown('<div class="section-header">🕸&nbsp; Risk Drivers — Current Dekad</div>',
                unsafe_allow_html=True)

    axes        = ["drought_index", "water_index", "temp_c_norm", "do_mgl_norm", "storm_exposure"]
    axis_labels = ["Drought", "Water", "Temp", "Dissolved O₂", "Storm"]

    feat_plot = latest_feat.copy()

    # Normalise temperature and dissolved oxygen to 0–1
    t_min, t_max = feat_plot["temp_c"].min(), feat_plot["temp_c"].max()
    feat_plot["temp_c_norm"] = (feat_plot["temp_c"] - t_min) / max(t_max - t_min, 1e-9)

    d_min, d_max = feat_plot["do_mgl"].min(), feat_plot["do_mgl"].max()
    feat_plot["do_mgl_norm"] = (feat_plot["do_mgl"] - d_min) / max(d_max - d_min, 1e-9)

    fig_radar = go.Figure()

    for _, row in feat_plot.iterrows():
        r_vals       = [row[a] for a in axes]
        target_score = float(row.get("risk_score_target", 0))
        rgb          = risk_color_rgb(target_score)[:3]

        fig_radar.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=axis_labels,
            fill="toself",
            fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.10)",
            line=dict(width=1.5, color=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.75)"),
            name=row["pond_id"].replace("POND_", "P"),
        ))

    # Light-theme radar layout
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#FFFFFF",
            radialaxis=dict(
                visible=True, range=[0, 1],
                color="#8E8E93", gridcolor="#F2F2F7",
                tickfont=dict(size=9, color="#8E8E93"),
                linecolor="#E5E5EA",
            ),
            angularaxis=dict(
                color="#6E6E73", gridcolor="#F2F2F7",
                tickfont=dict(size=11),
            ),
        ),
        paper_bgcolor="#FFFFFF",
        font=dict(
            color="#1D1D1F", size=11,
            family="-apple-system, 'SF Pro Display', 'Segoe UI', Helvetica, sans-serif",
        ),
        showlegend=True,
        legend=dict(
            orientation="v", x=1.05, y=1,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#8E8E93"),
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
    )
    st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})


with col_grid:
    st.markdown('<div class="section-header">🔲&nbsp; Pond Status Grid</div>',
                unsafe_allow_html=True)

    grid_df = latest_df.sort_values("pond_id").reset_index(drop=True)

    for i in range(0, len(grid_df), 2):
        g_col1, g_col2 = st.columns(2)
        for tile_col, idx in zip([g_col1, g_col2], [i, i + 1]):
            if idx >= len(grid_df):
                break
            row  = grid_df.iloc[idx]
            bg   = risk_hex_bg(row["risk_score"])      # pastel background
            css  = risk_css_class(row["risk_score"])   # muted text colour
            tile_col.markdown(f"""
            <div class="grid-tile" style="background:{bg};">
              <div class="gt-id">{row['pond_id'].replace('POND_', 'P')}</div>
              <div class="gt-score">
                <span class="{css}">{row['risk_score']:.2f}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:11px; color:#8E8E93; margin-top:4px; letter-spacing:0.01em;">
      Simplified pond status view · colour = risk band
    </div>
    """, unsafe_allow_html=True)
