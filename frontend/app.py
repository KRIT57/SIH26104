import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import io
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="VoiceShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = "models/cnn_voice_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# COLOR SYSTEM
# =========================================================

COLORS = {
    "primary": "#78C6A3",
    "primary_dark": "#5FAE8B",
    "secondary": "#A8DDB5",
    "accent": "#CDECCF",
    "bg": "#F7FBF8",
    "card": "#FFFFFF",
    "text": "#19352A",
    "text_sub": "#668074",
    "border": "#DDEBE2",
    "danger_bg": "#FDECEC",
    "danger_border": "#F0B4B4",
    "danger_text": "#B3261E",
    "warning_bg": "#FDF3E2",
    "warning_border": "#F0D6A0",
    "warning_text": "#92650C",
    "safe_bg": "#E9F7EF",
    "safe_text": "#146C43",
}

# =========================================================
# SESSION STATE
# =========================================================

if "nav" not in st.session_state:
    st.session_state.nav = "Dashboard"

if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# GLOBAL CSS
# =========================================================

st.html(
    f"""
<style>

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
}}

html, body, .stApp {{
    color-scheme: light !important;
    background: {COLORS['bg']} !important;
}}

* {{
    scrollbar-color: {COLORS['secondary']} {COLORS['bg']};
}}

[data-testid="stAppViewContainer"] {{
    background: {COLORS['bg']} !important;
}}

[data-testid="stHeader"] {{
    background: transparent !important;
}}

.block-container {{
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}}

p, span, li, label, div, h1, h2, h3, h4, h5, h6 {{
    color: {COLORS['text']};
}}

/* SIDEBAR */

[data-testid="stSidebar"] {{
    background: linear-gradient(
        180deg,
        #FFFFFF 0%,
        {COLORS['bg']} 100%
    ) !important;
    border-right: 1px solid {COLORS['border']};
}}

[data-testid="stSidebar"] * {{
    color: {COLORS['text']} !important;
}}

[data-testid="stSidebar"] .block-container {{
    padding-top: 1.2rem;
}}

.sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 4px 18px 4px;
    border-bottom: 1px solid {COLORS['border']};
    margin-bottom: 18px;
}}

.sidebar-brand .logo {{
    font-size: 26px;
    background: {COLORS['accent']};
    border-radius: 12px;
    padding: 6px 10px;
}}

.sidebar-brand .name {{
    font-weight: 800;
    font-size: 18px;
    color: {COLORS['text']} !important;
    line-height: 1.1;
}}

.sidebar-brand .tagline {{
    font-size: 11px;
    color: {COLORS['text_sub']} !important;
}}

.sidebar-section-label {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: {COLORS['text_sub']} !important;
    text-transform: uppercase;
    margin: 18px 4px 6px 4px;
}}

.sidebar-info-box {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 14px;
    padding: 14px 16px;
    margin-top: 8px;
    font-size: 13px;
    line-height: 1.7;
}}

.sidebar-info-box b {{
    color: {COLORS['text']} !important;
}}

.sidebar-info-box,
.sidebar-info-box * {{
    color: {COLORS['text_sub']} !important;
}}

/* RADIO */

div[role="radiogroup"] {{
    gap: 4px;
    flex-direction: column;
}}

[data-testid="stSidebar"] div[role="radiogroup"] {{
    flex-direction: column;
}}

.block-container div[role="radiogroup"] {{
    flex-direction: row;
}}

div[role="radiogroup"] label {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 9px 14px;
    transition: all 180ms ease;
    cursor: pointer;
}}

[data-testid="stSidebar"] div[role="radiogroup"] label {{
    width: 100%;
}}

div[role="radiogroup"] label:hover {{
    background: {COLORS['accent']} !important;
    border-color: {COLORS['secondary']};
}}

div[role="radiogroup"] label[data-checked="true"],
div[role="radiogroup"] label:has(input:checked) {{
    background: {COLORS['accent']} !important;
    border-color: {COLORS['primary']} !important;
}}

div[role="radiogroup"] label p,
div[role="radiogroup"] label span,
div[role="radiogroup"] label div {{
    color: {COLORS['text']} !important;
    font-weight: 500;
}}

input[type="radio"],
input[type="checkbox"] {{
    accent-color: {COLORS['primary']} !important;
}}

/* TOP HEADER */

.topbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 18px;
    padding: 18px 24px;
    margin-bottom: 22px;
    box-shadow: 0 2px 10px rgba(25, 53, 42, 0.04);
}}

.topbar .crumb {{
    font-size: 12px;
    color: {COLORS['text_sub']} !important;
    margin-bottom: 4px;
}}

.topbar h1 {{
    font-size: 22px;
    font-weight: 800;
    color: {COLORS['text']} !important;
    margin: 0;
}}

.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid {COLORS['border']};
}}

.status-online {{
    background: {COLORS['accent']};
    color: {COLORS['safe_text']} !important;
    border-color: {COLORS['secondary']};
}}

.status-offline {{
    background: {COLORS['danger_bg']};
    color: {COLORS['danger_text']} !important;
    border-color: {COLORS['danger_border']};
}}

/* CARDS */

.card {{
    background: {COLORS['card']};
    padding: 22px;
    border-radius: 18px;
    border: 1px solid {COLORS['border']};
    box-shadow: 0 2px 10px rgba(25, 53, 42, 0.04);
}}

.card,
.card * {{
    color: {COLORS['text']} !important;
}}

.section-title {{
    font-size: 18px;
    font-weight: 700;
    color: {COLORS['text']} !important;
    margin: 26px 0 12px 0;
}}

/* METRIC CARDS */

.metric-card {{
    background: {COLORS['card']};
    padding: 20px;
    border-radius: 18px;
    border: 1px solid {COLORS['border']};
    box-shadow: 0 2px 10px rgba(25, 53, 42, 0.04);
    transition: transform 180ms ease, box-shadow 180ms ease;
}}

.metric-card,
.metric-card * {{
    color: {COLORS['text']} !important;
}}

.metric-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(25, 53, 42,