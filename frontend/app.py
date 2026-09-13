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

st.markdown(
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
    background: linear-gradient(180deg, #FFFFFF 0%, {COLORS['bg']} 100%) !important;
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

.sidebar-info-box, .sidebar-info-box * {{
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

input[type="radio"], input[type="checkbox"] {{
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

.card, .card * {{
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

.metric-card, .metric-card * {{
    color: {COLORS['text']} !important;
}}

.metric-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(25, 53, 42, 0.08);
}}

.metric-icon {{
    font-size: 20px;
    background: {COLORS['accent']};
    display: inline-flex;
    padding: 8px 10px;
    border-radius: 12px;
    margin-bottom: 10px;
}}

.metric-title {{
    color: {COLORS['text_sub']} !important;
    font-size: 13px;
    font-weight: 500;
}}

.metric-value {{
    font-size: 28px;
    font-weight: 800;
    color: {COLORS['text']} !important;
    margin-top: 2px;
}}

.metric-delta {{
    font-size: 12px;
    font-weight: 600;
    margin-top: 6px;
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    background: #EEF3F0 !important;
    color: {COLORS['text_sub']} !important;
}}

/* RISK BOXES */

.danger {{
    background: {COLORS['danger_bg']};
    border: 1px solid {COLORS['danger_border']};
    padding: 22px;
    border-radius: 18px;
}}

.danger, .danger p, .danger li, .danger b {{
    color: {COLORS['danger_text']} !important;
}}

.danger h3 {{
    color: {COLORS['danger_text']} !important;
    margin-top: 0;
}}

.warning {{
    background: {COLORS['warning_bg']};
    border: 1px solid {COLORS['warning_border']};
    padding: 22px;
    border-radius: 18px;
}}

.warning, .warning p, .warning li, .warning b {{
    color: {COLORS['warning_text']} !important;
}}

.warning h3 {{
    color: {COLORS['warning_text']} !important;
    margin-top: 0;
}}

.safe {{
    background: {COLORS['safe_bg']};
    border: 1px solid {COLORS['secondary']};
    padding: 22px;
    border-radius: 18px;
}}

.safe, .safe p, .safe li, .safe b {{
    color: {COLORS['safe_text']} !important;
}}

.safe h3 {{
    color: {COLORS['safe_text']} !important;
    margin-top: 0;
}}

.blocked {{
    background: {COLORS['danger_bg']};
    border: 2px solid {COLORS['danger_border']};
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    font-size: 18px;
    font-weight: 700;
}}

.blocked, .blocked * {{
    color: {COLORS['danger_text']} !important;
}}

.blockchain {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['primary']};
    padding: 22px;
    border-radius: 18px;
}}

.blockchain, .blockchain p, .blockchain b {{
    color: {COLORS['text']} !important;
}}

.blockchain code {{
    background: {COLORS['bg']} !important;
    padding: 2px 6px;
    border-radius: 6px;
    color: {COLORS['text']} !important;
}}

/* BUTTONS */

.stButton > button {{
    background: {COLORS['primary']} !important;
    color: {COLORS['text']} !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 18px !important;
    font-weight: 700 !important;
    transition: all 180ms ease;
    box-shadow: 0 2px 8px rgba(120, 198, 163, 0.35);
}}

.stButton > button:hover {{
    background: {COLORS['secondary']} !important;
    color: {COLORS['text']} !important;
    transform: translateY(-1px);
}}

/* FILE UPLOADER */

[data-testid="stFileUploaderDropzone"] {{
    background: {COLORS['card']} !important;
    border: 1.5px dashed {COLORS['secondary']} !important;
    border-radius: 14px !important;
}}

[data-testid="stFileUploaderDropzone"] * {{
    color: {COLORS['text_sub']} !important;
}}

[data-testid="stFileUploaderDropzone"] svg {{
    fill: {COLORS['primary']} !important;
}}

[data-testid="stFileUploaderDropzone"] button {{
    background: {COLORS['primary']} !important;
    color: {COLORS['text']} !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}}

[data-testid="stFileUploaderFile"] {{
    background: {COLORS['accent']} !important;
    border-radius: 10px !important;
}}

/* AUDIO INPUT */

[data-testid="stAudioInput"] {{
    background: {COLORS['card']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 16px !important;
    padding: 12px !important;
}}

[data-testid="stAudioInput"] button {{
    background: {COLORS['primary']} !important;
    border-radius: 50% !important;
}}

audio {{
    border-radius: 10px;
    background: {COLORS['card']};
}}

/* ALERTS */

[data-testid="stAlert"] {{
    background: {COLORS['card']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 14px !important;
}}

[data-testid="stAlert"] * {{
    color: {COLORS['text']} !important;
}}

/* PROGRESS */

.stProgress > div > div > div {{
    background: {COLORS['border']} !important;
}}

.stProgress > div > div > div > div {{
    background-image: linear-gradient(
        90deg,
        {COLORS['primary']},
        {COLORS['secondary']}
    ) !important;
}}

hr {{
    border-color: {COLORS['border']} !important;
}}

</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# CNN MODEL
# =========================================================


class VoiceCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, x):

        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x


@st.cache_resource
def load_model():

    model = VoiceCNN()

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


# =========================================================
# AUDIO FEATURE EXTRACTION
# =========================================================


def extract_mel_spectrogram(audio_bytes):

    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

    target_length = 16000 * 4

    if len(audio) < target_length:

        audio = np.pad(audio, (0, target_length - len(audio)))

    else:

        audio = audio[:target_length]

    mel = librosa.feature.melspectrogram(
        y=audio, sr=16000, n_fft=1024, hop_length=256, n_mels=64, power=2.0
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    mel = (mel - mel.mean()) / (mel.std() + 1e-8)

    tensor = torch.tensor(mel, dtype=torch.float32)

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    return tensor.to(DEVICE)


# =========================================================
# AI DETECTION
# =========================================================


def detect_voice(audio_bytes):

    model = load_model()

    features = extract_mel_spectrogram(audio_bytes)

    with torch.no_grad():

        output = model(features)

        probabilities = torch.softmax(output, dim=1)

    bonafide_probability = probabilities[0][0].item() * 100

    spoof_probability = probabilities[0][1].item() * 100

    if spoof_probability < 40:
        risk_level = "LOW"

    elif spoof_probability < 70:
        risk_level = "MEDIUM"

    elif spoof_probability < 90:
        risk_level = "HIGH"

    else:
        risk_level = "CRITICAL"

    prediction = "SPOOF" if spoof_probability >= 50 else "BONAFIDE"

    if prediction == "SPOOF":

        message = "Possible voice cloning or synthetic speech detected."

    else:

        message = "Voice appears consistent with authentic speech."

    return {
        "prediction": prediction,
        "risk_score": spoof_probability,
        "risk_level": risk_level,
        "bonafide_probability": bonafide_probability,
        "spoof_probability": spoof_probability,
        "message": message,
        "device": str(DEVICE),
    }


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="logo">🛡️</div>
            <div>
                <div class="name">VoiceShield AI</div>
                <div class="tagline">Voice Clone Defense</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-label">Menu</div>', unsafe_allow_html=True)

    nav_choice = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🎙️ Voice Analysis"],
        index=0 if st.session_state.nav == "Dashboard" else 1,
        label_visibility="collapsed",
        key="nav_radio",
    )

    st.session_state.nav = (
        "Dashboard" if nav_choice == "🏠 Dashboard" else "Voice Analysis"
    )

    st.markdown(
        '<div class="sidebar-section-label">System</div>', unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-info-box">
        <b>AI Engine</b><br>CNN Voice Spoof Detection<br><br>

        <b>Features</b><br>Mel Spectrogram<br><br>

        <b>Risk Engine</b><br>Low / Medium / High / Critical<br><br>

        <b>Prevention</b><br>Sensitive Action Blocking<br><br>

        <b>Audit</b><br>Blockchain<br><br>

        <b>Privacy</b><br>
        Raw audio is not stored on blockchain.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# TOP HEADER
# =========================================================

page_title = "Dashboard" if st.session_state.nav == "Dashboard" else "Voice Analysis"

crumb = f"VoiceShield AI  ›  {page_title}"

st.markdown(
    f"""
    <div class="topbar">
        <div>
            <div class="crumb">{crumb}</div>
            <h1>{page_title}</h1>
        </div>

        <div class="status-pill status-online">
            🟢 AI Engine Online
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# DASHBOARD
# =========================================================

if st.session_state.nav == "Dashboard":

    st.markdown(
        """
        <div class="card" style="margin-bottom:22px;">
            <div style="font-size:20px; font-weight:800;">
                Welcome back 👋
            </div>

            <div style="margin-top:6px;">
                Monitor voice authenticity in real time and keep
                sensitive actions protected from synthetic voice attacks.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = st.session_state.history

    total_scans = len(history)

    high_risk_count = len(
        [h for h in history if h["risk_level"] in ("HIGH", "CRITICAL")]
    )

    audited_count = 0

    last_risk = history[-1]["risk_level"] if history else "—"

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">🎙️</div>
                <div class="metric-title">Total Scans</div>
                <div class="metric-value">{total_scans}</div>
                <span class="metric-delta">This session</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">🚨</div>
                <div class="metric-title">High Risk Detections</div>
                <div class="metric-value">{high_risk_count}</div>
                <span class="metric-delta">Blocked automatically</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">⛓️</div>
                <div class="metric-title">Audits Recorded</div>
                <div class="metric-value">{audited_count}</div>
                <span class="metric-delta">On-chain</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">📈</div>
                <div class="metric-title">Last Risk Level</div>
                <div class="metric-value">{last_risk}</div>
                <span class="metric-delta">Most recent scan</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Recent Activity</div>', unsafe_allow_html=True
    )

    if history:

        table_rows = [
            {
                "Time": h["time"],
                "Prediction": h["prediction"],
                "Risk Level": h["risk_level"],
                "Risk Score": h["risk_score"],
                "Audited": "—",
            }
            for h in reversed(history)
        ]

        st.dataframe(table_rows, use_container_width=True, hide_index=True)

    else:

        st.markdown(
            """
            <div class="card" style="text-align:center;">
                No scans yet. Run your first voice analysis
                to see activity here.
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# VOICE ANALYSIS
# =========================================================

else:

    st.markdown(
        '<div class="section-title">🎙️ Voice Analysis</div>', unsafe_allow_html=True
    )

    mode = st.radio(
        "Choose input method",
        ["📁 Upload Audio", "🎤 Record From Microphone"],
        horizontal=True,
    )

    audio_bytes = None
    audio_filename = None
    audio_type = None

    # =====================================================
    # UPLOAD
    # =====================================================

    if mode == "📁 Upload Audio":

        uploaded_file = st.file_uploader(
            "Upload voice sample", type=["wav", "flac", "mp3", "ogg"]
        )

        if uploaded_file is not None:

            audio_bytes = uploaded_file.getvalue()

            audio_filename = uploaded_file.name

            audio_type = uploaded_file.type

            st.audio(audio_bytes, format=audio_type)

    # =====================================================
    # MICROPHONE
    # =====================================================

    else:

        st.info("🎤 Record a short voice sample using your microphone.")

        recorded_audio = st.audio_input("Start recording")

        if recorded_audio is not None:

            audio_bytes = recorded_audio.getvalue()

            audio_filename = "live_voice.wav"

            audio_type = "audio/wav"

            st.success("✅ Voice recording captured.")

            st.audio(audio_bytes, format="audio/wav")

    # =====================================================
    # ANALYSIS
    # =====================================================

    if audio_bytes is not None:

        st.markdown("---")

        if st.button("🔍 Analyze Voice", use_container_width=True):

            with st.spinner("🤖 AI is analyzing the voice..."):

                try:

                    result = detect_voice(audio_bytes)

                    prediction = result["prediction"]

                    risk_score = result["risk_score"]

                    risk_level = result["risk_level"]

                    bonafide = result["bonafide_probability"]

                    spoof = result["spoof_probability"]

                    st.session_state.history.append(
                        {
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "prediction": prediction,
                            "risk_level": risk_level,
                            "risk_score": f"{risk_score:.2f}",
                        }
                    )

                    st.toast("Analysis complete", icon="✅")

                    # =====================================
                    # AI RESULT
                    # =====================================

                    st.markdown(
                        '<div class="section-title">' "🤖 AI Detection Result" "</div>",
                        unsafe_allow_html=True,
                    )

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:

                        st.markdown(
                            f"""
                            <div class="metric-card">
                            <div class="metric-title">
                            Prediction
                            </div>
                            <div class="metric-value">
                            {prediction}
                            </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with col2:

                        st.markdown(
                            f"""
                            <div class="metric-card">
                            <div class="metric-title">
                            Risk Score
                            </div>
                            <div class="metric-value">
                            {risk_score:.2f}
                            </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with col3:

                        st.markdown(
                            f"""
                            <div class="metric-card">
                            <div class="metric-title">
                            Risk Level
                            </div>
                            <div class="metric-value">
                            {risk_level}
                            </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with col4:

                        st.markdown(
                            f"""
                            <div class="metric-card">
                            <div class="metric-title">
                            AI Device
                            </div>
                            <div class="metric-value">
                            {DEVICE.type.upper()}
                            </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # =====================================
                    # CONFIDENCE
                    # =====================================

                    st.markdown(
                        '<div class="section-title">'
                        "📊 Detection Confidence"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.write(f"**Bonafide Probability:** " f"{bonafide:.2f}%")

                        st.progress(min(bonafide / 100, 1.0))

                    with col2:

                        st.write(f"**Spoof Probability:** " f"{spoof:.2f}%")

                        st.progress(min(spoof / 100, 1.0))

                    st.info(result["message"])

                    # =====================================
                    # PREVENTION
                    # =====================================

                    st.markdown("---")

                    st.markdown(
                        '<div class="section-title">' "🛡️ Security Action" "</div>",
                        unsafe_allow_html=True,
                    )

                    if risk_level in ["HIGH", "CRITICAL"]:

                        st.markdown(
                            """
                            <div class="danger">

                            <h3>
                            🚨 HIGH-RISK VOICE DETECTED
                            </h3>

                            <p>
                            The AI model detected characteristics
                            consistent with synthetic or manipulated speech.
                            </p>

                            <b>
                            Recommended Security Action:
                            </b>

                            <ul>
                            <li>Block sensitive action</li>
                            <li>Require secondary verification</li>
                            <li>Verify caller through trusted channel</li>
                            <li>Escalate incident if required</li>
                            </ul>

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            """
                            <div class="blocked">

                            🚫 SENSITIVE ACTION BLOCKED

                            <br>

                            <small>
                            Secondary verification required
                            before proceeding.
                            </small>

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    elif risk_level == "MEDIUM":

                        st.markdown(
                            """
                            <div class="warning">

                            <h3>
                            ⚠️ MEDIUM-RISK VOICE
                            </h3>

                            <p>
                            Additional verification is recommended
                            before sensitive actions.
                            </p>

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    else:

                        st.markdown(
                            """
                            <div class="safe">

                            <h3>
                            ✅ LOW-RISK VOICE
                            </h3>

                            <p>
                            No significant spoofing risk detected.
                            </p>

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # =====================================
                    # BLOCKCHAIN
                    # =====================================

                    st.markdown("---")

                    st.markdown(
                        '<div class="section-title">' "⛓️ Blockchain Audit" "</div>",
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        """
                        <div class="blockchain">

                        <h3>
                        ⛓️ Blockchain Audit
                        </h3>

                        <p>
                        <b>Status:</b>
                        Local blockchain audit is disabled
                        in the cloud deployment.
                        </p>

                        <p>
                        The AI detection engine is running
                        directly on Streamlit Cloud.
                        </p>

                        <p>
                        Raw audio is NOT stored on blockchain.
                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # =====================================
                    # PIPELINE
                    # =====================================

                    st.markdown("---")

                    st.markdown(
                        '<div class="section-title">' "🔐 Security Pipeline" "</div>",
                        unsafe_allow_html=True,
                    )

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:

                        st.success("🎙️ Voice Input")

                    with c2:

                        st.info("🤖 AI Detection")

                    with c3:

                        if risk_level in ["HIGH", "CRITICAL"]:

                            st.error("🚫 Action Blocked")

                        else:

                            st.success("✅ Action Allowed")

                    with c4:

                        st.warning("⛓️ Cloud Audit")

                except Exception as e:

                    st.error(f"AI Detection Error: {str(e)}")

    else:

        st.info("👆 Upload audio or record your voice to begin.")
