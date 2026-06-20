
"""
PhishGuard — AI Threat Detection Dashboard
Run: streamlit run app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import re

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG       = "#020817"
NAV_BG   = "#030d1f"
CARD     = "#061428"
CARD2    = "#071830"
BORDER   = "#0f2d52"
BORDER2  = "#1a4a7a"
BLUE     = "#3b82f6"
BLUELO   = "#60a5fa"
CYAN     = "#06b6d4"
GOLD     = "#f59e0b"
RED      = "#ef4444"
GREEN    = "#10b981"
VIOLET   = "#8b5cf6"
TEXT     = "#e2eeff"
MUTED    = "#4a7090"
FAINT    = "#0f2a45"
GLOW_B   = "rgba(59,130,246,0.18)"
GLOW_C   = "rgba(6,182,212,0.14)"
GLOW_G   = "rgba(245,158,11,0.14)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body, [class*="css"], .stApp {{
    background: {BG} !important;
    color: {TEXT};
    font-family: 'Inter', sans-serif;
}}

::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER2}; border-radius: 4px; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {NAV_BG} !important;
    border-right: 1px solid {BORDER} !important;
    min-width: 230px !important;
    max-width: 230px !important;
}}
[data-testid="stSidebar"] > div:first-child {{ padding: 0 !important; }}

/* ── Main ── */
.main .block-container {{
    padding: 28px 36px 64px !important;
    max-width: 1380px !important;
    background: transparent !important;
}}

/* ── Radio nav ── */
div[data-testid="stRadio"] > div {{
    gap: 1px !important;
    flex-direction: column !important;
}}
div[data-testid="stRadio"] label {{
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: {MUTED} !important;
    padding: 9px 16px 9px 20px !important;
    border-radius: 0 !important;
    border-left: 2px solid transparent !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    letter-spacing: 0.02em !important;
}}
div[data-testid="stRadio"] label:hover {{
    color: {BLUELO} !important;
    background: {GLOW_B} !important;
    border-left-color: {BLUE} !important;
}}
div[data-testid="stRadio"] label[aria-checked="true"] {{
    color: {BLUELO} !important;
    background: {GLOW_B} !important;
    border-left-color: {BLUE} !important;
}}

/* ── Inputs ── */
.stTextArea textarea, .stTextInput input {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    caret-color: {CYAN} !important;
    padding: 10px 12px !important;
}}
.stTextArea textarea:focus, .stTextInput input:focus {{
    border-color: {BLUE} !important;
    box-shadow: 0 0 0 3px {GLOW_B}, 0 0 20px {GLOW_B} !important;
    outline: none !important;
}}
.stTextArea label, .stTextInput label {{
    color: {MUTED} !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}}

/* ── Primary button ── */
.stButton > button {{
    background: linear-gradient(135deg, {BLUE} 0%, #1d4ed8 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    padding: 10px 20px !important;
    letter-spacing: 0.04em !important;
    transition: opacity 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 4px 20px {GLOW_B} !important;
    text-transform: uppercase !important;
}}
.stButton > button:hover {{
    opacity: 0.9 !important;
    box-shadow: 0 4px 28px rgba(59,130,246,0.45) !important;
}}

/* Ghost buttons */
.btn-ghost .stButton > button {{
    background: transparent !important;
    border: 1px solid {BORDER2} !important;
    color: {MUTED} !important;
    box-shadow: none !important;
    font-size: 0.75rem !important;
    padding: 8px 14px !important;
    text-transform: none !important;
    letter-spacing: 0.01em !important;
    font-weight: 400 !important;
}}
.btn-ghost .stButton > button:hover {{
    color: {BLUELO} !important;
    border-color: {BLUE} !important;
    background: {GLOW_B} !important;
    box-shadow: none !important;
    opacity: 1 !important;
}}

/* ── Warning ── */
.stAlert {{
    background: rgba(245,158,11,0.08) !important;
    border: 1px solid rgba(245,158,11,0.25) !important;
    border-radius: 8px !important;
    color: {GOLD} !important;
    font-size: 0.82rem !important;
}}

/* ── Dataframe ── */
.stDataFrame {{
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}}
[data-testid="stDataFrame"] th {{
    background: {CARD2} !important;
    color: {MUTED} !important;
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-bottom: 1px solid {BORDER} !important;
    padding: 10px 14px !important;
}}
[data-testid="stDataFrame"] td {{
    background: {CARD} !important;
    color: {TEXT} !important;
    font-size: 0.8rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-bottom: 1px solid {FAINT} !important;
    padding: 9px 14px !important;
}}

.js-plotly-plot {{ border-radius: 10px; overflow: hidden; }}
#MainMenu, footer, [data-testid="stDecoration"] {{ display: none !important; }}

/* ── Spinner ── */
.stSpinner > div {{ border-top-color: {BLUE} !important; }}

/* ── Animations ── */
@keyframes fadein {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.fadein {{ animation: fadein 0.4s ease forwards; }}

@keyframes pulse-dot {{
    0%, 100% {{ box-shadow: 0 0 0 0 currentColor; opacity:1; }}
    50%       {{ box-shadow: 0 0 0 5px transparent; opacity:0.7; }}
}}
</style>
""", unsafe_allow_html=True)



for k, v in [("scan_history", []), ("total_scanned", 0), ("threats_detected", 0), ("email_prefill", "")]:
    if k not in st.session_state:
        st.session_state[k] = v


def analyse(text: str) -> dict:
    t = text.lower()
    urgency    = sum(w in t for w in ["urgent","immediately","suspend","terminate","expire","act now"])
    credential = sum(w in t for w in ["password","verify","login","credential","account"])
    financial  = sum(w in t for w in ["payment","bank","transfer","bitcoin","gift card"])
    urls       = len(re.findall(r'https?://\S+', text))
    brand_imp  = any(b in t for b in ["paypal","apple","amazon","microsoft","google","netflix"])
    ip_url     = bool(re.search(r'https?://\d+\.\d+\.\d+\.\d+', text))
    shortener  = bool(re.search(r'bit\.ly|tinyurl|goo\.gl', text))

    score = min(max(int(
        urgency*12 + credential*10 + financial*8 +
        urls*5 + brand_imp*15 + ip_url*20 + shortener*12 +
        random.randint(-4, 4)
    ), 2), 100)

    factors = []
    if urgency > 0:    factors.append(("Urgency language",        urgency*12,    "red"))
    if credential > 0: factors.append(("Credential request",      credential*10, "red"))
    if brand_imp:      factors.append(("Brand impersonation",     15,            "red"))
    if ip_url:         factors.append(("IP-based URL",            20,            "red"))
    if shortener:      factors.append(("URL shortener detected",  12,            "gold"))
    if financial > 0:  factors.append(("Financial language",      financial*8,   "gold"))
    if urls > 1:       factors.append(("Multiple external links", urls*5,        "gold"))
    factors.sort(key=lambda x: x[1], reverse=True)

    votes = {
        "Logistic Regression": round(max(0.04, score/100 - 0.08 + random.uniform(-0.05, 0.05)), 3),
        "Random Forest":       round(max(0.04, score/100 + 0.02 + random.uniform(-0.04, 0.04)), 3),
        "XGBoost":             round(max(0.04, score/100 + 0.04 + random.uniform(-0.03, 0.03)), 3),
        "Ensemble":            round(score/100, 3),
    }
    level = "HIGH RISK" if score > 60 else ("SUSPICIOUS" if score > 30 else "SAFE")
    return {"score": score, "level": level, "phishing": score > 30, "factors": factors, "votes": votes}


def score_color(s):
    return RED if s > 60 else (GOLD if s > 30 else GREEN)

def score_glow(s):
    if s > 60:  return "rgba(239,68,68,0.20)"
    if s > 30:  return "rgba(245,158,11,0.18)"
    return "rgba(16,185,129,0.18)"

def score_level(s):
    return "HIGH RISK" if s > 60 else ("SUSPICIOUS" if s > 30 else "SAFE")

def badge(s):
    c   = score_color(s)
    bg  = score_glow(s)
    lbl = score_level(s)
    return f"""<span style="background:{bg}; color:{c}; border:1px solid {c}44;
        border-radius:4px; padding:3px 12px; font-family:'JetBrains Mono',monospace;
        font-size:0.65rem; font-weight:600; letter-spacing:0.1em;">{lbl}</span>"""

def section_hdr(title):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin:28px 0 16px;">
      <div style="font-family:'Rajdhani',sans-serif;font-size:0.72rem;font-weight:600;
                  text-transform:uppercase;letter-spacing:0.16em;color:{MUTED};">{title}</div>
      <div style="flex:1;height:1px;background:linear-gradient(to right,{BORDER},{BG});"></div>
    </div>""", unsafe_allow_html=True)

def plotly_base(fig, h=240):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD,
        font=dict(color=MUTED, family="Inter", size=10),
        margin=dict(l=0, r=0, t=8, b=0), height=h,
    )
    return fig

def glowcard(color=BLUE, style=""):
    return f"""background:{CARD};border:1px solid {BORDER};border-radius:12px;
               box-shadow:0 0 30px {color}18, inset 0 1px 0 {BORDER2};
               padding:22px 24px;{style}"""


with st.sidebar:
    # Logo
    st.markdown(f"""
    <div style="padding:22px 20px 22px;border-bottom:1px solid {BORDER};margin-bottom:8px;">
      <div style="display:flex;align-items:center;gap:10px;">
        <div style="width:32px;height:32px;border-radius:8px;
                    background:linear-gradient(135deg,{BLUE},{CYAN});
                    display:flex;align-items:center;justify-content:center;
                    font-size:0.9rem;box-shadow:0 0 16px {GLOW_B};">🛡</div>
        <div>
          <div style="font-family:'Rajdhani',sans-serif;font-size:1.1rem;font-weight:700;
                      color:{TEXT};letter-spacing:0.05em;line-height:1;">Phishing Detector</div>
          <div style="font-size:0.58rem;color:{MUTED};letter-spacing:0.14em;text-transform:uppercase;">AI Detection Engine</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("nav", ["Overview", "Analyse Email", "URL Scanner", "Analytics"], label_visibility="collapsed")

    # Stats mini
    if st.session_state.total_scanned > 0:
        t = st.session_state.total_scanned
        th = st.session_state.threats_detected
        st.markdown(f"""
        <div style="margin:16px 16px 0;padding:14px 16px;background:{CARD};
                    border:1px solid {BORDER};border-radius:10px;">
          <div style="font-size:0.6rem;font-weight:600;letter-spacing:0.12em;
                      text-transform:uppercase;color:{MUTED};margin-bottom:10px;">Session</div>
          <div style="display:flex;justify-content:space-between;font-size:0.75rem;margin-bottom:6px;">
            <span style="color:{MUTED};">Scanned</span>
            <span style="font-family:'JetBrains Mono',monospace;color:{BLUELO};">{t}</span>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.75rem;">
            <span style="color:{MUTED};">Threats</span>
            <span style="font-family:'JetBrains Mono',monospace;color:{RED};">{th}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # Engine status
    st.markdown(f"""
    <div style="margin:20px 16px 0;padding:14px 16px;background:{CARD};
                border:1px solid {BORDER};border-radius:10px;">
      <div style="font-size:0.6rem;font-weight:600;letter-spacing:0.12em;
                  text-transform:uppercase;color:{MUTED};margin-bottom:10px;">System Status</div>
    """, unsafe_allow_html=True)
    for name, val, col in [
        ("Parser",         "online",    GREEN),
        ("Feature engine", "online",    GREEN),
        ("ML models",      "demo mode", GOLD),
        ("Explainability", "online",    GREEN),
    ]:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:9px;">
          <div style="width:6px;height:6px;border-radius:50%;background:{col};
                      flex-shrink:0;box-shadow:0 0 6px {col}88;"></div>
          <span style="font-size:1.00rem;color:{MUTED};flex:1;">{name}</span>
          <span style="font-family:'JetBrains Mono',monospace;font-size:0.89rem;color:{col};">{val}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Version tag
    st.markdown(f"""
    <div style="margin:auto 16px 0;padding:16px 16px 0;position:absolute;bottom:16px;left:0;right:0;
                border-top:1px solid {BORDER};text-align:center;">
      <div style="font-size:0.6rem;color:{FAINT};letter-spacing:0.1em;">v2.0 · DEMO MODE</div>
    </div>""", unsafe_allow_html=True)



if page == "Overview":

    if not st.session_state.scan_history:
        demo = [
            ("URGENT: Verify your PayPal account", True, 91),
            ("Your GitHub PR was merged", False, 8),
            ("Reset your Apple ID NOW", True, 88),
            ("Q3 board meeting — agenda", False, 12),
            ("Amazon order flagged", True, 76),
            ("Team standup notes", False, 5),
            ("Claim $500 gift card!", True, 95),
            ("Invoice #2024-0042", False, 22),
        ]
        for subj, ph, sc in demo:
            st.session_state.scan_history.append({
                "time": datetime.now() - timedelta(hours=random.randint(1, 48)),
                "subject": subj, "phishing": ph, "score": sc,
                "level": score_level(sc),
            })
        st.session_state.total_scanned = 1284
        st.session_state.threats_detected = 487

    total   = st.session_state.total_scanned
    threats = st.session_state.threats_detected
    safe    = total - threats
    rate    = f"{(threats / total * 100):.1f}%"

    # Hero header
    st.markdown(f"""
    <div style="margin-bottom:32px;">
      <div style="font-family:'Rajdhani',sans-serif;font-size:2.2rem;font-weight:700;
                  color:{TEXT};letter-spacing:0.02em;line-height:1;margin-bottom:6px;">
        Threat Intelligence
        <span style="background:linear-gradient(90deg,{BLUE},{CYAN});
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;"> Overview</span>
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:{MUTED};">
        {datetime.now().strftime("%A, %d %B %Y  ·  %H:%M:%S")}
      </div>
    </div>""", unsafe_allow_html=True)

    # KPI row
    kpis = [
        (str(total),   "Total Scanned",    BLUE,   GLOW_B,  "+142 this week"),
        (str(threats), "Threats Detected", RED,    "rgba(239,68,68,0.15)",    "+31 this week"),
        (str(safe),    "Clean Emails",     GREEN,  "rgba(16,185,129,0.13)",   "97.3% clean rate"),
        (rate,         "Detection Rate",   GOLD,   GLOW_G,  "↑ 0.4% vs prior"),
    ]
    c1, c2, c3, c4 = st.columns(4, gap="small")
    for col, (val, label, fg, glow, delta) in zip([c1,c2,c3,c4], kpis):
        col.markdown(f"""
        <div style="{glowcard(fg)}position:relative;overflow:hidden;min-height:120px;">
          <div style="position:absolute;top:0;left:0;right:0;height:2px;
                      background:linear-gradient(90deg,{fg},{fg}44,transparent);"></div>
          <div style="font-size:0.62rem;font-weight:600;text-transform:uppercase;
                      letter-spacing:0.14em;color:{MUTED};margin-bottom:10px;">{label}</div>
          <div style="font-family:'Rajdhani',sans-serif;font-size:2.5rem;font-weight:700;
                      color:{fg};line-height:1;text-shadow:0 0 30px {fg}66;">{val}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                      color:{MUTED};margin-top:10px;">{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Charts
    left, right = st.columns([3, 2], gap="medium")

    with left:
        section_hdr("Detection Trend — Last 14 Days")
        days  = [(datetime.now() - timedelta(days=i)).strftime("%b %d") for i in range(13, -1, -1)]
        phish = [random.randint(28, 55) for _ in days]
        legit = [random.randint(60, 110) for _ in days]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=legit, name="Legitimate",
            line=dict(color=GREEN, width=2), fill="tozeroy",
            fillcolor="rgba(16,185,129,0.07)", mode="lines"))
        fig.add_trace(go.Scatter(x=days, y=phish, name="Phishing",
            line=dict(color=RED, width=2), fill="tozeroy",
            fillcolor="rgba(239,68,68,0.08)", mode="lines"))
        plotly_base(fig, 230)
        fig.update_layout(
            legend=dict(orientation="h", y=1.1, x=0, font=dict(size=10),
                        bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor=BORDER, showgrid=True, tickfont=dict(size=9), showline=False),
            yaxis=dict(gridcolor=BORDER, showgrid=True, tickfont=dict(size=9), showline=False),
        )
        st.markdown(f'<div style="{glowcard()}padding:0;overflow:hidden;">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        section_hdr("Classification Breakdown")
        fig2 = go.Figure(go.Pie(
            labels=["High Risk", "Suspicious", "Safe"],
            values=[max(threats-120,1), 120, safe],
            hole=0.70,
            marker=dict(colors=[RED, GOLD, GREEN], line=dict(color=CARD, width=3)),
            textinfo="percent",
            textfont=dict(color=TEXT, size=10, family="JetBrains Mono"),
        ))
        plotly_base(fig2, 230)
        fig2.update_layout(
            showlegend=True,
            legend=dict(orientation="v", x=0.72, y=0.43, font=dict(size=10),
                        bgcolor="rgba(0,0,0,0)"),
        )
        fig2.add_annotation(
            text=f"<b>{total}</b><br><span style='font-size:9px'>total</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color=TEXT, size=14, family="Rajdhani"),
            xref="paper", yref="paper",
        )
        st.markdown(f'<div style="{glowcard()}padding:0;overflow:hidden;">', unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Recent scans table
    section_hdr("Recent Scans")
    st.markdown(f"""
    <div style="{glowcard()}padding:0;overflow:hidden;">
      <div style="display:flex;padding:10px 20px;background:{CARD2};
           border-bottom:1px solid {BORDER};font-size:0.62rem;font-weight:600;
           text-transform:uppercase;letter-spacing:0.12em;color:{MUTED};gap:12px;">
        <div style="width:10px;"></div>
        <div style="flex:1;">Subject</div>
        <div style="width:80px;text-align:center;">Threat Level</div>
        <div style="width:60px;text-align:right;">Score</div>
        <div style="width:88px;text-align:right;">Timestamp</div>
      </div>""", unsafe_allow_html=True)

    rows = sorted(st.session_state.scan_history, key=lambda x: x["time"], reverse=True)[:8]
    for i, item in enumerate(rows):
        sc   = item["score"]
        fg   = score_color(sc)
        lv   = item["level"]
        last = i == len(rows)-1
        st.markdown(f"""
      <div style="display:flex;align-items:center;gap:12px;padding:12px 20px;
           {'border-bottom:1px solid '+FAINT+';' if not last else ''}
           font-size:0.82rem;transition:background 0.15s;"
           onmouseover="this.style.background='{GLOW_B}'"
           onmouseout="this.style.background='transparent'">
        <div style="width:7px;height:7px;border-radius:50%;background:{fg};
                    box-shadow:0 0 8px {fg};flex-shrink:0;"></div>
        <div style="flex:1;color:{TEXT};overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{item['subject']}</div>
        <div style="width:80px;text-align:center;">{badge(sc)}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                    color:{fg};width:60px;text-align:right;font-weight:600;">{sc}/100</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                    color:{MUTED};width:88px;text-align:right;">{item['time'].strftime('%d %b %H:%M')}</div>
      </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)



elif page == "Analyse Email":

    st.markdown(f"""
    <div style="margin-bottom:28px;">
      <div style="font-family:'Rajdhani',sans-serif;font-size:2.2rem;font-weight:700;
                  color:{TEXT};letter-spacing:0.02em;line-height:1;margin-bottom:6px;">
        Email
        <span style="background:linear-gradient(90deg,{BLUE},{CYAN});
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;"> Analysis</span>
      </div>
      <div style="font-size:0.78rem;color:{MUTED};">Paste an email to run the full AI threat analysis pipeline.</div>
    </div>""", unsafe_allow_html=True)

    PHISHING_EX = """From: PayPal Security <security@paypa1-secure.com>
Reply-To: noreply@foreign-domain.ru
Subject: URGENT: Your PayPal account has been SUSPENDED!

Dear valued customer,

We have detected unauthorized access to your PayPal account.
You must verify your password and credit card details IMMEDIATELY or your account will be permanently terminated.

Click here: http://paypal-secure-login.xyz/verify?id=99821
Or use: http://bit.ly/pp-restore-now

Failure to act within 24 hours will result in permanent closure and legal action.

PayPal Security Team"""

    LEGIT_EX = """From: GitHub <noreply@github.com>
Subject: Your pull request was merged

Hi there,

Your pull request #42 'Refactor authentication module' was successfully merged into main by @reviewer.

Changes: 14 files changed, 312 insertions, 87 deletions.

View: https://github.com/your-org/your-repo/pull/42

Thanks!
The GitHub Team"""

    c_in, c_meta = st.columns([3, 1], gap="medium")
    with c_in:
        email_body = st.text_area("Email body", height=180,
            value=st.session_state.email_prefill,
            placeholder="Paste the full email — headers, body, URLs, everything…",
            label_visibility="visible")
    with c_meta:
        sender  = st.text_input("Sender address", placeholder="user@domain.com")
        subject = st.text_input("Subject line",   placeholder="Email subject…")

    b1, b2, b3, _ = st.columns([2, 1.6, 1.5, 3], gap="small")
    with b1:
        run_btn = st.button("⬡  Run Analysis", use_container_width=True)
    with b2:
        st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
        ph_btn = st.button("⚠ Phishing demo", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with b3:
        st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
        ok_btn = st.button("✓ Legit demo", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if ph_btn:
        st.session_state.email_prefill = PHISHING_EX
        st.rerun()
    if ok_btn:
        st.session_state.email_prefill = LEGIT_EX
        st.rerun()

    if run_btn:
        if not email_body.strip():
            st.warning("Paste an email body first.")
        else:
            with st.spinner("Running threat analysis pipeline…"):
                r = analyse(email_body)

            st.session_state.total_scanned += 1
            if r["phishing"]:
                st.session_state.threats_detected += 1
            st.session_state.scan_history.append({
                "time": datetime.now(),
                "subject": subject or "(no subject)",
                "phishing": r["phishing"],
                "score": r["score"],
                "level": r["level"],
            })

            sc    = r["score"]
            fg    = score_color(sc)
            glow  = score_glow(sc)
            icon  = "⚠" if r["phishing"] else "✓"
            vtext = "PHISHING DETECTED" if r["phishing"] else "LEGITIMATE EMAIL"

            st.markdown("<div class='fadein'>", unsafe_allow_html=True)
            section_hdr("Analysis Result")

            rl, rr = st.columns([2, 3], gap="medium")

            with rl:
                # Gauge
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=sc,
                    number={"suffix": "/100", "font": {"color": fg, "size": 30, "family": "Rajdhani"}},
                    gauge={
                        "axis": {"range": [0,100], "tickwidth":1, "tickcolor": MUTED,
                                 "tickfont": {"size":8}},
                        "bar": {"color": fg, "thickness": 0.15},
                        "bgcolor": CARD, "bordercolor": BORDER,
                        "steps": [
                            {"range":[0,30],   "color": "rgba(16,185,129,0.10)"},
                            {"range":[30,60],  "color": "rgba(245,158,11,0.10)"},
                            {"range":[60,100], "color": "rgba(239,68,68,0.10)"},
                        ],
                        "threshold": {"line": {"color": fg, "width": 2}, "value": sc},
                    }
                ))
                plotly_base(fig_g, 185)
                fig_g.update_layout(margin=dict(l=14,r=14,t=14,b=4))
                st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})

                # Verdict card
                st.markdown(f"""
                <div style="{glowcard(fg)}text-align:center;margin-top:-4px;">
                  <div style="font-family:'Rajdhani',sans-serif;font-size:3.5rem;font-weight:700;
                              color:{fg};line-height:1;text-shadow:0 0 40px {fg}88;">
                    {sc}<span style="font-size:1.2rem;color:{MUTED};">/100</span>
                  </div>
                  <div style="font-family:'Rajdhani',sans-serif;font-size:1.1rem;font-weight:600;
                              color:{fg};margin-top:10px;letter-spacing:0.06em;">
                    {icon} {vtext}
                  </div>
                  <div style="margin-top:10px;">{badge(sc)}</div>
                </div>""", unsafe_allow_html=True)

            with rr:
                section_hdr("Contributing Factors")
                st.markdown(f'<div style="{glowcard()}padding:16px 20px;">', unsafe_allow_html=True)
                if r["factors"]:
                    for label, contrib, sev in r["factors"]:
                        bw   = min(int(contrib / 25 * 100), 100)
                        fcol = RED if sev == "red" else GOLD
                        st.markdown(f"""
                        <div style="display:flex;align-items:center;gap:12px;padding:9px 0;
                                    border-bottom:1px solid {FAINT};font-size:0.8rem;">
                          <div style="width:6px;height:6px;border-radius:50%;background:{fcol};
                                      box-shadow:0 0 6px {fcol};flex-shrink:0;"></div>
                          <span style="flex:2;color:{TEXT};">{label}</span>
                          <div style="flex:1;background:{FAINT};border-radius:3px;height:3px;">
                            <div style="width:{bw}%;background:linear-gradient(90deg,{fcol},{fcol}88);
                                        height:3px;border-radius:3px;"></div>
                          </div>
                          <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                                       color:{MUTED};width:28px;text-align:right;">+{min(contrib,25)}</span>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="color:{GREEN};padding:12px 0;font-size:0.82rem;">✓ No significant risk factors detected</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                section_hdr("Model Confidence")
                st.markdown(f'<div style="{glowcard()}padding:16px 20px;">', unsafe_allow_html=True)
                for model, prob in r["votes"].items():
                    bw   = int(prob * 100)
                    mcol = RED if prob > 0.6 else (GOLD if prob > 0.3 else GREEN)
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:11px;">
                      <span style="font-size:0.72rem;color:{MUTED};width:130px;flex-shrink:0;">{model}</span>
                      <div style="flex:1;background:{FAINT};border-radius:3px;height:5px;">
                        <div style="width:{bw}%;background:linear-gradient(90deg,{mcol},{mcol}88);
                                    height:5px;border-radius:3px;"></div>
                      </div>
                      <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                   color:{mcol};width:42px;text-align:right;">{prob:.1%}</span>
                    </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)



elif page == "URL Scanner":

    st.markdown(f"""
    <div style="margin-bottom:28px;">
      <div style="font-family:'Rajdhani',sans-serif;font-size:2.2rem;font-weight:700;
                  color:{TEXT};letter-spacing:0.02em;line-height:1;margin-bottom:6px;">
        URL
        <span style="background:linear-gradient(90deg,{BLUE},{CYAN});
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;"> Safety Scanner</span>
      </div>
      <div style="font-size:0.78rem;color:{MUTED};">Analyse URLs for phishing indicators and domain reputation.</div>
    </div>""", unsafe_allow_html=True)

    url_input = st.text_input("URL to analyse", placeholder="https://example.com/path?query=value")
    scan_btn  = st.button("⬡  Scan URL")

    if scan_btn:
        if not url_input.strip():
            st.warning("Enter a URL first.")
        else:
            url    = url_input.strip()
            domain = re.sub(r'^https?://', '', url).split('/')[0].split('?')[0]
            tld    = domain.split(".")[-1] if "." in domain else ""

            SHORTENERS = {"bit.ly","tinyurl.com","goo.gl","ow.ly","t.co","buff.ly","adf.ly"}
            SUS_TLDS   = {"xyz","tk","ml","ga","cf","gq","pw","top","click"}
            BRANDS     = {"paypal","apple","amazon","microsoft","google","netflix","facebook"}

            checks = [
                ("HTTPS protocol",           url.startswith("https://"),                                         True),
                ("No IP-based URL",          not bool(re.match(r'https?://\d+\.\d+', url)),                     True),
                ("No URL shortener",         domain not in SHORTENERS,                                          True),
                ("Standard TLD",             tld not in SUS_TLDS,                                               True),
                ("No @ symbol in URL",       "@" not in url,                                                    True),
                ("Reasonable URL length",    len(url) < 100,                                                    True),
                ("No brand/domain mismatch", not any(b in url.lower() and b not in domain.lower() for b in BRANDS), True),
                ("No double slash in path",  "//" not in url[8:],                                               True),
            ]

            passed = sum(1 for _, ok, _ in checks if ok)
            risk   = max(0, min(100, int((1 - passed / len(checks)) * 100) + random.randint(-5, 5)))
            fg     = score_color(risk)
            glow   = score_glow(risk)

            st.markdown("<div class='fadein'>", unsafe_allow_html=True)
            section_hdr("Scan Result")
            c1, c2 = st.columns([1, 2], gap="medium")

            with c1:
                st.markdown(f"""
                <div style="{glowcard(fg)}position:relative;overflow:hidden;">
                  <div style="position:absolute;top:0;left:0;right:0;height:2px;
                              background:linear-gradient(90deg,{fg},{fg}44,transparent);"></div>
                  <div style="font-size:0.6rem;font-weight:600;text-transform:uppercase;
                              letter-spacing:0.14em;color:{MUTED};margin-bottom:12px;">URL Risk Score</div>
                  <div style="font-family:'Rajdhani',sans-serif;font-size:3.8rem;font-weight:700;
                              color:{fg};line-height:1;text-shadow:0 0 40px {fg}66;">
                    {risk}<span style="font-size:1.2rem;color:{MUTED};">/100</span>
                  </div>
                  <div style="margin:14px 0;">{badge(risk)}</div>
                  <div style="border-top:1px solid {BORDER};padding-top:14px;font-size:0.75rem;line-height:1.8;">
                    <div style="color:{MUTED};margin-bottom:2px;">Domain</div>
                    <div style="font-family:'JetBrains Mono',monospace;color:{BLUELO};font-size:0.72rem;word-break:break-all;">{domain}</div>
                    <div style="color:{MUTED};margin-top:6px;margin-bottom:2px;">Length</div>
                    <div style="font-family:'JetBrains Mono',monospace;color:{TEXT};">{len(url)} chars</div>
                    <div style="color:{MUTED};margin-top:6px;margin-bottom:2px;">Protocol</div>
                    <div style="font-family:'JetBrains Mono',monospace;color:{'#10b981' if url.startswith('https') else '#f59e0b'};">
                      {'HTTPS  ✓' if url.startswith('https') else 'HTTP  ⚠'}
                    </div>
                    <div style="color:{MUTED};margin-top:6px;margin-bottom:2px;">Checks passed</div>
                    <div style="font-family:'JetBrains Mono',monospace;color:{fg};">{passed}/{len(checks)}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

            with c2:
                section_hdr("Security Checks")
                st.markdown(f'<div style="{glowcard()}padding:16px 20px;">', unsafe_allow_html=True)
                for label, ok, _ in checks:
                    icol = GREEN if ok else RED
                    icon = "✓" if ok else "✗"
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:12px;padding:10px 0;
                                border-bottom:1px solid {FAINT};font-size:0.82rem;">
                      <div style="width:22px;height:22px;border-radius:50%;
                                  background:{icol}18;border:1px solid {icol}44;
                                  display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                        <span style="font-size:0.7rem;font-weight:700;color:{icol};">{icon}</span>
                      </div>
                      <span style="color:{TEXT if ok else MUTED};">{label}</span>
                    </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)


elif page == "Analytics":

    st.markdown(f"""
    <div style="margin-bottom:28px;">
      <div style="font-family:'Rajdhani',sans-serif;font-size:2.2rem;font-weight:700;
                  color:{TEXT};letter-spacing:0.02em;line-height:1;margin-bottom:6px;">
        Threat
        <span style="background:linear-gradient(90deg,{BLUE},{CYAN});
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;"> Analytics</span>
      </div>
      <div style="font-size:0.78rem;color:{MUTED};">Aggregated detection statistics and attack pattern analysis.</div>
    </div>""", unsafe_allow_html=True)

    # Mini stat strip
    mini = [("38%","Credential Harvesting",RED),("26%","Brand Impersonation",GOLD),
            ("18%","Financial Scam",BLUE),("10%","Malware Delivery",VIOLET),("8%","AI-Generated",GREEN)]
    cols = st.columns(5, gap="small")
    for col, (val, label, fg) in zip(cols, mini):
        col.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;
                    padding:14px 14px 12px;border-top:2px solid {fg};">
          <div style="font-family:'Rajdhani',sans-serif;font-size:1.6rem;font-weight:700;
                      color:{fg};line-height:1;">{val}</div>
          <div style="font-size:0.65rem;color:{MUTED};margin-top:5px;line-height:1.3;">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # Charts row
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        section_hdr("SHAP Feature Importance")
        feats = {
            "Credential request":   0.42, "Urgency language":      0.38,
            "Brand impersonation":  0.35, "Suspicious URL / TLD":  0.31,
            "Reply-To mismatch":    0.28, "URL shortener":         0.22,
            "Generic salutation":   0.18, "IP-based URL":          0.15,
            "Executable attachment":0.12, "Excess capitalisation": 0.09,
        }
        bar_colors = [
            f"rgba(239,68,68,{0.5 + 0.5*v/0.42})" for v in feats.values()
        ]
        fig2 = go.Figure(go.Bar(
            x=list(feats.values()), y=list(feats.keys()), orientation="h",
            marker_color=bar_colors,
        ))
        plotly_base(fig2, 280)
        fig2.update_layout(
            xaxis=dict(gridcolor=BORDER, showgrid=True, title="Mean |SHAP value|",
                       tickfont=dict(size=9)),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
            margin=dict(l=0, r=20, t=8, b=0),
        )
        st.markdown(f'<div style="{glowcard()}padding:0;overflow:hidden;">', unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        section_hdr("Risk Score Distribution")
        np.random.seed(42)
        ls = np.random.beta(1.5, 9,   780) * 100
        ps = np.random.beta(8,   1.5, 487) * 100
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=ls, name="Legitimate",
            marker_color=GREEN, opacity=0.60, xbins=dict(size=5)))
        fig3.add_trace(go.Histogram(x=ps, name="Phishing",
            marker_color=RED,   opacity=0.60, xbins=dict(size=5)))
        fig3.add_vline(x=30, line_dash="dot", line_color=GOLD,
            annotation_text="Suspicious", annotation_font_color=GOLD, annotation_font_size=10)
        fig3.add_vline(x=60, line_dash="dot", line_color=RED,
            annotation_text="High Risk",  annotation_font_color=RED,  annotation_font_size=10)
        plotly_base(fig3, 280)
        fig3.update_layout(
            barmode="overlay",
            xaxis=dict(title="Risk Score", gridcolor=BORDER, tickfont=dict(size=9)),
            yaxis=dict(title="Email count", gridcolor=BORDER, tickfont=dict(size=9)),
            legend=dict(orientation="h", y=1.06, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        )
        st.markdown(f'<div style="{glowcard()}padding:0;overflow:hidden;">', unsafe_allow_html=True)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Model comparison table
    section_hdr("Model Performance Comparison")
    perf = pd.DataFrame({
        "Model":      ["Logistic Regression", "Random Forest", "XGBoost", "Ensemble"],
        "AUC-ROC":    [0.941, 0.967, 0.981, 0.984],
        "Precision":  [0.912, 0.944, 0.961, 0.968],
        "Recall":     [0.889, 0.931, 0.953, 0.959],
        "F1 Score":   [0.900, 0.937, 0.957, 0.963],
        "False Neg.": ["11.1%", "6.9%", "4.7%", "4.1%"],
    })
    st.dataframe(
        perf.style
            .highlight_max(subset=["AUC-ROC","Precision","Recall","F1 Score"],
                           props=f"background-color:rgba(16,185,129,0.15);color:{GREEN};")
            .format({"AUC-ROC":"{:.3f}","Precision":"{:.3f}","Recall":"{:.3f}","F1 Score":"{:.3f}"}),
        use_container_width=True, hide_index=True,
    )
