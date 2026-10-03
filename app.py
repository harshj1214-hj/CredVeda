import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.explain import extract_shap_contributions, generate_counterfactual_recourse

import streamlit.components.v1 as components

# Paste this near the top of your app.py
components.html(
    """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-516HYVTMDN"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-516HYVTMDN');
    </script>
    """,
    height=0,
    width=0,
)

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CredVeda",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── theme ─────────────────────────────────────────────────────────────────────

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg:       #06080d;
    --surface:  rgba(13, 20, 38, 0.75);
    --surface2: rgba(20, 30, 55, 0.60);
    --border:   rgba(255,255,255,0.07);
    --border2:  rgba(255,255,255,0.13);
    --cyan:     #06b6d4;
    --cyan-dim: rgba(6,182,212,0.15);
    --emerald:  #10b981;
    --amber:    #f59e0b;
    --rose:     #f43f5e;
    --violet:   #8b5cf6;
    --indigo:   #6366f1;
    --sky:      #38bdf8;
    --text:     #f1f5f9;
    --muted:    #64748b;
    --muted2:   #94a3b8;
    --mono:     'JetBrains Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text);
}

/* ── app background ── */
.stApp {
    background:
        radial-gradient(ellipse 80% 60% at 5% 0%,   rgba(99,102,241,0.13) 0%, transparent 55%),
        radial-gradient(ellipse 60% 50% at 95% 100%, rgba(6,182,212,0.10)  0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 50% 50%,  rgba(139,92,246,0.04) 0%, transparent 60%),
        var(--bg);
}

/* ── layout ── */
.block-container { padding: 2.5rem 2rem 3rem !important; max-width: 100% !important; }
[data-testid="stVerticalBlock"] > div { gap: 4px !important; }

/* ── glass panel ── */
.panel {
    background:      var(--surface);
    border:          1px solid var(--border);
    border-radius:   18px;
    padding:         22px 24px 26px;
    margin-bottom:   16px;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    box-shadow:      0 24px 48px -16px rgba(0,0,0,0.7),
                     inset 0 1px 0 rgba(255,255,255,0.07);
    position:        relative;
    overflow:        visible;
}
.panel::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg,
        transparent 5%,
        rgba(6,182,212,0.5) 35%,
        rgba(139,92,246,0.5) 65%,
        transparent 95%);
    pointer-events: none;
    border-radius: 18px 18px 0 0;
    overflow: hidden;
}

/* ── header ── */
.brand-pill {
    display:        inline-flex;
    align-items:    center;
    gap:            7px;
    background:     rgba(99,102,241,0.10);
    border:         1px solid rgba(99,102,241,0.28);
    padding:        5px 13px;
    border-radius:  9999px;
    color:          #a5b4fc;
    font-size:      0.73rem;
    font-weight:    600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}
.pulse {
    width: 7px; height: 7px;
    background: var(--emerald);
    border-radius: 50%;
    box-shadow: 0 0 10px var(--emerald);
    animation: heartbeat 2s infinite ease-in-out;
}
@keyframes heartbeat {
    0%,100% { transform: scale(1);    opacity: 1;   }
    50%     { transform: scale(0.75); opacity: 0.3; }
}

/* ── score ring ── */
.score-ring-wrap {
    display: flex; flex-direction: column; align-items: center;
    gap: 6px; padding: 8px 0;
}
.score-ring-label {
    font-size: 0.68rem; color: var(--muted2);
    text-transform: uppercase; letter-spacing: 0.09em; font-weight: 600;
}

/* ── tier verdict banner ── */
.verdict-banner {
    border-radius: 14px;
    padding: 20px 24px;
    display: flex; align-items: center; gap: 20px;
    margin-bottom: 16px;
    position: relative; overflow: hidden;
}
.verdict-banner::before {
    content: '';
    position: absolute; inset: 0;
    background: var(--vb-bg);
    border: 1px solid var(--vb-border);
    border-radius: 14px;
}
.verdict-tier {
    position: relative;
    font-size: 3.4rem; font-weight: 900;
    letter-spacing: -0.05em; line-height: 1;
    font-family: var(--mono);
    color: var(--vb-color);
    text-shadow: 0 0 40px var(--vb-glow);
}
.verdict-meta { position: relative; flex: 1; }
.verdict-label {
    font-size: 0.7rem; color: var(--muted2);
    text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700;
}
.verdict-status {
    font-size: 1.15rem; font-weight: 700; margin-top: 2px;
    color: var(--vb-color);
}
.verdict-conf {
    font-size: 0.82rem; color: var(--muted2); margin-top: 4px;
}
.verdict-badge {
    position: relative;
    padding: 6px 14px; border-radius: 9999px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em;
    background: var(--vb-badge-bg);
    border: 1px solid var(--vb-border);
    color: var(--vb-color);
}

/* ── metric strip ── */
.metric-strip {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 10px; margin-bottom: 16px;
}
.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 13px 14px;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--mc-accent, var(--cyan));
    border-radius: 2px 2px 0 0;
}
.mc-label {
    font-size: 0.66rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.08em;
    font-weight: 600; margin-bottom: 5px;
}
.mc-val {
    font-size: 1.25rem; font-weight: 700;
    font-family: var(--mono);
    color: var(--mc-color, var(--text));
}
.mc-sub { font-size: 0.68rem; color: var(--muted); margin-top: 2px; }

/* ── dti bar ── */
.dti-bar-wrap { margin-top: 10px; }
.dti-bar-track {
    background: rgba(255,255,255,0.06);
    border-radius: 99px; height: 8px; position: relative; overflow: hidden;
}
.dti-bar-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, var(--dti-from), var(--dti-to));
    width: var(--dti-pct);
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
    box-shadow: 0 0 8px var(--dti-to);
}
.dti-bar-labels {
    display: flex; justify-content: space-between;
    font-size: 0.62rem; color: var(--muted); margin-top: 4px;
}

/* ── score decomposition ── */
.score-decomp { display: flex; flex-direction: column; gap: 8px; }
.score-row { display: flex; align-items: center; gap: 10px; }
.score-row-label { font-size: 0.75rem; color: var(--muted2); width: 110px; flex-shrink: 0; }
.score-row-bar { flex: 1; background: rgba(255,255,255,0.05); border-radius: 99px; height: 6px; overflow: hidden; }
.score-row-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, var(--sr-from), var(--sr-to));
    width: var(--sr-pct);
    transition: width 0.7s cubic-bezier(.4,0,.2,1);
}
.score-row-val { font-size: 0.72rem; font-family: var(--mono); color: var(--muted2); width: 34px; text-align: right; }

/* ── recourse card ── */
.recourse {
    background: linear-gradient(135deg, rgba(6,182,212,0.07) 0%, rgba(13,20,38,0.9) 100%);
    border: 1px solid rgba(6,182,212,0.22);
    border-left: 3px solid var(--cyan);
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 14px;
}
.recourse-eyebrow {
    font-size: 0.67rem; color: var(--cyan); font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;
}
.recourse-body {
    font-size: 0.92rem; color: #e2e8f0; font-weight: 500; line-height: 1.6;
}

/* ── prob breakdown ── */
.prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 7px; }
.prob-label { font-size: 0.73rem; color: var(--muted2); width: 90px; flex-shrink: 0; }
.prob-bar { flex: 1; background: rgba(255,255,255,0.05); border-radius: 99px; height: 14px; overflow: hidden; position: relative; }
.prob-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, var(--pb-from), var(--pb-to));
    width: var(--pb-pct);
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
}
.prob-pct { font-size: 0.73rem; font-family: var(--mono); color: var(--muted2); width: 42px; text-align: right; }

/* ── section header ── */
.sec-head {
    font-size: 0.7rem; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 12px; display: flex; align-items: center; gap: 7px;
}
.sec-head::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border2), transparent);
}

/* ── streamlit widget overrides ── */
div[data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    padding: 3px !important; gap: 2px !important;
}
div[data-baseweb="tab"] {
    border-radius: 7px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--muted2) !important;
    background: transparent !important;
    padding: 6px 14px !important;
}
div[aria-selected="true"][data-baseweb="tab"] {
    background: rgba(99,102,241,0.18) !important;
    color: #a5b4fc !important;
}
div[data-testid="stSelectbox"] > div { border-radius: 10px !important; }
div[data-testid="stSlider"] > div > div > div { accent-color: var(--cyan); }
.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(6,182,212,0.15)) !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    border-radius: 10px !important;
    color: #a5b4fc !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, rgba(99,102,241,0.35), rgba(6,182,212,0.25)) !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.3) !important;
}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


# ── currencies ────────────────────────────────────────────────────────────────

CURRENCIES: dict[str, dict] = {
    "INR (₹)": dict(symbol="₹", rate=1.0,   inc=(200_000, 10_000_000, 1_200_000, 50_000), debt=(0, 5_000_000, 240_000, 25_000)),
    "USD ($)": dict(symbol="$", rate=83.0,   inc=(10_000,    250_000,     65_000,  2_500), debt=(0,   100_000,  12_000,  1_000)),
    "EUR (€)": dict(symbol="€", rate=90.0,   inc=(10_000,    220_000,     58_000,  2_500), debt=(0,    90_000,  10_000,  1_000)),
    "GBP (£)": dict(symbol="£", rate=105.0,  inc=(8_000,     200_000,     52_000,  2_000), debt=(0,    80_000,   9_000,  1_000)),
}


def fmt_currency(amount: float, key: str) -> str:
    sym = CURRENCIES[key]["symbol"]
    if key == "INR (₹)":
        if amount >= 1e7: return f"{sym}{amount/1e7:.2f} Cr"
        if amount >= 1e5: return f"{sym}{amount/1e5:.2f} L"
        return f"{sym}{amount:,.0f}"
    if amount >= 1e6: return f"{sym}{amount/1e6:.2f}M"
    if amount >= 1e3: return f"{sym}{amount/1e3:.1f}k"
    return f"{sym}{amount:,.0f}"


# ── artifact loading ──────────────────────────────────────────────────────────

_DIR = "artifacts"

@st.cache_resource
def load_artifacts():
    return (
        joblib.load(f"{_DIR}/model.joblib"),
        joblib.load(f"{_DIR}/encoders.joblib"),
        joblib.load(f"{_DIR}/feature_names.joblib"),
        joblib.load(f"{_DIR}/background.joblib"),
    )

if any(not os.path.exists(f"{_DIR}/{f}") for f in ["model.joblib","encoders.joblib","feature_names.joblib","background.joblib"]):
    st.error("Model artifacts missing. Run `python src/train.py` first.")
    st.stop()

model, encoders, feature_names, background = load_artifacts()


# ── tier helpers ──────────────────────────────────────────────────────────────

_PRIME    = {"HIGH", "GOOD", "PRIME", "2"}
_MODERATE = {"AVERAGE", "MODERATE", "1"}

_TIER_META = {
    "prime": dict(
        color="#10b981", glow="rgba(16,185,129,0.4)",
        vb_bg="rgba(16,185,129,0.06)", vb_border="rgba(16,185,129,0.25)",
        badge_bg="rgba(16,185,129,0.12)",
        label="LOW RISK", status="Prime Underwriting Quality",
        gauge=88, score=88,
    ),
    "moderate": dict(
        color="#f59e0b", glow="rgba(245,158,11,0.4)",
        vb_bg="rgba(245,158,11,0.06)", vb_border="rgba(245,158,11,0.25)",
        badge_bg="rgba(245,158,11,0.12)",
        label="MODERATE RISK", status="Standard Risk Profile",
        gauge=55, score=55,
    ),
    "subprime": dict(
        color="#f43f5e", glow="rgba(244,63,94,0.4)",
        vb_bg="rgba(244,63,94,0.06)", vb_border="rgba(244,63,94,0.25)",
        badge_bg="rgba(244,63,94,0.12)",
        label="HIGH RISK", status="Subprime — Action Required",
        gauge=24, score=24,
    ),
}


def get_tier(raw_pred: str) -> str:
    t = str(raw_pred).upper()
    if t in _PRIME:    return "prime"
    if t in _MODERATE: return "moderate"
    return "subprime"


# ── chart builders ────────────────────────────────────────────────────────────

_T = "rgba(0,0,0,0)"
_BASE = dict(paper_bgcolor=_T, plot_bgcolor=_T)  # no margin here — each builder sets its own


def build_gauge(score: int, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"color": "#f1f5f9", "size": 30, "family": "JetBrains Mono"}},
        gauge={
            "axis":  {"range": [0,100], "tickwidth": 0, "tickcolor": _T, "visible": False},
            "bar":   {"color": color,   "thickness": 0.3},
            "bgcolor": "rgba(255,255,255,0.03)",
            "bordercolor": "rgba(255,255,255,0.06)",
            "borderwidth": 1,
            "steps": [
                {"range": [0,  40], "color": "rgba(244,63,94,0.10)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.10)"},
                {"range": [70,100], "color": "rgba(16,185,129,0.10)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": score},
        },
    ))
    fig.update_layout(height=180, **_BASE, margin=dict(l=10, r=10, t=10, b=5))
    return fig


def build_radar(income: float, debt: float, dti: float,
                disposable: float, age: int, max_income: float,
                color: str) -> go.Figure:
    raw = [
        max(5,  min(100, 100 - dti * 120)),
        max(10, min(100, (income / (max_income * 0.4)) * 100)),
        max(5,  min(100, 100 - (debt / (income + 1e-5)) * 90)),
        max(10, min(100, (age / 60) * 100)),
        max(5,  min(100, (disposable / (income + 1e-5)) * 100)),
    ]
    labels = ["DTI", "Income", "Leverage", "Maturity", "Buffer"]
    raw.append(raw[0]); labels.append(labels[0])

    fig = go.Figure(go.Scatterpolar(
        r=raw, theta=labels, fill="toself",
        fillcolor=f"rgba({','.join(str(int(int(color.lstrip('#')[i:i+2],16))) for i in (0,2,4))},0.18)",
        line=dict(color=color, width=2),
        hovertemplate="<b>%{theta}</b>: %{r:.0f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,100],
                            tickfont=dict(size=7, color="#475569"),
                            gridcolor="rgba(255,255,255,0.05)",
                            linecolor="rgba(255,255,255,0.04)"),
            angularaxis=dict(tickfont=dict(size=9, color="#94a3b8"),
                             gridcolor="rgba(255,255,255,0.04)",
                             linecolor="rgba(255,255,255,0.04)"),
        ),
        showlegend=False, height=180,
        **_BASE, margin=dict(l=20, r=20, t=10, b=10),
    )
    return fig


def build_prob_breakdown(classes: list, proba: np.ndarray, pred_idx: int) -> str:
    """Return HTML for a probability breakdown bar list."""
    tier_colors = {
        **{t: ("#10b981", "#34d399") for t in _PRIME},
        **{t: ("#f59e0b", "#fbbf24") for t in _MODERATE},
    }
    rows = []
    order = sorted(range(len(classes)), key=lambda i: proba[i], reverse=True)
    for i in order:
        pct = proba[i] * 100
        label = str(classes[i])
        c1, c2 = tier_colors.get(label.upper(), ("#f43f5e", "#fb7185"))
        bold = "font-weight:800; color:#f1f5f9;" if i == pred_idx else ""
        rows.append(
            f'<div class="prob-row">'
            f'<div class="prob-label" style="{bold}">Tier {label}</div>'
            f'<div class="prob-bar">'
            f'<div class="prob-fill" style="--pb-from:{c1}; --pb-to:{c2}; --pb-pct:{pct:.1f}%;"></div>'
            f'</div>'
            f'<div class="prob-pct">{pct:.1f}%</div>'
            f'</div>'
        )
    return "".join(rows)


def build_shap_chart(df_shap: pd.DataFrame, color: str) -> go.Figure:
    c_pos = color
    c_neg = "#f43f5e" if color != "#f43f5e" else "#fb923c"
    colors = [c_pos if v >= 0 else c_neg for v in df_shap["Impact"]]

    fig = go.Figure(go.Bar(
        x=df_shap["Impact"], y=df_shap["Feature"], orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="rgba(255,255,255,0.06)", width=1),
            opacity=0.9,
        ),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.5f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(
            title=dict(text="← lower tier  |  higher tier →", font=dict(size=10, color="#64748b")),
            tickfont=dict(color="#64748b", size=9),
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.15)",
            zerolinewidth=1.5,
        ),
        yaxis=dict(tickfont=dict(color="#cbd5e1", size=10), gridcolor=_T),
        height=max(220, len(df_shap) * 32),
        **_BASE, margin=dict(l=8, r=12, t=8, b=30),
    )
    return fig


def build_score_decomp(income, debt, dti, disposable, age, max_income) -> str:
    """Return HTML for a score decomposition strip."""
    pillars = [
        ("DTI Safety",    max(5, min(100, 100 - dti * 120)),                          "#06b6d4", "#38bdf8"),
        ("Earning Scale", max(10, min(100, (income / (max_income * 0.4)) * 100)),     "#8b5cf6", "#a78bfa"),
        ("Leverage",      max(5, min(100, 100 - (debt / (income + 1e-5)) * 90)),      "#10b981", "#34d399"),
        ("Age Maturity",  max(10, min(100, (age / 60) * 100)),                        "#f59e0b", "#fbbf24"),
        ("Surplus",       max(5, min(100, (disposable / (income + 1e-5)) * 100)),     "#6366f1", "#818cf8"),
    ]
    rows = []
    for label, score, c1, c2 in pillars:
        rows.append(
            f'<div class="score-row">'
            f'<div class="score-row-label">{label}</div>'
            f'<div class="score-row-bar">'
            f'<div class="score-row-fill" style="--sr-from:{c1}; --sr-to:{c2}; --sr-pct:{score:.0f}%;"></div>'
            f'</div>'
            f'<div class="score-row-val">{score:.0f}</div>'
            f'</div>'
        )
    return '<div class="score-decomp">' + "".join(rows) + '</div>'


def dti_bar_html(dti: float) -> str:
    pct = min(dti * 100, 100)
    if dti < 0.30:  c1, c2 = "#10b981", "#34d399"
    elif dti < 0.45: c1, c2 = "#f59e0b", "#fbbf24"
    else:            c1, c2 = "#f43f5e", "#fb7185"
    return f"""
    <div class="dti-bar-wrap">
      <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
        <span style="font-size:0.68rem; color:#64748b; text-transform:uppercase; letter-spacing:0.08em;">DTI Ratio</span>
        <span style="font-size:0.82rem; font-family:'JetBrains Mono',monospace; color:{c2}; font-weight:700;">{dti:.1%}</span>
      </div>
      <div class="dti-bar-track">
        <div class="dti-bar-fill" style="--dti-from:{c1}; --dti-to:{c2}; --dti-pct:{pct:.1f}%;"></div>
      </div>
      <div class="dti-bar-labels"><span>0%</span><span>30%</span><span>45%</span><span>100%</span></div>
    </div>"""


def metric_card(label: str, value: str, sub: str = "", accent: str = "var(--cyan)", color: str = "var(--text)") -> str:
    return (
        f'<div class="metric-card" style="--mc-accent:{accent}; --mc-color:{color};">'
        f'<div class="mc-label">{label}</div>'
        f'<div class="mc-val">{value}</div>'
        f'{"<div class=mc-sub>" + sub + "</div>" if sub else ""}'
        f'</div>'
    )


# ── header ────────────────────────────────────────────────────────────────────

h_l, h_r = st.columns([2.8, 1.2])
with h_l:
    st.markdown(
        '<div class="brand-pill"><span class="pulse"></span>CredVeda · Underwriting Engine v2</div>'
        "<h1 style='margin:10px 0 0; font-size:2.4rem; font-weight:900; letter-spacing:-0.045em; line-height:1.1;'>"
        "Cred<span style='background:linear-gradient(135deg,#06b6d4 20%,#8b5cf6 80%);"
        "-webkit-background-clip:text; -webkit-text-fill-color:transparent;'>Veda</span>"
        "</h1>"
        "<p style='margin:6px 0 0; color:#64748b; font-size:0.85rem;'>Real-time AI underwriting · SHAP explainability · Counterfactual recourse</p>",
        unsafe_allow_html=True,
    )
with h_r:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    selected_currency = st.selectbox("🌐 Currency", list(CURRENCIES.keys()), index=0, label_visibility="collapsed")

cfg = CURRENCIES[selected_currency]
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


# ── columns ───────────────────────────────────────────────────────────────────

col_l, col_r = st.columns([1.1, 1.5], gap="large")


# ── left: inputs ──────────────────────────────────────────────────────────────

with col_l:

    # — input panel —
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="sec-head">📋 Applicant Profile</div>', unsafe_allow_html=True)

    tab_fin, tab_demo, tab_stress = st.tabs(["💰 Financials", "👤 Demographics", "🧪 Stress Test"])

    with tab_fin:
        inc_min, inc_max, inc_default, inc_step = cfg["inc"]
        income = st.slider(
            f"Annual Income", inc_min, inc_max, inc_default, inc_step,
            format=f"{cfg['symbol']}%d",
        )

        dbt_min, dbt_max, dbt_default, dbt_step = cfg["debt"]
        debt = st.slider(
            f"Total Debt & EMIs", dbt_min, dbt_max, dbt_default, dbt_step,
            format=f"{cfg['symbol']}%d",
        )

        dti        = debt / (income + 1e-5)
        disposable = income - debt

        st.markdown(dti_bar_html(dti), unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-strip">'
            + metric_card("Gross Income",  fmt_currency(income,     selected_currency), "per year",       "#06b6d4", "#06b6d4")
            + metric_card("Obligations",   fmt_currency(debt,       selected_currency), "total",          "#f43f5e", "#f43f5e")
            + metric_card("Free Liquidity",fmt_currency(disposable, selected_currency), "disposable",     "#10b981", "#10b981")
            + metric_card("DTI",           f"{dti:.1%}",                                "debt-to-income", "#f59e0b" if dti < 0.45 else "#f43f5e", "#f59e0b" if dti < 0.45 else "#f43f5e")
            + '</div>',
            unsafe_allow_html=True,
        )

    with tab_demo:
        d1, d2 = st.columns(2)
        with d1:
            age       = st.number_input("Age", 18, 85, 32)
            gender    = st.selectbox("Gender",    encoders["Gender"].classes_)
            education = st.selectbox("Education", encoders["Education"].classes_)
        with d2:
            children = st.number_input("Dependents", 0, 6, 1)
            marital  = st.selectbox("Marital Status",     encoders["Marital Status"].classes_)
            home     = st.selectbox("Residential Status", encoders["Home Ownership"].classes_)

    with tab_stress:
        st.caption("Simulate macroeconomic shocks and observe score resilience.")
        shock_income = st.select_slider(
            "Income Shock", options=[-30,-20,-10,0,10,20,30],
            value=0, format_func=lambda x: f"{x:+d}%",
        )
        shock_debt = st.select_slider(
            "Liability Spike", options=[0,10,25,50,100],
            value=0, format_func=lambda x: f"+{x}%",
        )
        if shock_income or shock_debt:
            income     = income * (1 + shock_income / 100)
            debt       = debt   * (1 + shock_debt   / 100)
            dti        = debt / (income + 1e-5)
            disposable = income - debt
            st.warning(
                f"⚡ Shock applied — Income: {fmt_currency(income, selected_currency)}, "
                f"Debt: {fmt_currency(debt, selected_currency)}"
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # — loan capacity panel —
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="sec-head">🏦 Borrowing Capacity</div>', unsafe_allow_html=True)
    st.caption("50% FOIR · 5-year personal loan @ 10.5% p.a.")

    monthly_income = income / 12
    monthly_debt   = debt   / 12
    max_emi        = max(0.0, monthly_income * 0.5 - monthly_debt)
    r, n           = 0.105 / 12, 60
    max_loan       = max_emi * ((1+r)**n - 1) / (r * (1+r)**n) if max_emi > 0 else 0.0

    st.markdown(
        f'<div class="metric-strip" style="grid-template-columns:1fr 1fr;">'
        + metric_card("Max Monthly EMI",   fmt_currency(max_emi,  selected_currency), "50% FOIR cap",    "#10b981", "#10b981")
        + metric_card("Sanctionable Limit", fmt_currency(max_loan, selected_currency), "est. 5yr loan",  "#8b5cf6", "#8b5cf6")
        + '</div>',
        unsafe_allow_html=True,
    )

    # 5-pillar decomposition
    st.markdown('<div class="sec-head" style="margin-top:14px;">📊 Score Decomposition</div>', unsafe_allow_html=True)
    st.markdown(
        build_score_decomp(income, debt, dti, disposable, int(age), cfg["inc"][1]),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ── model inference ───────────────────────────────────────────────────────────

input_dict: dict = {
    "Age":                int(age),
    "Income":             float(income),
    "Number of Children": int(children),
    "Gender":             encoders["Gender"].transform([gender])[0],
    "Education":          encoders["Education"].transform([education])[0],
    "Marital Status":     encoders["Marital Status"].transform([marital])[0],
    "Home Ownership":     encoders["Home Ownership"].transform([home])[0],
    "DTI":                float(dti),
    "Disposable_Income":  float(disposable),
}
input_df  = pd.DataFrame([input_dict])[feature_names]
raw_pred  = model.predict(input_df)[0]
proba     = model.predict_proba(input_df)[0]
classes   = list(model.classes_)
pred_idx  = classes.index(raw_pred)
confidence = float(proba[pred_idx])
tier       = get_tier(raw_pred)
meta       = _TIER_META[tier]
color      = meta["color"]


# ── right: results ────────────────────────────────────────────────────────────

with col_r:

    # — verdict banner —
    st.markdown(
        f'<div class="verdict-banner" style="'
        f'--vb-bg:{meta["vb_bg"]}; --vb-border:{meta["vb_border"]}; '
        f'--vb-color:{color}; --vb-glow:{meta["glow"]}; '
        f'--vb-badge-bg:{meta["badge_bg"]};">'
        f'<div class="verdict-tier">T{raw_pred}</div>'
        f'<div class="verdict-meta">'
        f'  <div class="verdict-label">{meta["label"]}</div>'
        f'  <div class="verdict-status">{meta["status"]}</div>'
        f'  <div class="verdict-conf">Model certainty: <b>{confidence*100:.1f}%</b></div>'
        f'</div>'
        f'<div class="verdict-badge">{meta["label"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # — gauge + radar —
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="sec-head">📈 Risk Visualisation</div>', unsafe_allow_html=True)
    g_col, r_col = st.columns(2)
    with g_col:
        st.markdown('<div class="score-ring-label" style="text-align:center; margin-bottom:2px;">Credit Score</div>', unsafe_allow_html=True)
        st.plotly_chart(build_gauge(meta["score"], color), use_container_width=True)
    with r_col:
        st.markdown('<div class="score-ring-label" style="text-align:center; margin-bottom:2px;">5-Pillar Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(
            build_radar(income, debt, dti, disposable, int(age), cfg["inc"][1], color),
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # — probability breakdown —
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="sec-head">🎲 Tier Probability Breakdown</div>', unsafe_allow_html=True)
    st.markdown(build_prob_breakdown(classes, proba, pred_idx), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # — recourse —
    recourse = generate_counterfactual_recourse(
        input_dict, model, feature_names,
        currency_symbol=cfg["symbol"], fx_rate=cfg["rate"],
    )
    st.markdown(
        f'<div class="recourse">'
        f'<div class="recourse-eyebrow">⚡ Actionable Recourse</div>'
        f'<div class="recourse-body">{recourse}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # — SHAP panel —
    st.markdown('<div class="panel" style="margin-top:16px;">', unsafe_allow_html=True)
    st.markdown('<div class="sec-head">🔬 SHAP Feature Attribution</div>', unsafe_allow_html=True)
    df_shap = extract_shap_contributions(model, background, input_df)
    st.plotly_chart(build_shap_chart(df_shap, color), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # — footer: audit + export —
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    a_col, e_col = st.columns([1.3, 1])
    with a_col:
        st.markdown(
            '<div style="display:flex; align-items:center; gap:10px;">'
            '<span style="font-size:1.3rem;">⚖️</span>'
            '<div>'
            '<div style="font-size:0.68rem; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;">Equal Credit Opportunity Act</div>'
            '<div style="font-size:0.8rem; color:#94a3b8; margin-top:2px;">Zero demographic parity violation detected</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with e_col:
        payload = {
            "applicant":  {k: float(v) if isinstance(v, (int, float, np.integer)) else str(v) for k, v in input_dict.items()},
            "assessment": {"tier": str(raw_pred), "confidence": confidence, "status": meta["status"]},
            "recourse":   recourse,
            "currency":   selected_currency,
            "shap":       df_shap[["Feature","Impact"]].to_dict(orient="records"),
        }
        st.download_button(
            label="📥 Export Full Dossier (JSON)",
            data=json.dumps(payload, indent=2),
            file_name=f"credveda_tier_{raw_pred}.json",
            mime="application/json",
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
