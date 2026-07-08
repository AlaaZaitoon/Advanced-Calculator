"""
app.py — Advanced Numerical Calculator (Streamlit Edition)
===========================================================
A production-ready Streamlit web dashboard wrapping the calculator.py backend.

Architecture
────────────
  Sidebar   : method selector + all input controls
  Main area : 3-tab output system (Summary / Step-by-Step / Visualization)
  Charts    : Plotly (interactive, dark-themed, hover tooltips)
  Matrices  : st.data_editor (Excel-style grid editing)
  State     : st.session_state for persistence across re-renders
  Math      : st.latex() / Markdown $...$ for all formulae

Methods (fixed order)
────────────────────
  0  Doolittle's LU Decomposition
  1  Gauss-Seidel Iteration
  2  Method of False Position       ← index 2 / 3rd item always
  3  Newton-Raphson Method
  4  Newton's Forward Interpolation
  5  Stirling's Central Difference Interpolation
  6  Lagrange's Interpolation

Run:  streamlit run app.py
"""

from __future__ import annotations

import math
import re
import traceback
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as _stc

try:
    import calculator
except ImportError as exc:
    st.error(
        "❌  **calculator.py not found.**  "
        "Place `calculator.py` in the same directory as `app.py`."
    )
    st.stop()

# Optional dep: streamlit-option-menu powers the elegant sidebar nav.
# We degrade gracefully to st.radio if the package is absent so the
# app still runs in a bare environment.
try:
    from streamlit_option_menu import option_menu
    _HAS_OPTION_MENU = True
except ImportError:
    _HAS_OPTION_MENU = False

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be the first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Advanced Numerical Calculator",
    page_icon="∑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS — polishes Streamlit's native dark theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* ══════════════════════════════════════════════════════════════════════
       Calm Midnight — premium academic theme for the
       Advanced Numerical Calculator.
       ──────────────────────────────────────────────────────────────────────
       • Background  : deep navy gradient (#0B1220 → #0F1A2E)
       • Surfaces    : slightly elevated #131F38 / #192847 cards
       • Accents     : soft cyan #7DD3FC + muted violet #A78BFA
       • Typography  : Inter (UI) + JetBrains Mono (code / step blocks)
       ══════════════════════════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg            : #0B1220;
        --bg-elevated   : #0F1A2E;
        --surface       : #131F38;
        --surface-2     : #192847;
        --border        : #1F2D4A;
        --border-soft   : #172339;
        --text          : #E6EDF7;
        --text-muted    : #94A3B8;
        --text-dim      : #64748B;
        --accent        : #7DD3FC;     /* soft cyan         */
        --accent-strong : #38BDF8;     /* bolder cyan       */
        --accent-violet : #A78BFA;     /* muted violet      */
        --accent-violet-strong: #8B5CF6;
        --ok            : #34D399;
        --warn          : #FBBF24;
        --err           : #F87171;
        --shadow-soft   : 0 1px 2px rgba(0,0,0,0.30), 0 4px 14px rgba(0,0,0,0.18);
        --shadow-card   : 0 1px 2px rgba(0,0,0,0.35), 0 8px 28px rgba(0,0,0,0.28);
    }

    /* ── Base styles (Mobile default — no query) ── */

    /* ── Global typography & body backdrop ───────────────────────────── */
    /* FIXED: Added fluid clamp() for body text to ensure readability on small screens */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: var(--text);
        font-size: clamp(0.875rem, 2vw, 1rem); 
        background:
            radial-gradient(120vw 60vh at 12% -10%, #14213B 0%, transparent 60%),
            radial-gradient(90vw 50vh at 110% 110%, #1A1240 0%, transparent 55%),
            linear-gradient(180deg, #0B1220 0%, #0A1020 100%);
        background-attachment: fixed;
        letter-spacing: 0.005em;
    }
    body { font-feature-settings: "ss01", "cv11"; }

    /* ── Hide default Streamlit chrome (white-label look) ────────────── */
    #MainMenu, footer { display: none !important; visibility: hidden; }
    
    [data-testid="stToolbarActions"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { 
        visibility: hidden !important; 
    }

    /* Force the Header to be structural but invisible and pass through clicks */
    header[data-testid="stHeader"] {
        background: transparent !important;
        pointer-events: none !important;
    }
    
    /* Ensure children of the header can still be clicked */
    header[data-testid="stHeader"] * {
        pointer-events: auto !important;
    }

    /* Force the Open Arrow to be visible, clickable, and on top of everything */
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        position: fixed !important;
        top: 0.625rem;
        left: 0.625rem;
        z-index: 1000000 !important;
        min-width: 48px;
        min-height: 48px;
        align-items: center;
        justify-content: center;
        background: rgba(11, 18, 32, 0.92);
        border-radius: 0.625rem;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid var(--border-soft);
        box-shadow: var(--shadow-soft);
    }

    /* Ensure the Close Arrow inside the sidebar is also visible */
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        display: flex !important;
        z-index: 1000000 !important;
        min-width: 48px;
        min-height: 48px;
        align-items: center;
        justify-content: center;
    }

    /* ── Main content padding & subtle fade-in ───────────────────────── */
    /* FIXED: Changed to mobile-first default padding, expanding in media queries */
    [data-testid="stMain"] .block-container,
    .main .block-container {
        padding-top: 1.4rem !important;
        padding-bottom: 3rem !important;
        width: 100%;
        max-width: 100%;
        animation: fadeUp 0.45s ease-out both;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Custom scrollbar ────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 0.625rem; height: 0.625rem; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: #1F2D4A; border-radius: 0.625rem;
        border: 2px solid transparent; background-clip: padding-box;
    }
    ::-webkit-scrollbar-thumb:hover { background: #2A3A5C; background-clip: padding-box; }

    /* ── Headings & links ────────────────────────────────────────────── */
    /* FIXED: Applied fluid typography with clamp() for all headings */
    h1, h2, h3, h4, h5 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.012em;
        color: var(--text);
    }
    h1 { font-weight: 800; font-size: clamp(1.8rem, 5vw, 2.5rem); }
    h2 { font-size: clamp(1.5rem, 4vw, 2rem); }
    h3 { font-size: clamp(1.25rem, 3vw, 1.75rem); }
    a, a:visited { color: var(--accent); text-decoration: none; }
    a:hover { color: var(--accent-strong); }

    /* ── Sidebar ─────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0C1426 0%, #0A1020 100%);
        border-right: 1px solid var(--border-soft);
        overflow-y: auto;
        -webkit-overflow-scrolling: touch;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stTextInput label {
        font-size: clamp(0.7rem, 1.5vw, 0.72rem);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        font-weight: 600;
    }
    [data-testid="stSidebar"] hr { border-color: var(--border-soft); }

    /* ── Forms & Inputs (number/text) ────────────────────────────────── */
    /* FIXED: Prevent iOS auto-zoom by enforcing font-size >= 16px (1rem). Set width to 100%. Min-height for touch targets. */
    .stTextInput > div > div input,
    .stNumberInput > div > div input,
    [data-baseweb="input"] input {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: 0.625rem !important;
        color: var(--text) !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        font-size: 1rem !important; /* 16px min */
        min-height: 44px;
        width: 100%;
        transition: border-color 0.18s, box-shadow 0.18s;
    }
    .stTextInput > div > div:focus-within input,
    .stNumberInput > div > div:focus-within input {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(125, 211, 252, 0.18) !important;
    }
    .stNumberInput button {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-muted) !important;
        min-height: 44px;
        min-width: 44px;
    }

    /* ── Selectbox / dropdown popovers ───────────────────────────────── */
    [data-baseweb="select"] > div {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: 0.625rem !important;
        min-height: 44px;
    }
    [data-baseweb="popover"] { border-radius: 0.75rem; box-shadow: var(--shadow-card); z-index: 999999 !important; }

    /* ── Tabs — modern pill / underline hybrid ───────────────────────── */
    /* FIXED: Removed fixed px height, added rems and min-height 44px */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        border-bottom: 1px solid var(--border-soft);
        padding-bottom: 0.125rem;
        background: transparent;
        display: flex;
        flex-wrap: nowrap;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }
    .stTabs [data-baseweb="tab"] {
        min-height: 44px;
        padding: 0 1.25rem;
        border-radius: 0.625rem 0.625rem 0 0;
        font-size: clamp(0.85rem, 2vw, 0.92rem);
        font-weight: 500;
        color: var(--text-muted);
        background: transparent;
        border: none;
        position: relative;
        white-space: nowrap;
        transition: color 0.18s, background 0.18s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text);
        background: rgba(125, 211, 252, 0.06);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent) !important;
        background: rgba(125, 211, 252, 0.10);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"]::after {
        content: "";
        position: absolute;
        left: 1rem; right: 1rem; bottom: -2px;
        height: 2px; border-radius: 2px;
        background: linear-gradient(90deg, var(--accent), var(--accent-violet));
    }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ── Metric cards ────────────────────────────────────────────────── */
    /* FIXED: Swapped px padding for rems, clamp() on metric value */
    [data-testid="stMetric"], [data-testid="metric-container"] {
        background: linear-gradient(180deg, var(--surface) 0%, #111B33 100%);
        border: 1px solid var(--border);
        border-radius: 0.875rem;
        padding: 1rem 1.25rem;
        box-shadow: var(--shadow-soft);
        transition: transform 0.18s, border-color 0.18s, box-shadow 0.18s;
    }
    [data-testid="stMetric"]:hover, [data-testid="metric-container"]:hover {
        border-color: rgba(125, 211, 252, 0.45);
        box-shadow: var(--shadow-card);
        transform: translateY(-1px);
    }
    [data-testid="stMetricLabel"] {
        font-size: clamp(0.65rem, 1.5vw, 0.72rem); text-transform: uppercase; letter-spacing: 0.12em;
        color: var(--text-muted); font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        font-size: clamp(1.25rem, 4vw, 1.6rem); font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: var(--text);
    }

    /* ── Dataframes & Tables ─────────────────────────────────────────── */
    /* FIXED: Enforce mobile horizontal scrolling and max-width 100% */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border-radius: 0.75rem;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        border: 1px solid var(--border);
        box-shadow: var(--shadow-soft);
        background: var(--bg-elevated);
        max-width: 100%;
    }
    [data-testid="stDataFrame"] [role="columnheader"] {
        background: #0E1729 !important;
        color: var(--text) !important;
        font-weight: 600 !important;
    }

    /* ── Expanders ───────────────────────────────────────────────────── */
    /* FIXED: Replaced px padding/margins with rem */
    [data-testid="stExpander"] {
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: 0.75rem;
        box-shadow: var(--shadow-soft);
        margin: 0.625rem 0;
        overflow: hidden;
    }
    [data-testid="stExpander"] details > summary {
        padding: 0.75rem 1.125rem;
        font-weight: 600;
        color: var(--text);
        background: linear-gradient(90deg, #142036 0%, #0E1729 100%);
        min-height: 44px;
    }
    [data-testid="stExpander"] details > summary:hover {
        background: linear-gradient(90deg, #182846 0%, #11203A 100%);
    }

    /* ── KaTeX (LaTeX) blocks — soft framed cards ────────────────────── */
    [data-testid="stMain"] .katex-display {
        background: var(--bg-elevated);
        border: 1px solid var(--border-soft);
        border-radius: 0.75rem;
        padding: 0.875rem 1.125rem;
        margin: 0.75rem 0;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        box-shadow: var(--shadow-soft);
    }
    .katex { color: var(--text) !important; font-size: clamp(0.9em, 2.5vw, 1.06em); }

    /* ── Alert boxes (info / success / warning / error) ──────────────── */
    [data-testid="stAlert"] {
        border-radius: 0.75rem;
        border: 1px solid var(--border);
        box-shadow: var(--shadow-soft);
    }

    /* ── Step-by-step monospaced output ──────────────────────────────── */
    /* FIXED: Fluid padding and clamp() typography for step-by-step blocks */
    .calc-steps {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: clamp(0.75rem, 2vw, 0.85rem);
        line-height: 1.7;
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: 0.75rem;
        padding: 1.25rem 1.5rem;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        white-space: pre;
        color: var(--text);
        box-shadow: var(--shadow-soft);
    }
    .calc-steps .hdr  { color: var(--accent); font-weight: 700; }
    .calc-steps .val  { color: var(--ok); }
    .calc-steps .num  { color: #BBE8B0; }
    .calc-steps .key  { color: #9CDCFE; }
    .calc-steps .warn { color: var(--warn); }
    .calc-steps .err  { color: var(--err); font-weight: 700; }
    .calc-steps .dim  { color: var(--text-dim); }
    .calc-steps .form { color: var(--accent-violet); font-style: italic; }
    .calc-steps .yel  { color: #F2D472; font-weight: 700; }

    /* ── Method info pill (in sidebar method card) ───────────────────── */
    .method-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: clamp(0.6rem, 1.5vw, 0.7rem);
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-right: 0.5rem;
    }
    .pill-direct        { background: rgba(52, 211, 153, 0.10); border: 1px solid rgba(52, 211, 153, 0.55); color: #6EE7B7; }
    .pill-iterative     { background: rgba(251, 191, 36, 0.10); border: 1px solid rgba(251, 191, 36, 0.55); color: #FCD34D; }
    .pill-bracketing    { background: rgba(125, 211, 252, 0.10); border: 1px solid rgba(125, 211, 252, 0.55); color: #7DD3FC; }
    .pill-open          { background: rgba(167, 139, 250, 0.10); border: 1px solid rgba(167, 139, 250, 0.55); color: #C4B5FD; }
    .pill-interpolation { background: rgba(56, 189, 248, 0.10); border: 1px solid rgba(56, 189, 248, 0.55); color: #38BDF8; }

    /* ── Calculate (primary) button ──────────────────────────────────── */
    /* FIXED: Fluid font, rem padding, explicit min-height */
    .stButton > button[kind="primary"] {
        width: 100%;
        background: linear-gradient(135deg, var(--accent-strong), var(--accent-violet-strong));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 0.75rem;
        color: #0B1220;
        font-size: clamp(0.9rem, 2.5vw, 0.98rem);
        font-weight: 700;
        letter-spacing: 0.04em;
        padding: 0.8rem 0;
        min-height: 44px;
        box-shadow: 0 6px 22px rgba(56, 189, 248, 0.25);
        transition: transform 0.15s, box-shadow 0.18s, filter 0.18s;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        filter: brightness(1.06);
        box-shadow: 0 10px 28px rgba(56, 189, 248, 0.32);
    }
    .stButton > button[kind="primary"]:active { transform: translateY(0); filter: brightness(0.95); }

    /* ── Secondary buttons ───────────────────────────────────────────── */
    .stButton > button:not([kind="primary"]) {
        background: var(--surface);
        border: 1px solid var(--border);
        color: var(--text);
        border-radius: 0.625rem;
        min-height: 44px;
        transition: border-color 0.18s, background 0.18s;
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: var(--accent);
        background: var(--surface-2);
    }

    /* ── Section labels (sidebar) ────────────────────────────────────── */
    .section-label {
        font-size: clamp(0.6rem, 1.5vw, 0.7rem);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--text-muted);
        margin: 1.375rem 0 0.5rem 0;
        padding-bottom: 0.375rem;
        border-bottom: 1px solid var(--border-soft);
    }

    /* ── Headline gradient on the main page title ────────────────────── */
    /* FIXED: Fluid clamp scaling for titles and logos */
    .anc-title {
        font-size: clamp(1.4rem, 4vw, 1.9rem);
        font-weight: 800;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.02em;
        background: linear-gradient(120deg, #E6EDF7 0%, #7DD3FC 55%, #A78BFA 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .anc-subtitle {
        color: var(--text-muted);
        font-size: clamp(0.75rem, 2vw, 0.88rem);
        margin: 0;
        letter-spacing: 0.01em;
    }

    /* ── Method header card (top-right of main area) ─────────────────── */
    .method-header-card {
        background: linear-gradient(135deg, rgba(19, 31, 56, 0.85), rgba(15, 26, 46, 0.55));
        border: 1px solid var(--border);
        border-radius: 0.875rem;
        padding: 0.875rem 1.125rem;
        margin-top: 0.375rem;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        box-shadow: var(--shadow-soft);
    }
    .method-header-card .mhc-name {
        font-size: clamp(0.8rem, 2vw, 0.9rem); font-weight: 600; color: var(--text);
    }

    /* ── Sidebar identity card (logo + product name) ─────────────────── */
    .anc-brand {
        text-align: center;
        padding: 0.875rem 0.375rem 0.625rem 0.375rem;
        margin-bottom: 0.375rem;
    }
    .anc-brand .anc-logo {
        font-size: clamp(1.6rem, 5vw, 2.2rem);
        background: linear-gradient(135deg, #7DD3FC, #A78BFA);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .anc-brand .anc-name {
        display: block;
        font-size: clamp(0.9rem, 2.5vw, 1.02rem);
        font-weight: 700;
        color: var(--text);
        margin-top: 0.125rem;
        letter-spacing: 0.005em;
    }
    .anc-brand .anc-tag {
        display: block;
        font-size: clamp(0.6rem, 1.5vw, 0.72rem);
        color: var(--text-muted);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 0.125rem;
    }

    /* ── Method info card body (description + complexity row) ───────── */
    .method-info-card {
        background: linear-gradient(180deg, var(--surface) 0%, #101A30 100%);
        border: 1px solid var(--border);
        border-radius: 0.75rem;
        padding: 0.875rem 1rem;
        margin: 0.75rem 0 0.375rem 0;
        box-shadow: var(--shadow-soft);
    }
    .method-info-card .mic-meta {
        font-size: clamp(0.6rem, 1.5vw, 0.7rem);
        color: var(--text-dim);
        letter-spacing: 0.06em;
    }
    .method-info-card .mic-desc {
        margin: 0.5rem 0 0 0;
        font-size: clamp(0.75rem, 2vw, 0.82rem);
        color: var(--text-muted);
        line-height: 1.55;
    }

    /* ── Sidebar footer (version line) ───────────────────────────────── */
    .anc-footer {
        text-align: center;
        font-size: clamp(0.55rem, 1.5vw, 0.68rem);
        color: var(--text-dim);
        letter-spacing: 0.1em;
        margin-top: 1.375rem;
        padding-top: 0.875rem;
        border-top: 1px solid var(--border-soft);
    }

    /* ── Plotly chart wrappers — soft framed cards ───────────────────── */
    [data-testid="stPlotlyChart"] {
        border-radius: 0.875rem;
        overflow: hidden;
        border: 1px solid var(--border);
        box-shadow: var(--shadow-soft);
        background: var(--bg-elevated);
        max-width: 100%;
    }

    /* ── Divider tint ────────────────────────────────────────────────── */
    hr, [data-testid="stDivider"] { border-color: var(--border-soft); }

    /* ── Selection colour ────────────────────────────────────────────── */
    ::selection { background: rgba(125, 211, 252, 0.32); color: var(--text); }

    /* ── Images ──────────────────────────────────────────────────────── */
    /* FIXED: Global image responsive rules */
    img {
        max-width: 100%;
        height: auto;
        display: block;
    }

    /* ══════════════════════════════════════════════════════════════════
       RESPONSIVE BREAKPOINTS  —  mobile-first
       ══════════════════════════════════════════════════════════════════ */

    /* ── Mobile only: below 768px ── */
    @media (max-width: 767px) {
        /* Sidebar takes full screen width on mobile */
        [data-testid="stSidebar"] {
            width: 100% !important;
            max-width: 100%;
        }

        /* Stack all columns vertically in the main area */
        [data-testid="stMain"] [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.5rem !important;
        }
        [data-testid="stMain"] [data-testid="stHorizontalBlock"] > div {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 0 !important;
        }

        /* Tighter main content padding */
        [data-testid="stMain"] .block-container,
        .main .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }

        /* Smaller tab padding for narrow screens */
        .stTabs [data-baseweb="tab"] {
            padding: 0 0.625rem;
            font-size: 0.8rem;
        }

        /* Tighter metric cards */
        [data-testid="stMetric"], [data-testid="metric-container"] {
            padding: 0.75rem 1rem;
        }

        /* KaTeX blocks: less padding on mobile */
        [data-testid="stMain"] .katex-display {
            padding: 0.5rem 0.625rem;
            margin: 0.5rem 0;
        }

        /* Step-by-step blocks tighter on mobile */
        .calc-steps {
            padding: 0.75rem 0.875rem;
            font-size: 0.73rem;
        }

        /* Method header card stacks nicely */
        .method-header-card {
            margin-top: 0.5rem;
        }

        /* Title sizing for small screens */
        .anc-title {
            font-size: clamp(1.1rem, 5vw, 1.5rem);
        }
        .anc-subtitle {
            font-size: clamp(0.68rem, 2vw, 0.82rem);
        }

        /* Plotly chart wrapper tighter */
        [data-testid="stPlotlyChart"] {
            border-radius: 0.625rem;
        }

        /* Expander tighter padding */
        [data-testid="stExpander"] details > summary {
            padding: 0.625rem 0.875rem;
        }
    }

    /* ── Tablet: 768px and up ── */
    @media (min-width: 768px) {
        [data-testid="stSidebar"] {
            width: auto !important;
        }
        [data-testid="stMain"] .block-container,
        .main .block-container {
            padding-top: 2rem !important;
        }
    }

    /* ── Desktop: 1025px and up ── */
    @media (min-width: 1025px) {
        [data-testid="stMain"] .block-container,
        .main .block-container {
            padding-top: 2.4rem !important;
            padding-bottom: 5rem !important;
            max-width: 1280px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
#  METHOD REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
METHODS: list[str] = [
    "Doolittle's Method (LU Decomposition)",         # 0
    "Gauss-Seidel Iteration",                        # 1
    "Method of False Position",                      # 2  ← index 2 always
    "Newton-Raphson Method",                         # 3
    "Newton's Forward Interpolation",                # 4
    "Stirling's Central Difference Interpolation",   # 5
    "Lagrange's Interpolation",                      # 6
]

# pill_class, complexity, latex_formula, description
METHOD_META: dict[str, tuple[str, str, str, str]] = {
    METHODS[0]: (
        "pill-direct", "O(n³)",
        r"A = LU \;\Rightarrow\; LV = b \;\Rightarrow\; UX = V",
        "Factorises **A = LU** then solves by forward & back substitution.",
    ),
    METHODS[1]: (
        "pill-iterative", "O(n²)/iter",
        r"x_i^{(k+1)} = \frac{b_i - \sum_{j \neq i} a_{ij} x_j^{(k)}}{a_{ii}}",
        "An iterative method to solve a system of linear equations. "
        "It is applicable and guaranteed to converge if the coefficient "
        "matrix is Strictly Diagonally Dominant (SDD).",
    ),
    METHODS[2]: (
        "pill-bracketing", "Linear conv.",
        r"x_{n} = \frac{a \cdot f(b) - b \cdot f(a)}{f(b) - f(a)}",
        "Draws a chord across a sign-change interval — also called **Regula Falsi**.",
    ),
    METHODS[3]: (
        "pill-open", "Quadratic conv.",
        r"x_{n} = x_{n-1} - \frac{f(x_{n-1})}{f'(x_{n-1})}",
        "Tangent-line iteration.  Requires **f(x)** and **f′(x)**.  Fast near a simple root.",
    ),
    METHODS[4]: (
        "pill-interpolation", "O(n²) setup",
        r"f(x) = y_0 + p\,\Delta y_0 + \frac{p(p-1)}{2!}\Delta^2 y_0 + \cdots,\quad p = \frac{x - x_0}{h}",
        "Newton's **forward difference table** — equally spaced nodes only.",
    ),
    METHODS[5]: (
        "pill-interpolation", "O(n²) setup",
        r"f(x) = y_0 + p\,\frac{\Delta y_0 + \Delta y_{-1}}{2} + \frac{p^2}{2!}\Delta^2 y_{-1} + \cdots,\quad p = \frac{x - x_0}{h}",
        "**Stirling central differences** — use when *x* is near the table midpoint; "
        "$x_0$ is chosen as the data point nearest to *x*.",
    ),
    METHODS[6]: (
        "pill-interpolation", "O(n²) eval",
        r"f(x) = \sum_{i=0}^{n} y_i \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}",
        "**Lagrange basis polynomials** — handles unequally spaced data.",
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
#  SAFE EXPRESSION EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════
_SAFE_NS: dict = {
    "__builtins__": None,
    "math": math, "np": np,
    "sin": math.sin,  "cos": math.cos,  "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "exp": math.exp,  "log": math.log,  "log10": math.log10,
    "log2": math.log2, "sqrt": math.sqrt, "abs": abs,
    "pi": math.pi,   "e": math.e,
}


def _ev(expr: str, x) -> float:
    ns = {**_SAFE_NS, "x": x}
    return float(eval(expr, ns))  # noqa: S307


def _make_f(expr: str) -> Callable[[float], float]:
    return lambda x: _ev(expr, x)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY CHART BUILDERS  (dark theme, interactive)
# ══════════════════════════════════════════════════════════════════════════════
_PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0F1A2E",
    plot_bgcolor="#0F1A2E",
    font=dict(family="Inter, sans-serif", size=12, color="#E6EDF7"),
    xaxis=dict(gridcolor="#1F2D4A", zerolinecolor="#2A3A5C", linecolor="#2A3A5C"),
    yaxis=dict(gridcolor="#1F2D4A", zerolinecolor="#2A3A5C", linecolor="#2A3A5C"),
    legend=dict(bgcolor="#131F38", bordercolor="#1F2D4A", borderwidth=1),
    margin=dict(l=50, r=30, t=60, b=50),
    hoverlabel=dict(bgcolor="#131F38", bordercolor="#1F2D4A",
                    font_size=12, font_color="#E6EDF7"),
)


def _fig_lu_heatmap(L: np.ndarray, U: np.ndarray) -> go.Figure:
    """Side-by-side heatmaps for L and U factor matrices."""
    from plotly.subplots import make_subplots

    n   = L.shape[0]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Lower Triangular  L", "Upper Triangular  U"],
    )

    fig.add_trace(go.Heatmap(
        z=L[::-1],
        x=[f"c{i+1}" for i in range(n)],
        y=[f"r{n-i}" for i in range(n)],
        colorscale="Blues", showscale=False,
        hovertemplate="L[%{y}, %{x}] = %{z:.5f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=U[::-1],
        x=[f"c{i+1}" for i in range(n)],
        y=[f"r{n-i}" for i in range(n)],
        colorscale="Teal", showscale=False,
        hovertemplate="U[%{y}, %{x}] = %{z:.5f}<extra></extra>",
    ), row=1, col=2)

    # Add per-cell value labels for both heatmaps
    def _cell_text(M):
        return [[f"{v:.3g}" for v in row] for row in M[::-1]]

    fig.data[0].update(text=_cell_text(L), texttemplate="%{text}",
                       textfont=dict(color="#E6EDF7", size=11,
                                     family="JetBrains Mono, monospace"))
    fig.data[1].update(text=_cell_text(U), texttemplate="%{text}",
                       textfont=dict(color="#E6EDF7", size=11,
                                     family="JetBrains Mono, monospace"))

    fig.update_layout(
        title="LU Factor Matrices",
        height=360,
        paper_bgcolor="#0F1A2E",
        plot_bgcolor="#0F1A2E",
        font=dict(family="Inter, sans-serif", color="#E6EDF7"),
        margin=dict(l=30, r=30, t=70, b=30),
    )
    return fig


# ─── Shared animation defaults ──────────────────────────────────────────────
_ANIM_BTN = dict(
    type="buttons",
    direction="left",
    showactive=False,
    x=0.02, xanchor="left",
    y=1.18, yanchor="top",
    pad=dict(t=4, r=8),
    bgcolor="rgba(19,31,56,0.85)",
    bordercolor="#1F2D4A",
    font=dict(color="#E6EDF7", size=12, family="Inter, sans-serif"),
    buttons=[
        dict(label="▶  Play", method="animate",
             args=[None, dict(frame=dict(duration=900, redraw=True),
                              fromcurrent=True,
                              transition=dict(duration=400, easing="cubic-in-out"))]),
        dict(label="❚❚ Pause", method="animate",
             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0))]),
    ],
)

_ANIM_SLIDER_BASE = dict(
    active=0, x=0.08, y=-0.05, len=0.9,
    pad=dict(b=10, t=40),
    currentvalue=dict(prefix="Iteration  ", visible=True,
                      xanchor="right",
                      font=dict(color="#7DD3FC", size=13,
                                family="JetBrains Mono, monospace")),
    bgcolor="#131F38",
    bordercolor="#1F2D4A",
    activebgcolor="#7DD3FC",
    tickcolor="#2A3A5C",
    font=dict(color="#A8B5CC", size=11),
    transition=dict(duration=300, easing="cubic-in-out"),
)


def _safe_eval_curve(f: Callable[[float], float], xs: np.ndarray) -> np.ndarray:
    """Evaluate *f* on an x-grid, falling back to a Python loop on failure.

    Many user-supplied expressions cannot be applied to a NumPy array
    directly (e.g.  ``math.sin`` rejects arrays).  This helper tries the
    vectorised path first then degrades to a per-element loop, replacing
    any exceptions with ``NaN`` so the curve can still render.
    """
    try:
        ys = f(xs)
        ys = np.asarray(ys, dtype=float)
        if ys.shape != xs.shape:
            raise ValueError("shape mismatch")
        return ys
    except Exception:
        out = np.empty_like(xs, dtype=float)
        for k, xv in enumerate(xs):
            try:
                out[k] = float(f(float(xv)))
            except Exception:
                out[k] = np.nan
        return out


def _fig_false_position_anim(
    f:       Callable[[float], float],
    eq:      str,
    history: list[dict],
    root:    float,
) -> go.Figure:
    """Animated False-Position visualization.

    Each frame shows one Regula-Falsi iteration:
      * the function curve  *f(x)*  (always visible)
      * the bracket endpoints  (a, f(a))  and  (b, f(b))
      * the chord between them
      * the next root estimate  c  with  (c, 0)  and  (c, f(c))

    A play/pause button and a per-iteration slider let the user step
    through convergence at their own pace.
    """
    # Determine an x-window large enough to show every bracket
    a_all = [step["a"] for step in history] + [step["b"] for step in history]
    x_lo, x_hi = min(a_all), max(a_all)
    span = max(x_hi - x_lo, 1e-6)
    x_lo -= 0.25 * span
    x_hi += 0.25 * span

    xs = np.linspace(x_lo, x_hi, 600)
    ys = _safe_eval_curve(f, xs)

    # Pre-compute y-axis range so frames don't rescale
    finite = ys[np.isfinite(ys)]
    y_pad  = (finite.max() - finite.min()) * 0.15 if finite.size else 1.0
    y_lo   = (finite.min() - y_pad) if finite.size else -1.0
    y_hi   = (finite.max() + y_pad) if finite.size else  1.0

    # ── Static base traces  (indices 0–1)
    base_traces = [
        go.Scatter(x=xs, y=ys, mode="lines",
                   name=f"f(x) = {eq}",
                   line=dict(color="#7DD3FC", width=2.5),
                   hovertemplate="x = %{x:.5f}<br>f(x) = %{y:.5f}<extra></extra>"),
        go.Scatter(x=[x_lo, x_hi], y=[0, 0], mode="lines",
                   name="y = 0",
                   line=dict(color="#475569", width=1, dash="dot"),
                   hoverinfo="skip", showlegend=False),
    ]

    # ── Animated overlay traces  (indices 2–6, populated per frame)
    def _frame_traces(step: dict) -> list[go.Scatter]:
        a, b   = step["a"], step["b"]
        fa, fb = step["fa"], step["fb"]
        c, fc  = step["x"],  step["fx"]
        return [
            # 2 — chord between (a, f(a)) and (b, f(b))
            go.Scatter(x=[a, b], y=[fa, fb], mode="lines",
                       name="Chord",
                       line=dict(color="#FBBF24", width=2, dash="dash"),
                       hoverinfo="skip", showlegend=False),
            # 3 — bracket endpoint markers (a, b)
            go.Scatter(x=[a, b], y=[fa, fb], mode="markers+text",
                       name="Bracket",
                       marker=dict(color=["#7DD3FC", "#A78BFA"],
                                   size=12, symbol="circle",
                                   line=dict(color="#FFFFFF", width=1.2)),
                       text=[" a", " b"], textposition="top center",
                       textfont=dict(color="#E6EDF7", size=11),
                       hovertemplate="(%{x:.5f}, %{y:.5f})<extra>%{text}</extra>",
                       showlegend=False),
            # 4 — vertical line at  x = c
            go.Scatter(x=[c, c], y=[y_lo, y_hi], mode="lines",
                       name="x = c",
                       line=dict(color="#F87171", width=1.4, dash="dot"),
                       hoverinfo="skip", showlegend=False),
            # 5 — root estimate marker on the x-axis
            go.Scatter(x=[c], y=[0], mode="markers+text",
                       name="c (estimate)",
                       marker=dict(color="#F87171", size=14, symbol="diamond",
                                   line=dict(color="#FFFFFF", width=1.5)),
                       text=[f"  c = {c:.5f}"], textposition="bottom center",
                       textfont=dict(color="#F87171", size=11,
                                     family="JetBrains Mono, monospace"),
                       hovertemplate=f"c = {c:.8f}<extra>Root estimate</extra>",
                       showlegend=False),
            # 6 — corresponding f(c) marker on the curve
            go.Scatter(x=[c], y=[fc], mode="markers",
                       name="f(c)",
                       marker=dict(color="#FBBF24", size=10, symbol="x-thin",
                                   line=dict(color="#FBBF24", width=2.5)),
                       hovertemplate=f"f(c) = {fc:.6g}<extra></extra>",
                       showlegend=False),
        ]

    # Build frames
    frames: list[go.Frame] = []
    slider_steps = []
    for k, step in enumerate(history, start=1):
        ann = (f"Iter {k:>2}   ·   a = {step['a']:.5f}   b = {step['b']:.5f}   "
               f"c = {step['x']:.7f}"
               + (f"   ·   |Δc| = {step['err']:.2e}"
                  if step['err'] is not None else "   ·   first step"))
        frames.append(go.Frame(
            data=base_traces + _frame_traces(step),
            name=str(k),
            layout=go.Layout(annotations=[dict(
                text=ann, showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=1.04, xanchor="center",
                font=dict(family="JetBrains Mono, monospace",
                          size=12, color="#7DD3FC"),
                bgcolor="rgba(19,31,56,0.85)",
                bordercolor="#1F2D4A", borderwidth=1, borderpad=6,
            )]),
        ))
        slider_steps.append(dict(
            method="animate", label=str(k),
            args=[[str(k)], dict(frame=dict(duration=600, redraw=True),
                                  mode="immediate",
                                  transition=dict(duration=300))],
        ))

    # Initial figure shows the LAST frame so the chart is informative even
    # before the user clicks Play.
    initial_overlay = _frame_traces(history[-1])
    fig = go.Figure(
        data=base_traces + initial_overlay,
        frames=frames,
    )
    final_ann = (f"Converged in {len(history)} iter   ·   "
                 f"root ≈ {root:.7f}")
    fig.update_layout(
        title=dict(text=f"<b>Regula Falsi</b>   ·   {eq}",
                   font=dict(size=15)),
        xaxis=dict(range=[x_lo, x_hi], title="x"),
        yaxis=dict(range=[y_lo, y_hi], title="f(x)"),
        hovermode="x unified",
        updatemenus=[_ANIM_BTN],
        sliders=[{**_ANIM_SLIDER_BASE, "steps": slider_steps,
                  "active": len(slider_steps) - 1}],
        annotations=[dict(text=final_ann, showarrow=False,
                          xref="paper", yref="paper",
                          x=0.5, y=1.04, xanchor="center",
                          font=dict(family="JetBrains Mono, monospace",
                                    size=12, color="#7DD3FC"),
                          bgcolor="rgba(19,31,56,0.85)",
                          bordercolor="#1F2D4A", borderwidth=1, borderpad=6)],
        **{k: v for k, v in _PLOTLY_LAYOUT.items()
           if k not in ("xaxis", "yaxis")},
    )
    return fig


def _fig_newton_raphson_anim(
    f:        Callable[[float], float],
    f_prime:  Callable[[float], float],
    eq:       str,
    fp_eq:    str,
    history:  list[dict],
    root:     float,
) -> go.Figure:
    """Animated Newton-Raphson visualization with tangent lines per step."""
    xs_iter = [step["x"] for step in history]
    x_lo = min(xs_iter + [root]) - 0.6 * (max(xs_iter) - min(xs_iter) + 1)
    x_hi = max(xs_iter + [root]) + 0.6 * (max(xs_iter) - min(xs_iter) + 1)

    xs = np.linspace(x_lo, x_hi, 600)
    ys = _safe_eval_curve(f, xs)

    finite = ys[np.isfinite(ys)]
    y_pad  = (finite.max() - finite.min()) * 0.15 if finite.size else 1.0
    y_lo   = (finite.min() - y_pad) if finite.size else -1.0
    y_hi   = (finite.max() + y_pad) if finite.size else  1.0

    base_traces = [
        go.Scatter(x=xs, y=ys, mode="lines",
                   name=f"f(x) = {eq}",
                   line=dict(color="#7DD3FC", width=2.5),
                   hovertemplate="x = %{x:.5f}<br>f(x) = %{y:.5f}<extra></extra>"),
        go.Scatter(x=[x_lo, x_hi], y=[0, 0], mode="lines",
                   line=dict(color="#475569", width=1, dash="dot"),
                   hoverinfo="skip", showlegend=False),
    ]

    def _tangent_xy(x_p: float, fx_p: float, fpx_p: float
                    ) -> tuple[list[float], list[float]]:
        """Return endpoints of the tangent line clipped to the chart window."""
        if abs(fpx_p) < 1e-14:
            return [x_p, x_p], [y_lo, y_hi]
        ax = x_lo
        bx = x_hi
        ay = fx_p + fpx_p * (ax - x_p)
        by = fx_p + fpx_p * (bx - x_p)
        return [ax, bx], [ay, by]

    def _frame_traces(k: int) -> list[go.Scatter]:
        prev = history[k - 1]
        curr = history[k]
        x_p, fx_p, fpx_p = prev["x"], prev["fx"], prev["fpx"]
        x_n              = curr["x"]
        tx, ty           = _tangent_xy(x_p, fx_p, fpx_p)
        return [
            # 2 — tangent line at x_{n-1}
            go.Scatter(x=tx, y=ty, mode="lines",
                       name="Tangent",
                       line=dict(color="#FBBF24", width=2, dash="dash"),
                       hoverinfo="skip", showlegend=False),
            # 3 — vertical drop from (x_{n-1}, f(x_{n-1})) to x-axis
            go.Scatter(x=[x_p, x_p], y=[0, fx_p], mode="lines",
                       line=dict(color="#A78BFA", width=1.2, dash="dot"),
                       hoverinfo="skip", showlegend=False),
            # 4 — current point on the curve  (x_{n-1}, f(x_{n-1}))
            go.Scatter(x=[x_p], y=[fx_p], mode="markers+text",
                       marker=dict(color="#A78BFA", size=12, symbol="circle",
                                   line=dict(color="#FFFFFF", width=1.2)),
                       text=[f"  xₙ₋₁ = {x_p:.5f}"], textposition="top right",
                       textfont=dict(color="#A78BFA", size=11),
                       hovertemplate=f"x = {x_p:.6f}<br>f(x) = {fx_p:.6g}<extra>iterate</extra>",
                       showlegend=False),
            # 5 — next estimate on the x-axis
            go.Scatter(x=[x_n], y=[0], mode="markers+text",
                       marker=dict(color="#F87171", size=14, symbol="diamond",
                                   line=dict(color="#FFFFFF", width=1.5)),
                       text=[f"  xₙ = {x_n:.5f}"], textposition="bottom center",
                       textfont=dict(color="#F87171", size=11,
                                     family="JetBrains Mono, monospace"),
                       hovertemplate=f"x = {x_n:.8f}<extra>next</extra>",
                       showlegend=False),
        ]

    frames, slider_steps = [], []
    for k in range(1, len(history)):
        prev = history[k - 1]
        curr = history[k]
        ann = (f"Iter {k:>2}   ·   xₙ₋₁ = {prev['x']:.6f}   "
               f"f(xₙ₋₁) = {prev['fx']:.4g}   "
               f"f'(xₙ₋₁) = {prev['fpx']:.4g}   ·   "
               f"xₙ = {curr['x']:.6f}"
               + (f"   ·   |Δx| = {curr['err']:.2e}"
                  if curr['err'] is not None else ""))
        frames.append(go.Frame(
            data=base_traces + _frame_traces(k),
            name=str(k),
            layout=go.Layout(annotations=[dict(
                text=ann, showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=1.04, xanchor="center",
                font=dict(family="JetBrains Mono, monospace",
                          size=12, color="#7DD3FC"),
                bgcolor="rgba(19,31,56,0.85)",
                bordercolor="#1F2D4A", borderwidth=1, borderpad=6)]),
        ))
        slider_steps.append(dict(
            method="animate", label=str(k),
            args=[[str(k)], dict(frame=dict(duration=700, redraw=True),
                                  mode="immediate",
                                  transition=dict(duration=300))],
        ))

    initial_overlay = _frame_traces(len(history) - 1)
    fig = go.Figure(data=base_traces + initial_overlay, frames=frames)
    final_ann = (f"Converged in {len(history) - 1} iter   ·   "
                 f"root ≈ {root:.8f}")
    fig.update_layout(
        title=dict(text=f"<b>Newton-Raphson</b>   ·   {eq}",
                   font=dict(size=15)),
        xaxis=dict(range=[x_lo, x_hi], title="x"),
        yaxis=dict(range=[y_lo, y_hi], title="f(x)"),
        hovermode="x unified",
        updatemenus=[_ANIM_BTN],
        sliders=[{**_ANIM_SLIDER_BASE, "steps": slider_steps,
                  "active": len(slider_steps) - 1}],
        annotations=[dict(text=final_ann, showarrow=False,
                          xref="paper", yref="paper",
                          x=0.5, y=1.04, xanchor="center",
                          font=dict(family="JetBrains Mono, monospace",
                                    size=12, color="#7DD3FC"),
                          bgcolor="rgba(19,31,56,0.85)",
                          bordercolor="#1F2D4A", borderwidth=1, borderpad=6)],
        **{k: v for k, v in _PLOTLY_LAYOUT.items()
           if k not in ("xaxis", "yaxis")},
    )
    return fig


def _fig_gauss_seidel_anim(
    A:       np.ndarray,
    b:       np.ndarray,
    x0:      np.ndarray,
    history: list[np.ndarray],
) -> go.Figure:
    """Dual-panel Gauss-Seidel viz: residual log + per-component trajectories.

    *Top panel*  — ‖Ax − b‖∞ on a log-y axis (linear convergence reads as a
    straight line).  A red marker tracks the iteration currently selected
    by the slider/play head.

    *Bottom panel* — one line per solution component xᵢ, plus markers at
    the current iteration so the user can watch each component settle.
    """
    from plotly.subplots import make_subplots

    iters     = list(range(1, len(history) + 1))
    residuals = [float(np.linalg.norm(A @ h - b, np.inf)) for h in history]
    n         = len(history[0])
    comps     = [[float(h[i]) for h in history] for i in range(n)]

    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.18,
        subplot_titles=(
            "Residual  ‖A x − b‖∞   (log scale)",
            "Per-component trajectory  xᵢ",
        ),
        row_heights=[0.42, 0.58],
    )

    # ── Top panel — full residual line  (idx 0)
    fig.add_trace(go.Scatter(
        x=iters, y=residuals, mode="lines+markers",
        name="‖residual‖∞",
        line=dict(color="#7DD3FC", width=2),
        marker=dict(color="#A78BFA", size=7,
                    line=dict(color="#7DD3FC", width=1)),
        hovertemplate="Iter %{x}<br>‖r‖∞ = %{y:.3e}<extra></extra>",
    ), row=1, col=1)

    # ── Bottom panel — one trace per component (idx 1..n)
    palette = ["#7DD3FC", "#A78BFA", "#34D399", "#FBBF24", "#F87171",
               "#22D3EE", "#F472B6"]
    for i in range(n):
        colour = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=iters, y=comps[i], mode="lines+markers",
            name=f"x{i+1}",
            line=dict(color=colour, width=2),
            marker=dict(color=colour, size=6,
                        line=dict(color="#FFFFFF", width=1)),
            hovertemplate=f"Iter %{{x}}<br>x{i+1} = %{{y:.6f}}<extra></extra>",
        ), row=2, col=1)

    # ── Animated overlay traces  (idx n+1, n+2)  — current-iter markers
    fig.add_trace(go.Scatter(
        x=[iters[-1]], y=[residuals[-1]], mode="markers",
        name="current iter",
        marker=dict(color="#F87171", size=14, symbol="diamond",
                    line=dict(color="#FFFFFF", width=2)),
        hovertemplate="Iter %{x}<br>‖r‖∞ = %{y:.3e}<extra>cursor</extra>",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[iters[-1]] * n,
        y=[c[-1] for c in comps],
        mode="markers",
        name="current iter",
        marker=dict(color="#F87171", size=12, symbol="diamond",
                    line=dict(color="#FFFFFF", width=1.5)),
        hovertemplate="Iter %{x}<br>x = %{y:.6f}<extra>cursor</extra>",
        showlegend=False,
    ), row=2, col=1)

    cursor_idx_top = n + 1   # index of top cursor trace
    cursor_idx_bot = n + 2   # index of bottom cursor trace

    # Build frames — only the two cursor traces change per frame
    frames, slider_steps = [], []
    for k_idx, k in enumerate(iters):
        new_cursor_top = go.Scatter(
            x=[k], y=[residuals[k - 1]], mode="markers",
            marker=dict(color="#F87171", size=14, symbol="diamond",
                        line=dict(color="#FFFFFF", width=2)),
            hovertemplate="Iter %{x}<br>‖r‖∞ = %{y:.3e}<extra>cursor</extra>",
            showlegend=False,
        )
        new_cursor_bot = go.Scatter(
            x=[k] * n, y=[c[k - 1] for c in comps], mode="markers",
            marker=dict(color="#F87171", size=12, symbol="diamond",
                        line=dict(color="#FFFFFF", width=1.5)),
            hovertemplate="Iter %{x}<br>x = %{y:.6f}<extra>cursor</extra>",
            showlegend=False,
        )
        frames.append(go.Frame(
            data=[new_cursor_top, new_cursor_bot],
            name=str(k),
            traces=[cursor_idx_top, cursor_idx_bot],
        ))
        slider_steps.append(dict(
            method="animate", label=str(k),
            args=[[str(k)], dict(frame=dict(duration=500, redraw=True),
                                  mode="immediate",
                                  transition=dict(duration=200))],
        ))

    fig.frames = frames

    fig.update_xaxes(title_text="Iteration", row=2, col=1,
                     gridcolor="#1F2D4A")
    fig.update_xaxes(showticklabels=False, row=1, col=1,
                     gridcolor="#1F2D4A")
    fig.update_yaxes(type="log", title_text="‖r‖∞", row=1, col=1,
                     gridcolor="#1F2D4A")
    fig.update_yaxes(title_text="component value", row=2, col=1,
                     gridcolor="#1F2D4A")

    fig.update_layout(
        title=dict(text="<b>Gauss-Seidel Convergence</b>",
                   font=dict(size=15)),
        height=560,
        paper_bgcolor="#0F1A2E",
        plot_bgcolor="#0F1A2E",
        font=dict(family="Inter, sans-serif", size=12, color="#E6EDF7"),
        legend=dict(bgcolor="#131F38", bordercolor="#1F2D4A", borderwidth=1,
                    orientation="h", x=0.5, xanchor="center", y=-0.18),
        hovermode="x unified",
        updatemenus=[_ANIM_BTN],
        sliders=[{**_ANIM_SLIDER_BASE, "steps": slider_steps,
                  "active": len(slider_steps) - 1}],
        margin=dict(l=60, r=30, t=80, b=110),
        hoverlabel=dict(bgcolor="#131F38", bordercolor="#1F2D4A",
                        font_size=12, font_color="#E6EDF7"),
    )
    # Style subplot titles to match palette
    for ann in fig["layout"]["annotations"]:
        if "text" in ann and ann["text"].startswith(("Residual", "Per-component")):
            ann["font"] = dict(color="#A8B5CC", size=12,
                               family="Inter, sans-serif")
    return fig


def _fig_interpolation_enhanced(
    xs:        list,
    ys:        list,
    t:         float,
    result:    float,
    interp_fn: Callable[[float], float],
    title:     str,
    *,
    kind:      str = "newton_forward",
    x0_idx:    int | None = None,
) -> go.Figure:
    """Interpolation chart with hover-aware tooltips and method-specific marks.

    Parameters
    ----------
    kind   : 'newton_forward' | 'stirling' | 'lagrange'
    x0_idx : index of the x-origin for Newton-Forward (always 0) or
             Stirling (the data point closest to *t*).  Used to draw a
             gold ring around the origin point.
    """
    margin   = (max(xs) - min(xs)) * 0.12 + 0.5
    x_curve  = np.linspace(min(xs) - margin, max(xs) + margin, 500)
    y_curve  = []
    for xv in x_curve:
        try:
            y_curve.append(float(interp_fn(float(xv))))
        except Exception:
            y_curve.append(float("nan"))

    fig = go.Figure()

    # Interpolant curve
    fig.add_trace(go.Scatter(
        x=x_curve, y=y_curve, mode="lines",
        name="Interpolant",
        line=dict(color="#7DD3FC", width=2.5),
        hovertemplate="x = %{x:.4f}<br>f(x) = %{y:.4f}<extra></extra>",
    ))

    # Data points (hover shows index too)
    point_text = [f"({xv:g}, {yv:g})  ·  i = {i}"
                  for i, (xv, yv) in enumerate(zip(xs, ys))]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        name="Data points",
        marker=dict(color="#34D399", size=11,
                    line=dict(color="#FFFFFF", width=1.5)),
        text=point_text,
        hovertemplate="%{text}<extra>data</extra>",
    ))

    # x₀ origin highlight (Newton-Forward / Stirling)
    if x0_idx is not None and 0 <= x0_idx < len(xs):
        fig.add_trace(go.Scatter(
            x=[xs[x0_idx]], y=[ys[x0_idx]], mode="markers",
            name=f"x₀ = {xs[x0_idx]:g}",
            marker=dict(color="rgba(0,0,0,0)", size=22, symbol="circle",
                        line=dict(color="#FBBF24", width=2.5)),
            hovertemplate=f"x₀ = {xs[x0_idx]:g}<br>y₀ = {ys[x0_idx]:g}<extra>origin</extra>",
        ))

    # Lagrange basis polynomials  (toggleable via legend click; hidden by default)
    if kind == "lagrange":
        x_arr = np.asarray(xs, dtype=float)
        n     = len(xs)
        palette = ["#7DD3FC", "#A78BFA", "#34D399", "#FBBF24",
                   "#F87171", "#22D3EE", "#F472B6", "#FB923C"]
        for i in range(n):
            mask = np.arange(n) != i
            den  = np.prod(x_arr[i] - x_arr[mask])
            # Vectorised L_i(x):  ∏_{j≠i} (x - x_j) / (x_i - x_j)
            num  = np.ones_like(x_curve)
            for j in range(n):
                if j == i:
                    continue
                num *= (x_curve - x_arr[j])
            Li = num / den
            fig.add_trace(go.Scatter(
                x=x_curve, y=Li, mode="lines",
                name=f"L<sub>{i}</sub>(x)",
                line=dict(color=palette[i % len(palette)],
                          width=1.2, dash="dot"),
                opacity=0.55,
                visible="legendonly",       # hidden until the user clicks
                hovertemplate=f"L<sub>{i}</sub>(%{{x:.4f}}) = %{{y:.4f}}<extra></extra>",
            ))

    # Target marker — the interpolated point f(t)
    fig.add_trace(go.Scatter(
        x=[t], y=[result], mode="markers+text",
        name=f"f({t}) ≈ {result:.4f}",
        marker=dict(color="#F87171", size=18, symbol="star",
                    line=dict(color="#FFFFFF", width=1.2)),
        text=[f"  f({t}) ≈ {result:.4f}"],
        textposition="top right",
        textfont=dict(color="#F87171", size=11,
                      family="JetBrains Mono, monospace"),
        hovertemplate=f"f({t}) ≈ {result:.8f}<extra>target</extra>",
    ))

    # Vertical guide at the target x
    fig.add_vline(x=t, line=dict(color="#F87171", width=1, dash="dash"),
                  opacity=0.5)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="x",
        yaxis_title="f(x)",
        hovermode="x unified",
        legend=dict(bgcolor="#131F38", bordercolor="#1F2D4A",
                    borderwidth=1, orientation="v",
                    x=1.02, xanchor="left", y=1.0),
        **{k: v for k, v in _PLOTLY_LAYOUT.items() if k != "legend"},
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  STEP-BY-STEP HTML BUILDER  (coloured, monospaced)
# ══════════════════════════════════════════════════════════════════════════════
def _steps_html(lines: list[tuple[str, str]]) -> str:
    """Build a coloured monospaced block from (text, tag) pairs."""
    body = ""
    for text, tag in lines:
        escaped = (text
                   .replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;"))
        if tag:
            body += f'<span class="{tag}">{escaped}</span>\n'
        else:
            body += escaped + "\n"
    return f'<div class="calc-steps">{body}</div>'


def _div(width: int = 60) -> tuple[str, str]:
    return ("─" * width, "dim")


def _hdr(title: str) -> list[tuple[str, str]]:
    return [_div(), (f"  {title}", "hdr"), _div()]


def _bmatrix_latex(M: np.ndarray, prec: int = 4) -> str:
    """Render a NumPy array as a LaTeX ``bmatrix`` string (no ``$$`` delimiters).

    1-D arrays are rendered as a column vector.  Trailing/leading sign and
    fixed precision keep matrices visually aligned in Streamlit's KaTeX output.
    """
    arr = np.asarray(M, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    rows = [
        " & ".join(f"{v:.{prec}f}" for v in row)
        for row in arr
    ]
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


# ── Gauss-Seidel LaTeX helpers ──────────────────────────────────────────────
def _fmt_num(v: float, prec: int = 4) -> str:
    """Pretty-print a number: integer if integral, else trimmed decimals."""
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    s = f"{v:.{prec}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _gs_diag_frac(a_ii: float) -> str:
    r"""LaTeX  \frac{1}{a_ii}  rendered nicely."""
    return rf"\frac{{1}}{{{_fmt_num(a_ii)}}}"


def _gs_isolated_rhs(A_row: np.ndarray, b_val: float, i: int,
                     var_names: list[str]) -> str:
    """Symbolic RHS for the isolated form of row *i*  (e.g. ``-y + z``)."""
    n = len(A_row)
    parts: list[tuple[str, str]] = []

    if not np.isclose(b_val, 0.0):
        sign = "+" if b_val > 0 else "-"
        parts.append((sign, _fmt_num(abs(b_val))))

    for j in range(n):
        if j == i:
            continue
        coef = -A_row[j]                       # term moved to RHS
        if np.isclose(coef, 0.0):
            continue
        sign = "+" if coef > 0 else "-"
        mag  = abs(coef)
        body = (var_names[j] if np.isclose(mag, 1.0)
                else f"{_fmt_num(mag)}{var_names[j]}")
        parts.append((sign, body))

    if not parts:
        return "0"

    first_sign, first_body = parts[0]
    rhs = first_body if first_sign == "+" else f"-{first_body}"
    for sign, body in parts[1:]:
        rhs += f" {sign} {body}"
    return rhs


def _gs_substitution_rhs(A_row: np.ndarray, b_val: float, i: int,
                         x_subst: np.ndarray) -> tuple[str, float]:
    """Numeric RHS with values plugged in.  Returns ``(latex_rhs, value)``."""
    n = len(A_row)
    parts: list[tuple[str, str]] = []
    accum = 0.0

    if not np.isclose(b_val, 0.0):
        sign = "+" if b_val > 0 else "-"
        parts.append((sign, _fmt_num(abs(b_val))))
        accum += b_val

    for j in range(n):
        if j == i:
            continue
        coef = -A_row[j]
        if np.isclose(coef, 0.0):
            continue
        val   = float(x_subst[j])
        accum += coef * val
        sign  = "+" if coef > 0 else "-"
        mag   = abs(coef)
        val_s = f"({val:.4f})"
        body  = (val_s if np.isclose(mag, 1.0)
                 else rf"{_fmt_num(mag)}\cdot {val_s}")
        parts.append((sign, body))

    if not parts:
        return "0", 0.0

    first_sign, first_body = parts[0]
    rhs = first_body if first_sign == "+" else f"-{first_body}"
    for sign, body in parts[1:]:
        rhs += f" {sign} {body}"
    return rhs, accum


# ── Newton-Raphson LaTeX helpers ────────────────────────────────────────────
def _py_expr_to_latex(expr: str, sub_val: float | None = None) -> str:
    """Render a Python expression string as LaTeX, optionally substituting *x*.

    Handles the common subset used by this app's safe evaluator
    (``+ - * / ** ()`` and the function names exposed in ``_SAFE_NS``).
    When ``sub_val`` is given, every standalone ``x`` is replaced by
    ``({sub_val:.5f})`` so the rendered LaTeX shows explicit substitution
    (e.g.  ``(1.90476)^{4} - 11\\cdot(1.90476) + 8``).

    The output is intentionally minimal — sufficient for displaying
    polynomial / elementary expressions inside ``st.latex`` without
    requiring sympy.
    """
    s = expr.strip()
    if sub_val is not None:
        # Format: drop trailing zeros if integral, else 5 decimals
        if abs(sub_val - round(sub_val)) < 1e-12:
            val_str = f"{int(round(sub_val))}"
        else:
            val_str = f"{sub_val:.5f}"
        # Replace standalone x (not part of a longer identifier like "exp")
        s = re.sub(r'(?<![A-Za-z_0-9])x(?![A-Za-z_0-9])', f'({val_str})', s)
    # ** with parenthesised exponent  →  ^{...}
    s = re.sub(r'\*\*\(([^()]+)\)', r'^{\1}', s)
    # ** with bare numeric / identifier exponent  →  ^{n}
    s = re.sub(r'\*\*([A-Za-z0-9_.]+)', r'^{\1}', s)
    # Implicit multiplication: number*( → number(  (lecture-style)
    s = re.sub(r'(\d)\s*\*\s*\(', r'\1(', s)
    # Remaining * → \cdot
    s = s.replace('*', r' \cdot ')
    # Tighten whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  MATRIX HELPERS  (DataFrame ↔ NumPy)
# ══════════════════════════════════════════════════════════════════════════════
def _zero_df(n: int) -> pd.DataFrame:
    cols = [f"x{i+1}" for i in range(n)]
    return pd.DataFrame(np.zeros((n, n)), columns=cols, dtype=float)


def _prefill_A(data: list[list[float]], n: int) -> pd.DataFrame:
    df = _zero_df(n)
    for i in range(min(len(data), n)):
        for j in range(min(len(data[i]), n)):
            df.iat[i, j] = data[i][j]
    return df


def _prefill_b(data: list[float], n: int) -> pd.DataFrame:
    col = pd.DataFrame(
        {"b": [float(data[i]) if i < len(data) else 0.0 for i in range(n)]}
    )
    return col


def _df_to_array(df: pd.DataFrame) -> np.ndarray:
    return df.to_numpy(dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
#  FORMATTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _fmt_matrix(mat: np.ndarray, prec: int = 4) -> str:
    if mat.ndim == 1:
        inner = "  ".join(f"{v:>12.{prec}f}" for v in mat)
        return f"[ {inner} ]"
    rows = ["  ".join(f"{v:>12.{prec}f}" for v in row) for row in mat]
    w   = max(len(r) for r in rows)
    top = "┌" + " " * (w + 2) + "┐"
    mid = [f"│ {r} │" for r in rows]
    bot = "└" + " " * (w + 2) + "┘"
    return "\n".join([top] + mid + [bot])


def _iter_df_nr(log: list) -> pd.DataFrame:
    return pd.DataFrame(
        log, columns=["Iter", "xₙ", "f(xₙ)", "f′(xₙ)", "xₙ₊₁", "Error"]
    )


def _iter_df_fp(log: list) -> pd.DataFrame:
    return pd.DataFrame(
        log, columns=["Iter", "a", "b", "c", "f(c)", "|f(c)|"]
    )


def _diff_df(diff: list[list[float]]) -> pd.DataFrame:
    max_cols = max(len(col) for col in diff)
    records = {}
    for k, col in enumerate(diff):
        padded = col + [None] * (max_cols - len(col))
        records[f"Δ^{k}y"] = padded
    return pd.DataFrame(records)


def _fwd_diff_num_str(v: float) -> str:
    """Format a forward-difference value: integer when exact, else 4dp trimmed."""
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _forward_diff_staircase_html(
    xs:   list[float],
    ys:   list[float],
    diff: list[list[float]],
) -> str:
    """Render the forward-difference table in lecture *staircase* form.

    Layout
    ------
    The table has ``2n − 1`` rows (n = len(xs)).  Even-indexed rows host
    the (x, y) data; odd-indexed rows host the differences offset down
    by one.  The k-th column (Δᵏy) places its j-th entry at row ``k + 2j``,
    producing the characteristic staircase / triangular pattern from the
    HUE lecture slide.

    Highlighting
    ------------
    The *forward diagonal* — i.e. ``y₀, Δy₀, Δ²y₀, …`` — is the column
    of values that actually feed into Newton's forward formula.  These
    cells are tinted with an accent background and bold weight so the
    student can see at a glance which numbers contribute to the
    interpolation polynomial.
    """
    n = len(xs)
    max_order = len(diff) - 1
    # column 0 = x, column 1 = y, columns 2..max_order+1 = Δ¹..Δ^max_order
    headers: list[str] = ["x", "y"]
    for k in range(1, max_order + 1):
        sup = "" if k == 1 else f"<sup>{k}</sup>"
        headers.append(f"Δ{sup}y")

    n_cols     = len(headers)
    total_rows = 2 * n - 1
    hi_bg      = "#1976D255"   # translucent blue for the diagonal
    hi_bd      = "2px solid #4FC3F7"

    rows_html: list[str] = []
    for r in range(total_rows):
        cells: list[str] = []
        for c in range(n_cols):
            value     = ""
            highlight = False
            if c == 0:                        # x column
                if r % 2 == 0:
                    value = _fwd_diff_num_str(xs[r // 2])
            elif c == 1:                      # y column
                if r % 2 == 0:
                    j     = r // 2
                    value = _fwd_diff_num_str(ys[j])
                    if j == 0:
                        highlight = True      # y₀
            else:                             # Δᵏy column  (k = c − 1)
                k = c - 1
                if r >= k and (r - k) % 2 == 0:
                    j = (r - k) // 2
                    if k <= max_order and j < len(diff[k]):
                        value = _fwd_diff_num_str(diff[k][j])
                        if j == 0:
                            highlight = True  # Δᵏy₀
            style_parts = [
                "padding:10px 18px",
                "text-align:center",
                "border:1px solid #2B2B2B",
                "min-width:70px",
                "font-family:'Fira Code',monospace",
                "font-size:0.95rem",
            ]
            if highlight:
                style_parts.extend([
                    f"background:{hi_bg}",
                    f"border:{hi_bd}",
                    "color:#FFFFFF",
                    "font-weight:700",
                ])
            else:
                style_parts.append("color:#D4D4D4")
            cells.append(f'<td style="{";".join(style_parts)}">{value}</td>')
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    head_cells = "".join(
        f'<th style="padding:12px 18px;text-align:center;'
        f'background:#0E0E1A;color:#9CDCFE;border:1px solid #2B2B2B;'
        f'font-weight:700;letter-spacing:0.04em;">{h}</th>'
        for h in headers
    )

    legend = (
        '<div style="margin-top:10px;font-size:0.78rem;color:#A0A0A0;">'
        '<span style="display:inline-block;width:14px;height:14px;'
        f'background:{hi_bg};vertical-align:middle;margin-right:6px;'
        'border-radius:3px;border:1px solid #4FC3F7;"></span>'
        'Forward diagonal: <code>y₀, Δy₀, Δ²y₀, …</code> — '
        'the values used in Newton\'s forward formula.'
        '</div>'
    )

    return (
        '<div style="overflow-x:auto;margin:6px 0 4px 0;">'
        '<table style="border-collapse:collapse;margin:0 auto;'
        'background:#141420;border-radius:10px;overflow:hidden;">'
        f'<thead><tr>{head_cells}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table>'
        f'{legend}'
        '</div>'
    )


def _central_diff_staircase_html(
    xs:        list[float],
    ys:        list[float],
    diff:      list[list[float]],
    center_idx: int,
) -> str:
    """Render a *central* difference table in lecture staircase form.

    Layout
    ------
    Same staircase grid as the forward-difference variant: ``2n − 1`` rows,
    where the *i*-th data point sits on row ``2 * i``.  The k-th column
    (Δᵏy) places its j-th entry at row ``k + 2 * j``.

    Highlighting
    ------------
    Two visual layers help the student trace Stirling's central path:

    1. **Central row** — the entire row of ``x₀ = xs[center_idx]`` is tinted
       with a soft gold background.  This makes the origin of the formula
       unmistakable.

    2. **Central path** — the specific Δᵏy values consumed by Stirling's
       formula are outlined with a coloured border:
         * Even *k*: a single value at row ``2·center_idx``  (Δᵏy_{-k/2}).
         * Odd  *k*: two values at rows ``2·center_idx ± 1``
           (Δᵏy_{-(k+1)/2} and Δᵏy_{-(k-1)/2}), which the formula averages.

    The extra column index ``c0_y`` highlights the y-cell at the central row
    (i.e. ``y₀``) with the same path-border as the central differences.
    """
    n = len(xs)
    max_order = len(diff) - 1
    headers: list[str] = ["x", "y"]
    for k in range(1, max_order + 1):
        sup = "" if k == 1 else f"<sup>{k}</sup>"
        headers.append(f"Δ{sup}y")

    n_cols     = len(headers)
    total_rows = 2 * n - 1
    center_row = 2 * center_idx

    row_bg     = "#FFC10733"   # translucent amber for the x₀ row
    path_bd    = "2px solid #FFD54F"
    path_bg    = "#FFC10755"
    path_text  = "#FFFFFF"

    rows_html: list[str] = []
    for r in range(total_rows):
        is_center_row = (r == center_row)
        cells: list[str] = []
        for c in range(n_cols):
            value     = ""
            on_path   = False     # cell is part of Stirling's central path
            if c == 0:                        # x column
                if r % 2 == 0:
                    value = _fwd_diff_num_str(xs[r // 2])
            elif c == 1:                      # y column
                if r % 2 == 0:
                    j     = r // 2
                    value = _fwd_diff_num_str(ys[j])
                    if j == center_idx:
                        on_path = True        # y₀
            else:                             # Δᵏy column  (k = c − 1)
                k = c - 1
                if r >= k and (r - k) % 2 == 0:
                    j = (r - k) // 2
                    if k <= max_order and j < len(diff[k]):
                        value = _fwd_diff_num_str(diff[k][j])
                        # Stirling's central j for column k
                        if k % 2 == 0:
                            target_js = {center_idx - k // 2}
                        else:
                            target_js = {
                                center_idx - (k + 1) // 2,
                                center_idx - (k - 1) // 2,
                            }
                        if j in target_js and 0 <= j < len(diff[k]):
                            on_path = True

            style_parts = [
                "padding:10px 18px",
                "text-align:center",
                "border:1px solid #2B2B2B",
                "min-width:70px",
                "font-family:'Fira Code',monospace",
                "font-size:0.95rem",
                "color:#D4D4D4",
            ]
            # Layer 1: x₀ row tint (applied first so path overrides)
            if is_center_row:
                style_parts.append(f"background:{row_bg}")
            # Layer 2: central path overrides background + adds bold border
            if on_path:
                style_parts.extend([
                    f"background:{path_bg}",
                    f"border:{path_bd}",
                    f"color:{path_text}",
                    "font-weight:700",
                ])
            cells.append(f'<td style="{";".join(style_parts)}">{value}</td>')
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    head_cells = "".join(
        f'<th style="padding:12px 18px;text-align:center;'
        f'background:#0E0E1A;color:#9CDCFE;border:1px solid #2B2B2B;'
        f'font-weight:700;letter-spacing:0.04em;">{h}</th>'
        for h in headers
    )

    legend = (
        '<div style="margin-top:10px;font-size:0.78rem;color:#A0A0A0;'
        'line-height:1.7;">'
        '<span style="display:inline-block;width:14px;height:14px;'
        f'background:{row_bg};vertical-align:middle;margin-right:6px;'
        'border-radius:3px;border:1px solid #FFD54F66;"></span>'
        f'Central row: $x_0 = {_fwd_diff_num_str(xs[center_idx])}$'
        '&nbsp;&nbsp;&nbsp;'
        '<span style="display:inline-block;width:14px;height:14px;'
        f'background:{path_bg};border:2px solid #FFD54F;'
        'vertical-align:middle;margin-right:6px;border-radius:3px;"></span>'
        'Central path: $y_0,\\;\\Delta y_0,\\;\\Delta y_{-1},\\;'
        '\\Delta^{2}y_{-1},\\;\\Delta^{3}y_{-1},\\;\\Delta^{3}y_{-2},\\;'
        '\\Delta^{4}y_{-2},\\;\\ldots$'
        '</div>'
    )

    return (
        '<div style="overflow-x:auto;margin:6px 0 4px 0;">'
        '<table style="border-collapse:collapse;margin:0 auto;'
        'background:#141420;border-radius:10px;overflow:hidden;">'
        f'<thead><tr>{head_cells}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table>'
        f'{legend}'
        '</div>'
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════
def _init_state() -> None:
    defaults: dict = {
        # shared
        "method":     METHODS[0],
        "result":     None,   # stores last computation payload
        # matrix
        "matrix_n":   3,
        # gauss-seidel
        "gs_tol":     1e-6,
        "gs_max":     1000,
        # false position
        "fp_eq":      "exp(x) - 3*x**2",
        "fp_a":       0.5,
        "fp_b":       1.0,
        "fp_tol":     0.00001,
        "fp_max":     100,
        # newton-raphson
        "nr_f":       "x**4 - 11*x + 8",
        "nr_fp":      "4*x**3 - 11",
        "nr_x0":      2.0,
        "nr_tol":     0.00001,
        # interpolation — Newton's Forward & Lagrange (shared keys)
        "interp_xs":  "1990, 1993, 1996, 1999, 2002",
        "interp_ys":  "120, 100, 111, 108, 99",
        "interp_t":   1991.0,
        # interpolation — Stirling (own keys, lecture defaults)
        "stir_xs":    "20, 25, 30, 35, 40",
        "stir_ys":    "48234, 47354, 46267, 44978, 43389",
        "stir_t":     28.0,
        # interpolation — Lagrange (own keys, HUE lecture defaults)
        "lag_xs":     "5, 6, 9, 11",
        "lag_ys":     "380, -2, 196, 508",
        "lag_t":      10.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap-icon glyph for each method (used by streamlit-option-menu).
# See https://icons.getbootstrap.com/ for the full library.
_METHOD_ICONS: dict[str, str] = {
    METHODS[0]: "grid-3x3",          # LU decomposition  —  matrix grid
    METHODS[1]: "arrow-clockwise",   # Gauss-Seidel       —  iteration
    METHODS[2]: "rulers",            # False Position     —  bracketing
    METHODS[3]: "graph-up",          # Newton-Raphson     —  tangent line
    METHODS[4]: "table",             # Newton's Forward   —  difference table
    METHODS[5]: "bullseye",          # Stirling           —  central point
    METHODS[6]: "diagram-2",         # Lagrange           —  basis polynomials
}

# Short, navigation-friendly labels for the sidebar option-menu.
# The full canonical method names (used by METHOD_META and the calculation
# branches) remain unchanged in the METHODS list above.
_METHOD_SHORT: dict[str, str] = {
    METHODS[0]: "Doolittle's LU",
    METHODS[1]: "Gauss-Seidel",
    METHODS[2]: "False Position",
    METHODS[3]: "Newton-Raphson",
    METHODS[4]: "Newton's Forward",
    METHODS[5]: "Stirling",
    METHODS[6]: "Lagrange",
}
_SHORT_TO_METHOD: dict[str, str] = {v: k for k, v in _METHOD_SHORT.items()}


with st.sidebar:
    # ── Brand / app identity ------------------------------------------------
    st.markdown(
        """
        <div class='anc-brand'>
          <div class='anc-logo'>∑</div>
          <span class='anc-name'>Advanced Numerical Calculator</span>
          <span class='anc-tag'>Numerical Methods Suite</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Method selector -----------------------------------------------------
    st.markdown('<div class="section-label">Numerical Method</div>',
                unsafe_allow_html=True)

    if _HAS_OPTION_MENU:
        _short_options = [_METHOD_SHORT[m] for m in METHODS]
        _icon_list     = [_METHOD_ICONS[m] for m in METHODS]
        _current_short = _METHOD_SHORT[st.session_state["method"]]
        _selected_short = option_menu(
            menu_title=None,
            options=_short_options,
            icons=_icon_list,
            default_index=_short_options.index(_current_short),
            orientation="vertical",
            key="method_menu",
            styles={
                "container": {
                    "padding": "4px 0",
                    "background": "transparent",
                },
                "icon": {
                    "color": "#7DD3FC",
                    "font-size": "1.0rem",
                },
                "nav-link": {
                    "font-family": "Inter, sans-serif",
                    "font-size": "0.9rem",
                    "font-weight": "500",
                    "color": "#E6EDF7",
                    "padding": "10px 14px",
                    "margin": "3px 6px",
                    "border-radius": "10px",
                    "--hover-color": "#172339",
                    "transition": "background 0.18s, color 0.18s",
                },
                "nav-link-selected": {
                    "background": "linear-gradient(135deg, rgba(56,189,248,0.18), rgba(167,139,250,0.18))",
                    "color": "#7DD3FC",
                    "font-weight": "600",
                    "border": "1px solid rgba(125,211,252,0.35)",
                },
            },
        )
        method = _SHORT_TO_METHOD[_selected_short]
        st.session_state["method"] = method
    else:
        # Fallback: vanilla selectbox (still session-state aware).
        # Use a separate widget key to avoid StreamlitAPIException when
        # st.session_state["method"] is also written programmatically.
        _sel = st.selectbox(
            label="Method",
            options=METHODS,
            index=METHODS.index(st.session_state["method"]),
            label_visibility="collapsed",
            key="method_select_widget",
        )
        method = _sel
        st.session_state["method"] = method

    # ── Method info card ---------------------------------------------------
    pill_cls, complexity, formula_latex, desc = METHOD_META[method]
    if "Gauss" in method:
        st.markdown(
            f"""
            <div class='method-info-card'>
              <p class='mic-desc' style='margin:0;'>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class='method-info-card'>
              <span class='method-pill {pill_cls}'>{pill_cls.replace('pill-', '').replace('-', ' ').title()}</span>
              <span class='mic-meta'>complexity · {complexity}</span>
              <p class='mic-desc'>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    # Hide the formula card for Gauss-Seidel (matches lecture-style layout).
    if "Gauss" not in method:
        st.latex(formula_latex)

    st.markdown('<hr style="margin:18px 0 6px 0;">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    #  PER-METHOD INPUT CONTROLS
    # ══════════════════════════════════════════════════════════════
    if "Doolittle" in method or "Gauss" in method:
        st.markdown('<div class="section-label">Matrix Size</div>',
                    unsafe_allow_html=True)
        n = st.slider("n×n", min_value=2, max_value=5,
                      value=st.session_state["matrix_n"],
                      label_visibility="collapsed",
                      key="matrix_n")

    elif "False" in method:
        st.markdown('<div class="section-label">Function</div>',
                    unsafe_allow_html=True)
        st.write("Enter $f(x)$:")
        fp_eq = st.text_input("f(x)", value=st.session_state["fp_eq"],
                              label_visibility="collapsed", key="fp_eq")

        st.markdown('<div class="section-label">Bracket Interval</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fp_a = st.number_input("$a$", value=st.session_state["fp_a"],
                                   key="fp_a", format="%.4f")
        with c2:
            fp_b = st.number_input("$b$", value=st.session_state["fp_b"],
                                   key="fp_b", format="%.4f")

        st.markdown('<div class="section-label">Precision</div>',
                    unsafe_allow_html=True)
        fp_tol = st.number_input(
            "Tolerance  $|x_n - x_{n-1}| <$",
            value=st.session_state["fp_tol"],
            min_value=1e-15,
            max_value=1e-1,
            step=1e-5,
            format="%.5f",
            help="Stopping criterion on the absolute step size. "
                 "Default 0.00001 ≈ 5 decimal places of precision.",
            key="fp_tol",
        )

    elif "Newton-Raph" in method:
        st.markdown('<div class="section-label">Equations</div>',
                    unsafe_allow_html=True)
        st.write("Enter $f(x)$:")
        nr_f  = st.text_input("f(x)", value=st.session_state["nr_f"],
                               label_visibility="collapsed", key="nr_f")
        st.write("Enter $f'(x)$:")
        nr_fp = st.text_input("f'(x)", value=st.session_state["nr_fp"],
                               label_visibility="collapsed", key="nr_fp")

        st.markdown('<div class="section-label">Initial Conditions</div>',
                    unsafe_allow_html=True)
        nr_x0  = st.number_input("Initial guess $x_0$",
                                  value=st.session_state["nr_x0"],
                                  key="nr_x0", format="%.4f")
        nr_tol = st.number_input(
            "Tolerance  $|x_n - x_{n-1}| <$",
            value=st.session_state["nr_tol"],
            min_value=1e-15,
            max_value=1e-1,
            step=1e-5,
            format="%.5f",
            help="Stopping criterion on the absolute step size.  "
                 "Default 0.00001 ≈ 5 decimal places of precision.",
            key="nr_tol",
        )

    else:  # interpolation family
        # Stirling and Lagrange each have their own session-state keys so
        # their lecture defaults don't collide with Newton's Forward.
        if "Stirling" in method:
            kx, ky, kt = "stir_xs",   "stir_ys",   "stir_t"
        elif "Lagrange" in method:
            kx, ky, kt = "lag_xs",    "lag_ys",    "lag_t"
        else:
            kx, ky, kt = "interp_xs", "interp_ys", "interp_t"

        st.markdown('<div class="section-label">Dataset</div>',
                    unsafe_allow_html=True)
        st.write("$X$ values (comma-separated):")
        st.text_input("X", value=st.session_state[kx],
                       label_visibility="collapsed", key=kx)
        st.write("$Y = f(X)$ values (comma-separated):")
        st.text_input("Y", value=st.session_state[ky],
                       label_visibility="collapsed", key=ky)

        st.markdown('<div class="section-label">Target Point</div>',
                    unsafe_allow_html=True)
        st.write("Interpolate at $x =$")
        st.number_input("x", value=st.session_state[kt],
                         label_visibility="collapsed",
                         key=kt, format="%.4f")

    # ── Calculate button
    st.markdown("<br>", unsafe_allow_html=True)
    try:
        calculate = st.button("▶  Calculate", type="primary", use_container_width=True)
    except TypeError:
        calculate = st.button("▶  Calculate", type="primary")

    st.markdown(
        '<div class="anc-footer">v2.1 · Streamlit Edition</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA — page title + method header
# ══════════════════════════════════════════════════════════════════════════════
col_title, col_method = st.columns([2, 3])
with col_title:
    st.markdown(
        "<h1 class='anc-title'>Advanced Numerical Calculator</h1>"
        "<p class='anc-subtitle'>Numerical Methods · Interactive Web Dashboard</p>",
        unsafe_allow_html=True,
    )
with col_method:
    pill_cls, complexity, _, _ = METHOD_META[method]
    st.markdown(
        f"""
        <div class='method-header-card'>
          <span class='mhc-name'>{method}</span>
          <span class='method-pill {pill_cls}' style='float:right; margin-top:1px;'>
            {complexity}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  MATRIX INPUT AREA
# ══════════════════════════════════════════════════════════════════════════════
A_arr = B_arr = X0_arr = None

if "Doolittle" in method or "Gauss" in method:
    n = st.session_state["matrix_n"]

    # ── Pre-fill data based on method
    if "Doolittle" in method:
        default_A  = [[2, -6, 8], [5, 4, -3], [3, 1, 2]]
        default_B  = [24, 2, 16]
        default_X0 = None
    else:
        default_A  = [[3, 1, -1], [1, 2, 1], [1, -1, 4]]
        default_B  = [0, 0, 3]
        default_X0 = [0, 0, 0]

    st.markdown('<div class="section-label">Matrix A  ·  Coefficient Matrix</div>',
                unsafe_allow_html=True)

    df_A_default = _prefill_A(default_A, n)
    edited_A = st.data_editor(
        df_A_default,
        use_container_width=True,
        hide_index=False,
        num_rows="fixed",
        key=f"editor_A_{method}_{n}",
    )

    cols_bx0 = st.columns(2 if "Gauss" in method else 1)
    with cols_bx0[0]:
        st.markdown('<div class="section-label">Vector b  ·  Constants</div>',
                    unsafe_allow_html=True)
        df_B_default = _prefill_b(default_B, n)
        edited_B = st.data_editor(
            df_B_default,
            use_container_width=True,
            hide_index=False,
            num_rows="fixed",
            key=f"editor_B_{method}_{n}",
        )

    if "Gauss" in method and default_X0 is not None:
        with cols_bx0[1]:
            st.markdown('<div class="section-label">Initial Guess  x₀</div>',
                        unsafe_allow_html=True)
            df_X0_default = _prefill_b(default_X0, n)
            df_X0_default.columns = ["x₀"]
            edited_X0 = st.data_editor(
                df_X0_default,
                use_container_width=True,
                hide_index=False,
                num_rows="fixed",
                key=f"editor_X0_{method}_{n}",
            )
            X0_arr = edited_X0["x₀"].to_numpy(dtype=float)

    A_arr = _df_to_array(edited_A)
    B_arr = edited_B["b"].to_numpy(dtype=float)

    st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  CALCULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
result_payload: dict | None = None

if calculate:
    with st.spinner("🔄  Calculating…"):
        try:
            # ─────────────── Doolittle ────────────────────────
            if "Doolittle" in method:
                L, U, x = calculator.doolittle_lu_decomposition(A_arr, B_arr)
                # y comes from the same factorisation: L y = b
                y = np.linalg.solve(L, B_arr)

                result_payload = dict(
                    kind="doolittle",
                    A=A_arr.copy(), b=B_arr.copy(),
                    L=L, U=U, y=y, x=x,
                    fig=_fig_lu_heatmap(L, U),
                    n=len(x),
                )

            # ─────────────── Gauss-Seidel ────────────────────────────────
            elif "Gauss" in method:
                # Hardcoded university-style settings: exactly 6 iterations,
                # no early termination via tolerance.
                GS_ITERS = 6
                GS_TOL   = 0.0
                x0_used  = (X0_arr if X0_arr is not None
                            else np.zeros(len(B_arr))).astype(float)
                sol, iters, converged, history = calculator.gauss_seidel(
                    A_arr, B_arr, x0_used, GS_TOL, GS_ITERS)
                sdd = calculator.is_strictly_diagonally_dominant(A_arr)
                n   = len(B_arr)

                # Residuals derived directly from the iteration history
                residuals = [float(np.linalg.norm(A_arr @ h - B_arr, np.inf))
                             for h in history]
                res_fin   = (residuals[-1] if residuals
                             else float(np.linalg.norm(A_arr @ sol - B_arr, np.inf)))

                result_payload = dict(
                    kind="gauss_seidel",
                    A=A_arr.copy(), b=B_arr.copy(), x0=x0_used.copy(),
                    sol=sol, iters=iters, converged=converged,
                    sdd=sdd, res_fin=res_fin,
                    history=[h.copy() for h in history],
                    fig=_fig_gauss_seidel_anim(
                        A_arr, B_arr, x0_used,
                        [h.copy() for h in history]),
                    n=n,
                )

            # ─────────────── False Position ──────────────────────────────
            elif "False" in method:
                eq  = str(st.session_state["fp_eq"])
                a   = float(st.session_state["fp_a"])
                b   = float(st.session_state["fp_b"])
                tol = float(st.session_state["fp_tol"])
                mx  = int(st.session_state["fp_max"])
                f   = _make_f(eq)

                fa0, fb0 = f(a), f(b)
                if fa0 * fb0 > 0:
                    raise ValueError(
                        f"f(a)·f(b) > 0 — no sign change on [{a}, {b}].  "
                        "Choose an interval that brackets a root.")

                root, iters, history = calculator.false_position(
                    f, a, b, tol, mx)

                margin = (b - a) * 0.35
                result_payload = dict(
                    kind="false_position",
                    root=root, iters=iters,
                    a0=a, b0=b, fa0=fa0, fb0=fb0,
                    tol=tol, history=history,
                    fig=_fig_false_position_anim(f, eq, history, root),
                    eq=eq,
                )

            # ─────────────── Newton-Raphson ───────────────────────────────
            elif "Newton-Raph" in method:
                eq    = str(st.session_state["nr_f"])
                fp_eq = str(st.session_state["nr_fp"])
                x0    = float(st.session_state["nr_x0"])
                tol   = float(st.session_state["nr_tol"])
                mx    = 100   # internal safety cap (no UI; lecture defaults converge in ≤4)
                f     = _make_f(eq)
                fp    = _make_f(fp_eq)

                root, iters, history = calculator.newton_raphson(
                    f, fp, x0, tol, mx)

                span = abs(root - x0) * 1.6 + 1.5
                result_payload = dict(
                    kind="newton_raphson",
                    root=root, iters=iters,
                    x0=x0, tol=tol, history=history,
                    fig=_fig_newton_raphson_anim(
                        f, fp, eq, fp_eq, history, root),
                    eq=eq, fp_eq=fp_eq,
                )

            # ─────────────── Interpolation family ────────────────────────
            else:
                # Stirling and Lagrange have their own session-state keys
                # (lecture defaults); Newton's Forward uses the legacy keys.
                if "Stirling" in method:
                    xs_raw = str(st.session_state["stir_xs"])
                    ys_raw = str(st.session_state["stir_ys"])
                    t      = float(st.session_state["stir_t"])
                elif "Lagrange" in method:
                    xs_raw = str(st.session_state["lag_xs"])
                    ys_raw = str(st.session_state["lag_ys"])
                    t      = float(st.session_state["lag_t"])
                else:
                    xs_raw = str(st.session_state["interp_xs"])
                    ys_raw = str(st.session_state["interp_ys"])
                    t      = float(st.session_state["interp_t"])
                xs = [float(v.strip()) for v in xs_raw.split(",")]
                ys = [float(v.strip()) for v in ys_raw.split(",")]

                if len(xs) != len(ys):
                    raise ValueError("X and Y arrays must have equal length.")

                if "Forward" in method:
                    result, diff = calculator.newton_forward_interpolation(xs, ys, t)
                    algo = "Newton's Forward Difference"
                    interp_fn = lambda xv, _xs=xs, _ys=ys: (
                        calculator.newton_forward_interpolation(_xs, _ys, xv)[0])
                    interp_kind = "newton_forward"
                    interp_x0_idx = 0                       # x₀ = first table point
                elif "Stirling" in method:
                    result, diff = calculator.stirling_interpolation(xs, ys, t)
                    algo = "Stirling's Central Difference"
                    interp_fn = lambda xv, _xs=xs, _ys=ys: (
                        calculator.stirling_interpolation(_xs, _ys, xv)[0])
                    interp_kind = "stirling"
                    interp_x0_idx = int(np.argmin(
                        [abs(xv - t) for xv in xs]))        # central node
                else:
                    result    = calculator.lagrange_interpolation(xs, ys, t)
                    diff      = None
                    algo      = "Lagrange's Polynomial"
                    interp_fn = lambda xv, _xs=xs, _ys=ys: (
                        calculator.lagrange_interpolation(_xs, _ys, xv))
                    interp_kind = "lagrange"
                    interp_x0_idx = None

                rows = []
                rows += _hdr(algo) + [("", "")]
                rows += [
                    (f"  X data   :  {xs}", "key"),
                    (f"  Y data   :  {ys}", "key"),
                    (f"  Target x :  {t}", "num"),
                    ("", ""),
                ]
                rows += _hdr("Result")
                rows += [(f"  f({t})  ≈  {result:.10f}", "val")]

                result_payload = dict(
                    kind="interpolation",
                    result=result, algo=algo, t=t, diff=diff,
                    steps_html=_steps_html(rows),
                    fig=_fig_interpolation_enhanced(
                        xs, ys, t, result, interp_fn,
                        f"{algo}  —  f({t}) ≈ {result:.4f}",
                        kind=interp_kind, x0_idx=interp_x0_idx),
                    xs=xs, ys=ys,
                )

            st.session_state["result"] = result_payload

        except Exception as exc:
            st.session_state["result"] = dict(
                kind="error",
                message=str(exc),
                traceback=traceback.format_exc(),
            )

# ── Auto-close sidebar on mobile after Calculate ─────────────────────────────
if calculate:
    _stc.html(
        """
        <script>
        (function() {
            try {
                var pw = window.parent || window;
                if (pw.innerWidth <= 768) {
                    var closeBtn = pw.document.querySelector(
                        '[data-testid="stSidebarCollapseButton"] button'
                    );
                    if (closeBtn) { closeBtn.click(); }
                    setTimeout(function() {
                        var main = pw.document.querySelector('[data-testid="stMain"]');
                        if (main) main.scrollTo({top: 0, behavior: 'smooth'});
                    }, 400);
                }
            } catch(e) {}
        })();
        </script>
        """,
        height=0,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT AREA — 3-tab system
# ══════════════════════════════════════════════════════════════════════════════
res = st.session_state.get("result")

tab_sum, tab_steps, tab_vis = st.tabs(
    ["📊  Summary", "📝  Step-by-Step", "📈  Visualization"]
)

# ─── TAB 1: Summary ──────────────────────────────────────────────────────────
with tab_sum:
    if res is None:
        st.info("👈  Select a method and press  **▶ Calculate**  to see results.")

    elif res["kind"] == "error":
        st.error(f"⚠  **Engine Fault**\n\n{res['message']}")
        with st.expander("Full traceback"):
            st.code(res["traceback"], language="python")

    elif res["kind"] == "doolittle":
        x, n_val = res["x"], res["n"]
        st.success("✅  **LU Decomposition Solved**")
        cols = st.columns(min(n_val, 4))
        comp_labels = ["x", "y", "z"] if n_val == 3 else [f"x_{{{i+1}}}" for i in range(n_val)]
        for i, v in enumerate(x):
            with cols[i % len(cols)]:
                st.metric(label=comp_labels[i], value=f"{v:+.6f}")
        st.markdown("---")
        st.metric("System size", f"{n_val} × {n_val}")

        st.markdown("**Strategy:**")
        st.latex(r"A = LU \quad\Rightarrow\quad LV = b \quad\Rightarrow\quad UX = V")

    elif res["kind"] == "gauss_seidel":
        sol = res["sol"]
        n_val = res["n"]
        st.success(f"✅  **Gauss-Seidel — completed {res['iters']} iterations**")

        gs_labels = (["x", "y", "z"] if n_val == 3
                     else [f"x_{{{i+1}}}" for i in range(n_val)])
        cols = st.columns(min(n_val, 4))
        for i, v in enumerate(sol):
            with cols[i % len(cols)]:
                st.metric(label=gs_labels[i], value=f"{v:+.6f}")
        st.markdown("---")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Iterations", res["iters"])
        with mc2:
            st.metric("SDD", "✓ Pass" if res["sdd"] else "✗ Fail")

    elif res["kind"] == "false_position":
        root, iters = res["root"], res["iters"]
        st.success("✅  **Root Found — Regula Falsi**")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Root  x ≈", f"{root:.5f}")
        with mc2:
            st.metric("Iterations", iters)
        st.markdown("**Formula:**")
        st.latex(r"x_n = \frac{a \cdot f(b) - b \cdot f(a)}{f(b) - f(a)}")
        st.markdown(f"**Equation:** $f(x) = {res['eq']}$")

    elif res["kind"] == "newton_raphson":
        root, iters = res["root"], res["iters"]
        st.success("✅  **Root Found — Newton-Raphson**")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Root  x ≈", f"{root:.5f}")
        with mc2:
            st.metric("Iterations", iters)
        st.markdown("**Newton-Raphson Formula:**")
        st.latex(r"x_n = x_{n-1} - \frac{f(x_{n-1})}{f'(x_{n-1})}")
        st.markdown(f"$f(x) = {res['eq']}$  ·  $f'(x) = {res['fp_eq']}$")

    elif res["kind"] == "interpolation":
        result, algo, t = res["result"], res["algo"], res["t"]
        st.success(f"✅  **{algo}**")

        if "Forward" in algo:
            # Single, lecture-style metric — no generic counters
            st.metric(f"f({t:g}) ≈", f"{result:.5f}")
            st.markdown(
                f"Interpolated value at $x = {t:g}$ from "
                f"**{len(res['xs'])}** equally-spaced data points using "
                f"Newton's forward-difference polynomial."
            )
            st.latex(
                r"f(x) \;=\; y_0 + p\,\Delta y_0 + \frac{p(p-1)}{2!}\Delta^2 y_0"
                r" + \frac{p(p-1)(p-2)}{3!}\Delta^3 y_0 + \cdots,"
                r"\qquad p = \frac{x - x_0}{h}"
            )
        elif "Stirling" in algo:
            # Lecture-style Stirling summary — clean, no generic counters
            xs_s = list(res["xs"])
            idx_s = int(np.argmin([abs(xv - t) for xv in xs_s]))
            x0_s  = xs_s[idx_s]
            st.metric(f"f({t:g}) ≈", f"{result:.3f}")
            st.markdown(
                f"Centred at $x_0 = {x0_s:g}$ "
                f"(nearest data point to $x = {t:g}$)."
            )
            st.latex(
                r"f(x) \;=\; y_0 + p\,\frac{\Delta y_0 + \Delta y_{-1}}{2}"
                r" + \frac{p^{2}}{2!}\Delta^{2} y_{-1}"
                r" + \frac{p(p^{2}-1)}{3!}\,\frac{\Delta^{3} y_{-1} + \Delta^{3} y_{-2}}{2}"
                r" + \frac{p^{2}(p^{2}-1)}{4!}\Delta^{4} y_{-2} + \cdots,"
                r"\qquad p = \frac{x - x_0}{h}"
            )
        else:
            xs_l = list(res["xs"])
            n_l  = len(xs_l)
            # Show enough decimals to reveal repeating fractions like 396.6̅
            st.metric(f"f({t:g}) ≈", f"{result:.4f}")
            st.markdown(
                f"Lagrange polynomial of degree **{n_l - 1}** evaluated at "
                f"$x = {t:g}$ from the {n_l} given data points."
            )
            st.latex(
                r"f(x) \;=\; \sum_{i=0}^{n} y_i \,\prod_{\substack{j=0\\ j \neq i}}^{n}"
                r" \frac{x - x_j}{x_i - x_j}"
            )

# ─── TAB 2: Step-by-Step ─────────────────────────────────────────────────────
with tab_steps:
    if res is None:
        st.info("Results will appear here after calculating.")

    elif res["kind"] == "error":
        st.error(res["message"])
        st.code(res["traceback"], language="python")

    elif res["kind"] == "doolittle":
        # ── Premium LaTeX-rendered walkthrough for Doolittle ────────────────
        A_d, b_d = res["A"], res["b"]
        L_d, U_d = res["L"], res["U"]
        V_d, X_d = res["y"], res["x"]
        n_d      = res["n"]

        st.markdown("### Doolittle's LU Decomposition — Step-by-Step")
        st.markdown(
            "We solve $A\\mathbf{X} = \\mathbf{b}$ in three stages: factor "
            "$A = LU$ (with $L$ unit-lower-triangular, $U$ upper-triangular), "
            "introduce the intermediate vector $\\mathbf{V}$ via "
            "$U\\mathbf{X} = \\mathbf{V}$ so that the system reduces to "
            "$L\\mathbf{V} = \\mathbf{b}$, forward-solve for $\\mathbf{V}$, "
            "then back-solve $U\\mathbf{X} = \\mathbf{V}$ for $\\mathbf{X}$."
        )

        st.markdown("#### Step 1 · Original system")
        st.latex(
            rf"A \;=\; {_bmatrix_latex(A_d)}"
            rf"\qquad \mathbf{{b}} \;=\; {_bmatrix_latex(b_d)}"
        )

        st.markdown("#### Step 2 · Factor  $A = LU$")
        col_L, col_U = st.columns(2)
        with col_L:
            st.markdown("**Lower triangular  $L$**")
            st.latex(rf"L \;=\; {_bmatrix_latex(L_d)}")
        with col_U:
            st.markdown("**Upper triangular  $U$**")
            st.latex(rf"U \;=\; {_bmatrix_latex(U_d)}")

        st.markdown(
            "#### Step 3 · Substitution\n"
            "Let $U\\mathbf{X} = \\mathbf{V}$, then the system becomes "
            "$L\\mathbf{V} = \\mathbf{b}$."
        )
        st.latex(
            r"A\mathbf{X} = \mathbf{b} \;\Longrightarrow\; (LU)\mathbf{X} = \mathbf{b}"
            r" \;\Longrightarrow\; L\underbrace{(U\mathbf{X})}_{\mathbf{V}} = \mathbf{b}"
        )

        st.markdown("#### Step 4 · Forward substitution  $L\\mathbf{V} = \\mathbf{b}$")
        st.latex(
            rf"{_bmatrix_latex(L_d)} \, \begin{{bmatrix}} v_1 \\ v_2 \\ v_3 \end{{bmatrix}}"
            rf" \;=\; {_bmatrix_latex(b_d)}"
            rf" \quad\Longrightarrow\quad \mathbf{{V}} \;=\; {_bmatrix_latex(V_d, prec=6)}"
            if n_d == 3 else
            rf"{_bmatrix_latex(L_d)} \, \mathbf{{V}} \;=\; {_bmatrix_latex(b_d)}"
            rf" \quad\Longrightarrow\quad \mathbf{{V}} \;=\; {_bmatrix_latex(V_d, prec=6)}"
        )

        st.markdown("#### Step 5 · Back substitution  $U\\mathbf{X} = \\mathbf{V}$")
        st.latex(
            rf"{_bmatrix_latex(U_d)} \, \begin{{bmatrix}} x \\ y \\ z \end{{bmatrix}}"
            rf" \;=\; {_bmatrix_latex(V_d, prec=6)}"
            rf" \quad\Longrightarrow\quad \mathbf{{X}} \;=\; {_bmatrix_latex(X_d, prec=6)}"
            if n_d == 3 else
            rf"{_bmatrix_latex(U_d)} \, \mathbf{{X}} \;=\; {_bmatrix_latex(V_d, prec=6)}"
            rf" \quad\Longrightarrow\quad \mathbf{{X}} \;=\; {_bmatrix_latex(X_d, prec=6)}"
        )

        st.markdown("#### Solution")
        sol_labels = ["x", "y", "z"] if n_d == 3 else [f"x_{{{i+1}}}" for i in range(n_d)]
        sol_cols = st.columns(min(n_d, 4))
        for i, v in enumerate(X_d):
            with sol_cols[i % len(sol_cols)]:
                st.metric(label=sol_labels[i], value=f"{v:+.6f}")

        with st.expander("Numeric details · full precision"):
            col_l_df, col_u_df = st.columns(2)
            with col_l_df:
                st.markdown("**$L$**")
                df_L = pd.DataFrame(
                    L_d, columns=[f"c{i+1}" for i in range(n_d)],
                )
                st.dataframe(df_L.style.format("{:.6f}"),
                             use_container_width=True)
            with col_u_df:
                st.markdown("**$U$**")
                df_U = pd.DataFrame(
                    U_d, columns=[f"c{i+1}" for i in range(n_d)],
                )
                st.dataframe(df_U.style.format("{:.6f}"),
                             use_container_width=True)
            df_VX = pd.DataFrame({
                "V (forward sub)":  V_d,
                "X (back sub)":     X_d,
            })
            st.dataframe(df_VX.style.format("{:.10f}"),
                         use_container_width=True)

    elif res["kind"] == "gauss_seidel":
        # ── Premium LaTeX-rendered walkthrough for Gauss-Seidel ─────────────
        A_g       = res["A"]
        b_g       = res["b"]
        x0_g      = res["x0"]
        history_g = res["history"]
        sdd_g     = res["sdd"]
        n_g       = res["n"]

        var_names = (["x", "y", "z"] if n_g == 3
                     else [f"x_{{{i+1}}}" for i in range(n_g)])

        st.markdown("### Gauss-Seidel Iterative Method — Step-by-Step")
        st.markdown(
            "We solve $A\\mathbf{x} = \\mathbf{b}$ by isolating each diagonal "
            "variable and substituting the **most recent** estimates of the "
            "other unknowns at every sweep."
        )

        # ── Step 1 · SDD check ───────────────────────────────────────────────
        st.markdown("#### Step 1 · Strict Diagonal Dominance Check")
        st.markdown(
            "A sufficient condition for Gauss-Seidel convergence is "
            "$|a_{ii}| > \\sum_{j\\ne i} |a_{ij}|$ for every row."
        )
        sdd_lines = []
        for i in range(n_g):
            diag_val = abs(A_g[i, i])
            off_sum  = sum(abs(A_g[i, j]) for j in range(n_g) if j != i)
            ok       = diag_val > off_sum
            cmp      = ">" if ok else r"\le"
            mark     = r"\;\checkmark" if ok else r"\;\times"
            sdd_lines.append(
                rf"\text{{Row {i+1}: }}\;|{_fmt_num(A_g[i,i])}|"
                rf" = {_fmt_num(diag_val)} \;{cmp}\; {_fmt_num(off_sum)}{mark}"
            )
        st.latex(r"\begin{aligned}" + r" \\ ".join(sdd_lines) + r"\end{aligned}")
        if sdd_g:
            st.success("✅  **A is strictly diagonally dominant — convergence guaranteed.**")
        else:
            st.warning("⚠  **A is not strictly diagonally dominant — convergence is not guaranteed.**")

        # ── Step 2 · Isolated form ───────────────────────────────────────────
        st.markdown("#### Step 2 · Isolated Form")
        st.markdown(
            "Solve each equation for its diagonal variable so it can be "
            "updated iteratively:"
        )
        iso_lines = []
        for i in range(n_g):
            rhs   = _gs_isolated_rhs(A_g[i], float(b_g[i]), i, var_names)
            frac  = _gs_diag_frac(float(A_g[i, i]))
            iso_lines.append(rf"{var_names[i]} &= {frac}\left[{rhs}\right]")
        st.latex(r"\begin{aligned}" + r" \\ ".join(iso_lines) + r"\end{aligned}")

        # ── Step 3 · Iteration history ───────────────────────────────────────
        st.markdown("#### Step 3 · Iteration History")
        init_line = ", \\;".join(
            rf"{var_names[i]}^{{(0)}} = {x0_g[i]:.4f}" for i in range(n_g)
        )
        st.latex(rf"\textbf{{Initial guess:}}\quad {init_line}")

        x_prev = x0_g.astype(float).copy()
        for k, x_new in enumerate(history_g, start=1):
            st.markdown(f"##### Iteration {k}")
            sub_lines = []
            for i in range(n_g):
                # In Gauss-Seidel: use the just-updated values for j < i and
                # previous-iterate values for j > i.
                x_subst = np.array(
                    [x_new[j] if j < i else x_prev[j] for j in range(n_g)],
                    dtype=float,
                )
                rhs, _val = _gs_substitution_rhs(
                    A_g[i], float(b_g[i]), i, x_subst
                )
                frac     = _gs_diag_frac(float(A_g[i, i]))
                final_v  = float(x_new[i])
                sub_lines.append(
                    rf"{var_names[i]}^{{({k})}} &= {frac}\left[{rhs}\right]"
                    rf" \;=\; {final_v:+.4f}"
                )
            st.latex(r"\begin{aligned}" + r" \\ ".join(sub_lines) + r"\end{aligned}")
            x_prev = np.asarray(x_new, dtype=float).copy()

        # ── Final solution metrics ───────────────────────────────────────────
        st.markdown("#### Final Solution")
        sol_cols = st.columns(min(n_g, 4))
        final_x  = history_g[-1] if history_g else x0_g
        for i, v in enumerate(final_x):
            with sol_cols[i % len(sol_cols)]:
                st.metric(label=var_names[i], value=f"{v:+.6f}")

        with st.expander("Iteration table · numeric values"):
            cols = ([var_names[i] for i in range(n_g)] if n_g == 3
                    else [f"x_{i+1}" for i in range(n_g)])
            rows = [["0"] + [f"{v:.6f}" for v in x0_g]]
            for k, h in enumerate(history_g, start=1):
                rows.append([str(k)] + [f"{v:.6f}" for v in h])
            df_iter = pd.DataFrame(rows, columns=["Iter"] + cols)
            st.dataframe(df_iter, use_container_width=True, hide_index=True)

    elif res["kind"] == "false_position":
        # ── Premium LaTeX walkthrough for False Position (Regula Falsi) ─────
        eq        = res["eq"]
        a0, b0    = res["a0"], res["b0"]
        fa0, fb0  = res["fa0"], res["fb0"]
        history_f = res["history"]
        tol_f     = res["tol"]
        root_f    = res["root"]

        st.markdown("### False Position Method (Regula Falsi) — Step-by-Step")
        st.markdown(
            f"Find a root of $f(x) = {eq}$ on the interval "
            f"$[{a0:g},\\,{b0:g}]$, correct to "
            f"**{max(0, -int(round(math.log10(tol_f)))):d} decimal places** "
            f"(tolerance $= {tol_f:g}$)."
        )

        # ── Initial Check ────────────────────────────────────────────────────
        st.markdown("#### Initial Check")
        st.latex(rf"f(x) = {eq}")
        st.latex(
            rf"f({a0:g}) \;=\; {fa0:.5f} "
            rf"\;{'>' if fa0 > 0 else '<'}\; 0"
        )
        st.latex(
            rf"f({b0:g}) \;=\; {fb0:.5f} "
            rf"\;{'>' if fb0 > 0 else '<'}\; 0"
        )
        st.success(
            f"Since $f({a0:g}) \\cdot f({b0:g}) < 0$, "
            f"a root exists between **{a0:g}** and **{b0:g}**."
        )

        # ── Per-iteration walkthrough ───────────────────────────────────────
        st.markdown("#### Iterations")
        st.latex(
            r"x_n \;=\; \frac{a \cdot f(b) - b \cdot f(a)}{f(b) - f(a)}"
        )

        for k, h in enumerate(history_f, start=1):
            a_k, b_k = h["a"], h["b"]
            fa_k, fb_k = h["fa"], h["fb"]
            x_k, fx_k = h["x"], h["fx"]
            err_k = h["err"]

            st.markdown(
                f"##### Iteration {k} :  "
                f"$a = {a_k:.5f},\\; b = {b_k:.5f}$"
            )

            # Substitution into the formula  -----------------------------------
            num_sym = r"a\cdot f(b) - b\cdot f(a)"
            den_sym = r"f(b) - f(a)"
            num_val = a_k * fb_k - b_k * fa_k
            den_val = fb_k - fa_k

            st.latex(
                rf"x_{{{k}}} \;=\; \frac{{{num_sym}}}{{{den_sym}}}"
                rf" \;=\; \frac{{({a_k:.5f})\cdot({fb_k:+.5f}) "
                rf"- ({b_k:.5f})\cdot({fa_k:+.5f})}}"
                rf"{{({fb_k:+.5f}) - ({fa_k:+.5f})}}"
                rf" \;=\; \frac{{{num_val:+.5f}}}{{{den_val:+.5f}}}"
                rf" \;=\; {x_k:.5f}"
            )

            # Evaluate f(x_k)  -------------------------------------------------
            st.markdown(f"**Evaluate $f(x_{{{k}}})$:**")
            st.latex(
                rf"f({x_k:.5f}) \;=\; e^{{{x_k:.5f}}} - 3\,({x_k:.5f})^2"
                rf" \;=\; {fx_k:+.5f}"
                rf" \;{'>' if fx_k > 0 else '<'}\; 0"
                if eq.replace(" ", "") == "exp(x)-3*x**2"
                else rf"f({x_k:.5f}) \;=\; {fx_k:+.6f}"
                rf" \;{'>' if fx_k > 0 else '<'}\; 0"
            )

            # Error  (from iteration 2 onwards)  -------------------------------
            if err_k is not None:
                x_prev = history_f[k - 2]["x"]
                st.latex(
                    rf"|x_{{{k}}} - x_{{{k-1}}}| \;=\; "
                    rf"|{x_k:.5f} - {x_prev:.5f}| \;=\; {err_k:.5f}"
                )

            # Update interval --------------------------------------------------
            # The endpoint that *retains* sign change with x_k stays;
            # the other endpoint is replaced by x_k.
            if fa_k * fx_k < 0:
                fixed_label, fixed_val, fixed_fval = "b", b_k, fb_k
                new_a, new_b = a_k, x_k
            else:
                fixed_label, fixed_val, fixed_fval = "a", a_k, fa_k
                new_a, new_b = x_k, b_k

            fixed_sign = "negative" if fixed_fval < 0 else "positive"
            fx_sign    = "negative" if fx_k       < 0 else "positive"

            st.markdown(
                f"**Update Interval:** Since $f({fixed_val:g})$ is "
                f"**{fixed_sign}** and $f({x_k:.5f})$ is "
                f"**{fx_sign}**, the new interval is "
                f"$[{new_a:.5f},\\;{new_b:.5f}]$."
            )

            st.markdown("---")

        # ── Final answer ─────────────────────────────────────────────────────
        st.markdown("#### Final Answer")
        prec_dp = max(0, -int(round(math.log10(tol_f))))
        st.success(
            f"The root is **{root_f:.{prec_dp}f}** "
            f"accurate to **{prec_dp} decimal places** "
            f"(achieved in {res['iters']} iterations, "
            f"final error $|x_{{n}} - x_{{n-1}}| = "
            f"{history_f[-1]['err']:.2e}$)."
        )

        with st.expander("Iteration table · numeric values"):
            df_fp = pd.DataFrame([
                {
                    "Iter": k,
                    "a":     h["a"],
                    "b":     h["b"],
                    "f(a)":  h["fa"],
                    "f(b)":  h["fb"],
                    "xₙ":    h["x"],
                    "f(xₙ)": h["fx"],
                    "|xₙ − xₙ₋₁|": h["err"],
                }
                for k, h in enumerate(history_f, start=1)
            ])
            st.dataframe(
                df_fp.style.format({
                    "a":     "{:.6f}",
                    "b":     "{:.6f}",
                    "f(a)":  "{:+.6f}",
                    "f(b)":  "{:+.6f}",
                    "xₙ":    "{:.6f}",
                    "f(xₙ)": "{:+.2e}",
                    "|xₙ − xₙ₋₁|": lambda v: "—" if v is None or pd.isna(v) else f"{v:.2e}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    elif res["kind"] == "newton_raphson":
        # ── Premium LaTeX walkthrough for Newton-Raphson ────────────────────
        eq         = res["eq"]
        fp_eq      = res["fp_eq"]
        x0_n       = res["x0"]
        tol_n      = res["tol"]
        history_n  = res["history"]
        root_n     = res["root"]
        iters_n    = res["iters"]

        prec_dp = max(0, -int(round(math.log10(tol_n))))

        st.markdown("### Newton-Raphson Method — Step-by-Step")
        st.markdown(
            f"Find a real root of $f(x) = 0$ near $x_0 = {x0_n:g}$, "
            f"accurate to **{prec_dp} decimal places** "
            f"(tolerance $|x_n - x_{{n-1}}| < {tol_n:g}$)."
        )

        # ── Initial Setup ────────────────────────────────────────────────────
        st.markdown("#### Initial Setup")
        st.markdown("Here,")
        st.latex(rf"f(x) \;=\; {_py_expr_to_latex(eq)}")
        st.latex(rf"f'(x) \;=\; {_py_expr_to_latex(fp_eq)}")

        h0      = history_n[0]
        fx0     = h0["fx"]
        fpx0    = h0["fpx"]
        f_sub0  = _py_expr_to_latex(eq, sub_val=x0_n)
        fp_sub0 = _py_expr_to_latex(fp_eq, sub_val=x0_n)

        st.markdown(f"The initial guess of the solution is $x_0 = {x0_n:g}$.")
        st.latex(
            rf"f(x_0) \;=\; f({x0_n:g}) \;=\; {f_sub0}"
            rf" \;=\; {fx0:g}"
        )
        st.latex(
            rf"f'(x_0) \;=\; f'({x0_n:g}) \;=\; {fp_sub0}"
            rf" \;=\; {fpx0:g}"
        )

        # ── Iterations ───────────────────────────────────────────────────────
        st.markdown("#### Iterations")
        st.markdown("Therefore, applying the Newton-Raphson formula:")
        st.latex(
            r"x_n \;=\; x_{n-1} \;-\; \frac{f(x_{n-1})}{f'(x_{n-1})}"
        )

        for k in range(1, len(history_n)):
            h_prev = history_n[k - 1]
            h_curr = history_n[k]
            x_prev   = h_prev["x"]
            fx_prev  = h_prev["fx"]
            fpx_prev = h_prev["fpx"]
            x_new    = h_curr["x"]
            err_k    = h_curr["err"]

            st.markdown(f"##### Iteration {k}")

            # Substitute x into f and f' for lecture-style expanded display
            f_sub_latex  = _py_expr_to_latex(eq,    sub_val=x_prev)
            fp_sub_latex = _py_expr_to_latex(fp_eq, sub_val=x_prev)

            x_prev_str = (f"{int(round(x_prev))}"
                          if abs(x_prev - round(x_prev)) < 1e-12
                          else f"{x_prev:.5f}")

            if k == 1:
                # Iteration 1: simple form, values fresh from Initial Setup
                st.latex(
                    rf"x_{{{k}}} \;=\; x_{{{k-1}}} - \frac{{f(x_{{{k-1}}})}}{{f'(x_{{{k-1}}})}}"
                    rf" \;=\; {x_prev_str} - \frac{{{fx_prev:g}}}{{{fpx_prev:g}}}"
                    rf" \;=\; {x_new:.5f}"
                )
            else:
                # Iterations 2+: expanded form mirrors lecture exactly
                st.latex(
                    rf"x_{{{k}}} \;=\; x_{{{k-1}}} - \frac{{f(x_{{{k-1}}})}}{{f'(x_{{{k-1}}})}}"
                    rf" \;=\; {x_prev_str} - \frac{{{f_sub_latex}}}{{{fp_sub_latex}}}"
                    rf" \;=\; {x_new:.5f}"
                )

            # Error |x_k - x_{k-1}|
            if err_k is not None:
                st.latex(
                    rf"|x_{{{k}}} - x_{{{k-1}}}| \;=\; "
                    rf"|{x_new:.5f} - {x_prev_str}| \;=\; {err_k:.7f}"
                )

            st.markdown("---")

        # ── Final answer ─────────────────────────────────────────────────────
        st.markdown("#### Final Answer")
        st.success(
            f"Hence, the root of the equation is **{root_n:.{prec_dp}f}** "
            f"(achieved in {iters_n} iterations, "
            f"final step $|x_n - x_{{n-1}}| = "
            f"{history_n[-1]['err']:.2e}$)."
        )

        with st.expander("Iteration table · numeric values"):
            df_nr = pd.DataFrame([
                {
                    "n":       k,
                    "xₙ":      h["x"],
                    "f(xₙ)":   h["fx"],
                    "f′(xₙ)":  h["fpx"],
                    "|xₙ − xₙ₋₁|": h["err"],
                }
                for k, h in enumerate(history_n)
            ])
            st.dataframe(
                df_nr.style.format({
                    "xₙ":      "{:.6f}",
                    "f(xₙ)":   "{:+.6e}",
                    "f′(xₙ)":  "{:+.6f}",
                    "|xₙ − xₙ₋₁|": lambda v: "—" if v is None or pd.isna(v) else f"{v:.2e}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    elif res["kind"] == "interpolation" and "Forward" in res["algo"]:
        # ── Premium LaTeX walkthrough for Newton's Forward Interpolation ────
        xs_n   = list(res["xs"])
        ys_n   = list(res["ys"])
        diff_n = res["diff"]
        t_n    = float(res["t"])
        result_n = float(res["result"])
        n_n    = len(xs_n)
        x0_n   = xs_n[0]
        h_n    = xs_n[1] - xs_n[0]
        p_n    = (t_n - x0_n) / h_n

        st.markdown("### Newton's Forward Interpolation — Step-by-Step")
        st.markdown(
            f"Estimate $f({t_n:g})$ from the **{n_n}** equally-spaced data "
            f"points using Newton's forward-difference polynomial centred at "
            f"$x_0 = {x0_n:g}$."
        )

        # ── Step 1 · Data ────────────────────────────────────────────────────
        st.markdown("#### Step 1 · Given Data")
        data_html = (
            '<table style="border-collapse:collapse;margin:8px 0;'
            'background:#141420;border-radius:8px;overflow:hidden;">'
            '<thead><tr>'
            f'<th style="padding:10px 18px;text-align:center;background:#0E0E1A;'
            f'color:#9CDCFE;border:1px solid #2B2B2B;">Year x</th>'
            + "".join(
                f'<th style="padding:10px 18px;text-align:center;background:#0E0E1A;'
                f'color:#D4D4D4;border:1px solid #2B2B2B;">{_fwd_diff_num_str(x)}</th>'
                for x in xs_n
            )
            + '</tr></thead><tbody><tr>'
            f'<td style="padding:10px 18px;text-align:center;background:#0E0E1A;'
            f'color:#9CDCFE;border:1px solid #2B2B2B;font-weight:700;">f(x)</td>'
            + "".join(
                f'<td style="padding:10px 18px;text-align:center;'
                f'border:1px solid #2B2B2B;color:#D4D4D4;">{_fwd_diff_num_str(y)}</td>'
                for y in ys_n
            )
            + '</tr></tbody></table>'
        )
        st.markdown(data_html, unsafe_allow_html=True)

        # ── Step 2 · Forward Difference Table ────────────────────────────────
        st.markdown("#### Step 2 · Forward Difference Table")
        st.markdown(
            "Each column is the forward difference of the previous one:  "
            "$\\Delta y_i = y_{i+1} - y_i$,  "
            "$\\Delta^{2} y_i = \\Delta y_{i+1} - \\Delta y_i$,  and so on."
        )
        st.markdown(
            _forward_diff_staircase_html(xs_n, ys_n, diff_n),
            unsafe_allow_html=True,
        )

        # ── Step 3 · Interval width h and parameter p ───────────────────────
        st.markdown("#### Step 3 · Compute  $h$  and  $p$")
        st.markdown(
            f"The data are equally spaced; the **interval width** is "
            f"$h = x_1 - x_0$:"
        )
        st.latex(
            rf"h \;=\; {_fwd_diff_num_str(xs_n[1])} - {_fwd_diff_num_str(xs_n[0])}"
            rf" \;=\; {_fwd_diff_num_str(h_n)}"
        )
        st.markdown(
            f"With $x = {t_n:g}$ and $x_0 = {x0_n:g}$, the **forward "
            f"parameter** is:"
        )
        st.latex(
            rf"p \;=\; \frac{{x - x_0}}{{h}}"
            rf" \;=\; \frac{{{_fwd_diff_num_str(t_n)} - {_fwd_diff_num_str(x0_n)}}}"
            rf"{{{_fwd_diff_num_str(h_n)}}}"
            rf" \;=\; \frac{{{_fwd_diff_num_str(t_n - x0_n)}}}{{{_fwd_diff_num_str(h_n)}}}"
            rf" \;=\; {p_n:.2f}"
        )

        # ── Step 4 · Newton's Forward Formula ────────────────────────────────
        st.markdown("#### Step 4 · Newton's Forward Formula")
        st.latex(
            r"f(x) \;=\; y_0 + p\,\Delta y_0"
            r" + \frac{p(p-1)}{2!}\Delta^{2} y_0"
            r" + \frac{p(p-1)(p-2)}{3!}\Delta^{3} y_0"
            r" + \frac{p(p-1)(p-2)(p-3)}{4!}\Delta^{4} y_0 + \cdots"
        )

        # ── Step 5 · Substitution ────────────────────────────────────────────
        st.markdown("#### Step 5 · Substitute the Diagonal Values")
        # Forward-diagonal values feeding the polynomial
        diag = [diff_n[k][0] for k in range(min(n_n, len(diff_n)))]

        def _fmt_diag(v: float) -> str:
            return f"({v:+g})" if v < 0 else f"{v:g}"

        # Build the f(t_n) = ... line term-by-term to mirror the lecture exactly
        sub_lines: list[str] = [rf"f({t_n:g}) \;=\; {_fmt_diag(diag[0])}"]
        if len(diag) >= 2:
            sub_lines.append(rf"\; + \; ({p_n:.2f}){_fmt_diag(diag[1])}")
        if len(diag) >= 3:
            sub_lines.append(
                rf"\; + \; \frac{{({p_n:.2f})({p_n:.2f} - 1)}}{{2!}}{_fmt_diag(diag[2])}"
            )
        if len(diag) >= 4:
            sub_lines.append(
                rf"\; + \; \frac{{({p_n:.2f})({p_n:.2f} - 1)({p_n:.2f} - 2)}}{{3!}}"
                rf"{_fmt_diag(diag[3])}"
            )
        if len(diag) >= 5:
            sub_lines.append(
                rf"\; + \; \frac{{({p_n:.2f})({p_n:.2f} - 1)({p_n:.2f} - 2)({p_n:.2f} - 3)}}{{4!}}"
                rf"{_fmt_diag(diag[4])}"
            )
        # Higher orders (rare for the lecture example) — append generically
        for k in range(5, len(diag)):
            factors = "".join(
                rf"({p_n:.2f} - {j})" if j else f"({p_n:.2f})"
                for j in range(k)
            )
            sub_lines.append(
                rf"\; + \; \frac{{{factors}}}{{{k}!}}{_fmt_diag(diag[k])}"
            )
        st.latex(r" \\ ".join(sub_lines))

        # ── Step 6 · Term-by-term values ────────────────────────────────────
        st.markdown("#### Step 6 · Evaluate Each Term")

        # Compute each term numerically, then sum
        terms: list[tuple[str, float]] = []
        p_prod = 1.0
        fact_k = 1.0
        for k in range(len(diag)):
            if k == 0:
                tval  = diag[0]
                label = "y_0"
            else:
                p_prod *= (p_n - (k - 1))
                fact_k *= k
                tval  = (p_prod / fact_k) * diag[k]
                if k == 1:
                    label = r"p\,\Delta y_0"
                else:
                    factors = "".join(
                        rf"({p_n:.2f} - {j})" if j else f"({p_n:.2f})"
                        for j in range(k)
                    )
                    label = rf"\frac{{{factors}}}{{{k}!}}\Delta^{{{k}}} y_0"
            terms.append((label, float(tval)))

        term_lines = []
        running = 0.0
        for k, (label, tval) in enumerate(terms):
            running += tval
            term_lines.append(
                rf"T_{{{k}}} &= {label} \;=\; {tval:+.5f}"
            )
        st.latex(
            r"\begin{aligned}" + r" \\ ".join(term_lines) + r"\end{aligned}"
        )

        # ── Step 7 · Final Answer ────────────────────────────────────────────
        st.markdown("#### Step 7 · Final Answer")
        sum_str = "".join(f"({tval:+.5f})" for _, tval in terms)
        st.latex(
            rf"f({t_n:g}) \;=\; {sum_str} \;=\; \boxed{{{result_n:.2f}}}"
        )
        st.success(
            f"**The interpolated profit at year {t_n:g} is "
            f"${result_n:.2f}$ thousand rupees** "
            f"(rounded to 2 decimals: **{result_n:.2f}**)."
        )

        with st.expander("Forward difference table · numeric values"):
            df_diff = _diff_df(diff_n)
            st.dataframe(
                df_diff.style.format(
                    {c: "{:.4f}" for c in df_diff.columns},
                    na_rep="—",
                ),
                use_container_width=True,
            )

    elif res["kind"] == "interpolation" and "Stirling" in res["algo"]:
        # ── Premium LaTeX walkthrough for Stirling's Central Differences ────
        xs_s    = list(res["xs"])
        ys_s    = list(res["ys"])
        diff_s  = res["diff"]
        t_s     = float(res["t"])
        result_s = float(res["result"])
        n_s     = len(xs_s)
        # Origin = nearest data point to t (matches calculator.stirling_interpolation)
        idx_s   = int(np.argmin([abs(xv - t_s) for xv in xs_s]))
        x0_s    = xs_s[idx_s]
        h_s     = xs_s[1] - xs_s[0]
        p_s     = (t_s - x0_s) / h_s

        st.markdown("### Stirling's Central Difference Interpolation — Step-by-Step")
        st.markdown(
            f"Estimate $f({t_s:g})$ using Stirling's formula, which is most "
            f"accurate when $x$ lies near the **centre** of the data table. "
            f"Here the closest tabulated point is $x_0 = {x0_s:g}$."
        )

        # ── Step 1 · Given Data ───────────────────────────────────────────────
        st.markdown("#### Step 1 · Given Data")
        data_html = (
            '<table style="border-collapse:collapse;margin:8px 0;'
            'background:#141420;border-radius:8px;overflow:hidden;">'
            '<thead><tr>'
            f'<th style="padding:10px 18px;text-align:center;background:#0E0E1A;'
            f'color:#9CDCFE;border:1px solid #2B2B2B;">x</th>'
            + "".join(
                f'<th style="padding:10px 18px;text-align:center;background:#0E0E1A;'
                f'color:#D4D4D4;border:1px solid #2B2B2B;">{_fwd_diff_num_str(x)}</th>'
                for x in xs_s
            )
            + '</tr></thead><tbody><tr>'
            f'<td style="padding:10px 18px;text-align:center;background:#0E0E1A;'
            f'color:#9CDCFE;border:1px solid #2B2B2B;font-weight:700;">y</td>'
            + "".join(
                f'<td style="padding:10px 18px;text-align:center;'
                f'border:1px solid #2B2B2B;color:#D4D4D4;">{_fwd_diff_num_str(y)}</td>'
                for y in ys_s
            )
            + '</tr></tbody></table>'
        )
        st.markdown(data_html, unsafe_allow_html=True)

        # ── Step 2 · Central Difference Table ────────────────────────────────
        st.markdown("#### Step 2 · Central Difference Table")
        st.markdown(
            f"The differences are computed identically to the forward table; "
            f"Stirling's formula only changes which entries are *selected*. "
            f"The **entire row** of $x_0 = {x0_s:g}$ is highlighted in gold, "
            f"and the **central path** values consumed by the formula "
            f"(symmetric pairs above/below this row, plus even-order "
            f"differences sitting on the row itself) are bordered in bright "
            f"yellow."
        )
        st.markdown(
            _central_diff_staircase_html(xs_s, ys_s, diff_s, idx_s),
            unsafe_allow_html=True,
        )

        # ── Step 3 · Compute h, x0, p ────────────────────────────────────────
        st.markdown("#### Step 3 · Compute  $h$,  $x_0$  and  $p$")
        st.markdown(
            "The data are equally spaced; the **interval width** is "
            f"$h = x_1 - x_0$ where the indices refer to the tabulated "
            f"points (not the central origin)."
        )
        st.latex(
            rf"h \;=\; {_fwd_diff_num_str(xs_s[1])} - {_fwd_diff_num_str(xs_s[0])}"
            rf" \;=\; {_fwd_diff_num_str(h_s)}"
        )
        st.markdown(
            f"Choose $x_0$ as the data point nearest to "
            f"$x = {t_s:g}$:&nbsp; $x_0 = {x0_s:g}$."
        )
        st.markdown(
            f"Then the **central parameter** is "
            f"$p = (x - x_0)/h$:"
        )
        st.latex(
            rf"p \;=\; \frac{{x - x_0}}{{h}}"
            rf" \;=\; \frac{{{_fwd_diff_num_str(t_s)} - {_fwd_diff_num_str(x0_s)}}}"
            rf"{{{_fwd_diff_num_str(h_s)}}}"
            rf" \;=\; \frac{{{_fwd_diff_num_str(t_s - x0_s)}}}{{{_fwd_diff_num_str(h_s)}}}"
            rf" \;=\; {p_s:.2f}"
        )

        # ── Step 4 · Stirling's Formula ──────────────────────────────────────
        st.markdown("#### Step 4 · Stirling's Formula")
        st.markdown(
            "Odd-order terms use the **average** of the two neighbouring "
            "differences; even-order terms use a single central difference:"
        )
        st.latex(
            r"f(x) \;=\; y_0"
            r" \;+\; p\,\frac{\Delta y_0 + \Delta y_{-1}}{2}"
            r" \;+\; \frac{p^{2}}{2!}\Delta^{2} y_{-1}"
            r" \;+\; \frac{p(p^{2}-1)}{3!}\,\frac{\Delta^{3} y_{-1} + \Delta^{3} y_{-2}}{2}"
            r" \;+\; \frac{p^{2}(p^{2}-1)}{4!}\Delta^{4} y_{-2}"
            r" \;+\; \cdots"
        )

        # ── Step 5 · Substitution ────────────────────────────────────────────
        st.markdown("#### Step 5 · Substitute the Central Values")
        # Pull the central-path values from the diff table.
        # (Some may be unavailable if the table is too small / x₀ is near an edge.)
        def _safe(k: int, j: int):
            if 0 <= k < len(diff_s) and 0 <= j < len(diff_s[k]):
                return diff_s[k][j]
            return None

        y0       = _safe(0, idx_s)
        d1_m1    = _safe(1, idx_s - 1)         # Δy_{-1}
        d1_0     = _safe(1, idx_s)             # Δy_0
        d2_m1    = _safe(2, idx_s - 1)         # Δ²y_{-1}
        d3_m2    = _safe(3, idx_s - 2)         # Δ³y_{-2}
        d3_m1    = _safe(3, idx_s - 1)         # Δ³y_{-1}
        d4_m2    = _safe(4, idx_s - 2)         # Δ⁴y_{-2}

        def _vfmt(v):
            if v is None:
                return r"\,?\,"
            return f"{v:+g}" if v < 0 else f"{v:g}"

        def _vbrk(v):
            """Wrap negatives in parentheses for clean lecture style."""
            if v is None:
                return r"(\,?\,)"
            return f"({v:+g})" if v < 0 else f"{v:g}"

        p_str = f"{p_s:.2f}"
        # Build the substitution line term-by-term, matching the lecture exactly.
        sub_lines: list[str] = [rf"f({t_s:g}) \;=\; {y0:g}"]
        if d1_0 is not None and d1_m1 is not None:
            sub_lines.append(
                rf"\; + \; ({p_str})\,\frac{{{_vbrk(d1_0)} + {_vbrk(d1_m1)}}}{{2}}"
            )
        if d2_m1 is not None:
            sub_lines.append(
                rf"\; + \; \frac{{({p_str})^{{2}}}}{{2!}}\,{_vbrk(d2_m1)}"
            )
        if d3_m1 is not None and d3_m2 is not None:
            sub_lines.append(
                rf"\; + \; \frac{{({p_str})\bigl(({p_str})^{{2}} - 1\bigr)}}{{3!}}"
                rf"\,\frac{{{_vbrk(d3_m1)} + {_vbrk(d3_m2)}}}{{2}}"
            )
        if d4_m2 is not None:
            sub_lines.append(
                rf"\; + \; \frac{{({p_str})^{{2}}\bigl(({p_str})^{{2}} - 1\bigr)}}{{4!}}"
                rf"\,{_vbrk(d4_m2)}"
            )
        st.latex(r" \\ ".join(sub_lines))

        # ── Step 6 · Term-by-term values ─────────────────────────────────────
        st.markdown("#### Step 6 · Evaluate Each Term")
        terms: list[tuple[str, float]] = []
        # T0
        terms.append((r"y_0", float(y0)))
        # T1 (odd-1)
        if d1_0 is not None and d1_m1 is not None:
            t1 = p_s * (d1_0 + d1_m1) / 2.0
            terms.append((
                r"p\,\frac{\Delta y_0 + \Delta y_{-1}}{2}",
                t1,
            ))
        # T2 (even-2)
        if d2_m1 is not None:
            t2 = (p_s ** 2 / 2.0) * d2_m1
            terms.append((
                r"\frac{p^{2}}{2!}\Delta^{2} y_{-1}",
                t2,
            ))
        # T3 (odd-3)
        if d3_m1 is not None and d3_m2 is not None:
            t3 = (p_s * (p_s ** 2 - 1) / 6.0) * (d3_m1 + d3_m2) / 2.0
            terms.append((
                r"\frac{p(p^{2}-1)}{3!}\,\frac{\Delta^{3} y_{-1} + \Delta^{3} y_{-2}}{2}",
                t3,
            ))
        # T4 (even-4)
        if d4_m2 is not None:
            t4 = (p_s ** 2 * (p_s ** 2 - 1) / 24.0) * d4_m2
            terms.append((
                r"\frac{p^{2}(p^{2}-1)}{4!}\Delta^{4} y_{-2}",
                t4,
            ))

        term_lines = [
            rf"T_{{{k}}} &= {label} \;=\; {tval:+.5f}"
            for k, (label, tval) in enumerate(terms)
        ]
        st.latex(r"\begin{aligned}" + r" \\ ".join(term_lines) + r"\end{aligned}")

        # ── Step 7 · Final Answer ────────────────────────────────────────────
        st.markdown("#### Step 7 · Final Answer")
        running_sum = sum(v for _, v in terms)
        sum_str = "".join(f"({tval:+.5f})" for _, tval in terms)
        st.latex(
            rf"f({t_s:g}) \;=\; {sum_str} \;=\; \boxed{{{result_s:.3f}}}"
        )
        st.success(
            f"**The Stirling-interpolated value at $x = {t_s:g}$ is "
            f"${result_s:.3f}$** "
            f"(centred at $x_0 = {x0_s:g}$, $h = {h_s:g}$, $p = {p_s:.2f}$)."
        )

        with st.expander("Central difference table · numeric values"):
            df_diff = _diff_df(diff_s)
            st.dataframe(
                df_diff.style.format(
                    {c: "{:.4f}" for c in df_diff.columns},
                    na_rep="—",
                ),
                use_container_width=True,
            )

    elif res["kind"] == "interpolation" and "Lagrange" in res["algo"]:
        # ── Premium LaTeX walkthrough for Lagrange Interpolation ────────────
        # Mirrors the HUE lecture slide: general formula, given variables,
        # explicit numerical substitution with \dfrac fractions, final value.
        xs_l     = list(res["xs"])
        ys_l     = list(res["ys"])
        t_l      = float(res["t"])
        result_l = float(res["result"])
        n_l      = len(xs_l)

        def _fmt_num_l(v: float) -> str:
            """Integer-when-integral formatter for Lagrange data values."""
            if abs(v - round(v)) < 1e-12:
                return f"{int(round(v))}"
            return f"{v:g}"

        st.markdown("### Lagrange's Interpolation — Step-by-Step")
        st.markdown(
            f"Using **Lagrange's interpolation formula**, find the value of "
            f"$y$ corresponding to $x = {t_l:g}$ from the following data."
        )

        # ── Given Data table ────────────────────────────────────────
        data_html = (
            '<table style="border-collapse:collapse;margin:8px 0 14px 0;'
            'background:#141420;border-radius:8px;overflow:hidden;">'
            '<thead><tr>'
            '<th style="padding:10px 22px;text-align:center;background:#0E0E1A;'
            'color:#9CDCFE;border:1px solid #2B2B2B;font-weight:700;">x</th>'
            + "".join(
                f'<th style="padding:10px 22px;text-align:center;background:#0E0E1A;'
                f'color:#D4D4D4;border:1px solid #2B2B2B;">{_fmt_num_l(x)}</th>'
                for x in xs_l
            )
            + '</tr></thead><tbody><tr>'
            '<td style="padding:10px 22px;text-align:center;background:#0E0E1A;'
            'color:#9CDCFE;border:1px solid #2B2B2B;font-weight:700;">y</td>'
            + "".join(
                f'<td style="padding:10px 22px;text-align:center;'
                f'border:1px solid #2B2B2B;color:#D4D4D4;">{_fmt_num_l(y)}</td>'
                for y in ys_l
            )
            + '</tr></tbody></table>'
        )
        st.markdown(data_html, unsafe_allow_html=True)

        # ── Step 1 · General Lagrange formula ────────────────────────
        st.markdown("#### Step 1 · The Lagrange Interpolation Formula")
        st.markdown(
            f"For **{n_l}** data points the Lagrange polynomial is the sum of "
            f"$n+1 = {n_l}$ basis terms, each of which equals $1$ at its own "
            f"node and $0$ at every other node:"
        )
        if n_l == 4:
            st.latex(
                r"f(x) \;=\;"
                r"\dfrac{(x-x_{1})(x-x_{2})(x-x_{3})}{(x_{0}-x_{1})(x_{0}-x_{2})(x_{0}-x_{3})}\,y_{0}"
                r"\;+\;"
                r"\dfrac{(x-x_{0})(x-x_{2})(x-x_{3})}{(x_{1}-x_{0})(x_{1}-x_{2})(x_{1}-x_{3})}\,y_{1}"
            )
            st.latex(
                r"\qquad\;+\;"
                r"\dfrac{(x-x_{0})(x-x_{1})(x-x_{3})}{(x_{2}-x_{0})(x_{2}-x_{1})(x_{2}-x_{3})}\,y_{2}"
                r"\;+\;"
                r"\dfrac{(x-x_{0})(x-x_{1})(x-x_{2})}{(x_{3}-x_{0})(x_{3}-x_{1})(x_{3}-x_{2})}\,y_{3}"
            )
        else:
            st.latex(
                r"f(x) \;=\; \sum_{i=0}^{n} y_{i} \,"
                r"\prod_{\substack{j=0\\ j \neq i}}^{n} \frac{x - x_{j}}{x_{i} - x_{j}}"
            )

        # ── Step 2 · Given variables ──────────────────────────────────
        st.markdown("#### Step 2 · Given Variables")
        x_line = ",\\;".join(
            [rf"x = {_fmt_num_l(t_l)}"]
            + [rf"x_{{{i}}} = {_fmt_num_l(xs_l[i])}" for i in range(n_l)]
        )
        y_line = ",\\;".join(
            rf"y_{{{i}}} = {_fmt_num_l(ys_l[i])}" for i in range(n_l)
        )
        st.latex(x_line)
        st.latex(y_line)

        # ── Step 3 · Explicit numerical substitution ────────────────────
        st.markdown("#### Step 3 · Substitute the Numerical Values")
        st.markdown(
            "Replace every symbol $x$, $x_i$, $y_i$ with its numerical value. "
            f"Each fraction is the **Lagrange basis** $L_i({t_l:g})$:"
        )

        term_strs: list[str] = []
        for i in range(n_l):
            num = "".join(
                f"({_fmt_num_l(t_l)}-{_fmt_num_l(xs_l[j])})"
                for j in range(n_l) if j != i
            )
            den = "".join(
                f"({_fmt_num_l(xs_l[i])}-{_fmt_num_l(xs_l[j])})"
                for j in range(n_l) if j != i
            )
            term_strs.append(rf"\dfrac{{{num}}}{{{den}}}\,({_fmt_num_l(ys_l[i])})")

        sub_lines: list[str] = []
        head = rf"f({_fmt_num_l(t_l)}) &=\; {term_strs[0]}"
        if n_l >= 2:
            head += rf" \;+\; {term_strs[1]}"
        sub_lines.append(head)
        i = 2
        while i < n_l:
            line = rf"&\;+\; {term_strs[i]}"
            if i + 1 < n_l:
                line += rf" \;+\; {term_strs[i+1]}"
            sub_lines.append(line)
            i += 2
        st.latex(r"\begin{aligned}" + r" \\[6pt] ".join(sub_lines) + r"\end{aligned}")

        # ── Step 4 · Per-term numeric values ──────────────────────────
        st.markdown("#### Step 4 · Evaluate Each Lagrange Term")
        term_vals: list[tuple[float, float]] = []   # (L_i, y_i*L_i)
        for i in range(n_l):
            num_v = 1.0
            den_v = 1.0
            for j in range(n_l):
                if j == i:
                    continue
                num_v *= (t_l - xs_l[j])
                den_v *= (xs_l[i] - xs_l[j])
            L_i  = num_v / den_v
            yL_i = ys_l[i] * L_i
            term_vals.append((L_i, yL_i))
        eval_rows = [
            rf"L_{{{i}}}({_fmt_num_l(t_l)})\,y_{{{i}}} &"
            rf"=\; ({L_i:+.6f})\,({_fmt_num_l(ys_l[i])})"
            rf" \;=\; {yL_i:+.6f}"
            for i, (L_i, yL_i) in enumerate(term_vals)
        ]
        st.latex(r"\begin{aligned}" + r" \\ ".join(eval_rows) + r"\end{aligned}")

        # ── Step 5 · Final boxed answer ──────────────────────────
        st.markdown("#### Step 5 · Sum the Terms — Final Answer")
        sum_str = " \\;+\\; ".join(f"({yL_i:+.6f})" for _, yL_i in term_vals)
        st.latex(
            rf"f({_fmt_num_l(t_l)}) \;=\; {sum_str}"
            rf" \;=\; \boxed{{{result_l:.4f}}}"
        )
        st.success(
            f"**The Lagrange-interpolated value is "
            f"$f({t_l:g}) \\approx {result_l:.4f}$.**"
        )

        # ── HUE lecture cross-check footnote (transparent about slide typo) ──
        if (n_l == 4
            and xs_l == [5.0, 6.0, 9.0, 11.0]
            and ys_l == [380.0, -2.0, 196.0, 508.0]
            and abs(t_l - 10.0) < 1e-9):
            st.info(
                "📘  **Lecture cross-check.**  The HUE slide for this exact "
                "example ends with `= 330`, but the four basis terms above "
                "actually sum to "
                "$\\tfrac{1}{6}(380) + (-\\tfrac{1}{3})(-2)"
                " + \\tfrac{5}{6}(196) + \\tfrac{1}{3}(508) "
                "= \\tfrac{2380}{6} = 396.\\overline{6}$.  "
                "The `= 330` printed on the slide appears to be an arithmetic "
                "slip — the **correct** Lagrange value at $x = 10$ is "
                f"$\\mathbf{{{result_l:.4f}}}$."
            )

        with st.expander("Numeric details · basis values"):
            df_lag = pd.DataFrame([
                {"i": i, "x_i": xs_l[i], "y_i": ys_l[i],
                 "L_i(x)": L_i, "y_i · L_i": yL_i}
                for i, (L_i, yL_i) in enumerate(term_vals)
            ])
            st.dataframe(
                df_lag.style.format({
                    "x_i":         "{:g}",
                    "y_i":         "{:g}",
                    "L_i(x)":      "{:+.10f}",
                    "y_i · L_i":   "{:+.10f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    else:
        # Coloured monospaced output
        st.markdown(res.get("steps_html", ""), unsafe_allow_html=True)

        # Iteration DataFrames for root-finding methods
        iter_df = res.get("iter_df")
        if iter_df is not None and not iter_df.empty:
            st.markdown("#### Iteration History")
            st.dataframe(
                iter_df.style.format({
                    c: "{:.8f}" for c in iter_df.select_dtypes("float").columns
                    if c not in ("Error", "|f(c)|")
                }).format({
                    c: "{:.4e}" for c in ["Error", "|f(c)|"]
                    if c in iter_df.columns
                }),
                use_container_width=True,
            )

        # Difference table for interpolation
        diff = res.get("diff")
        if diff is not None:
            st.markdown("#### Forward Difference Table")
            df_diff = _diff_df(diff)
            st.dataframe(
                df_diff.style.format(
                    {c: "{:.5f}" for c in df_diff.columns},
                    na_rep="—",
                ),
                use_container_width=True,
            )


# ─── TAB 3: Visualization ────────────────────────────────────────────────────
with tab_vis:
    if res is None:
        st.info("📈  Run a calculation to see the interactive chart.")

    elif res["kind"] == "error":
        st.warning("No chart available — please fix the error first.")

    else:
        fig = res.get("fig")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No visualisation available for this method.")

        # Per-method walkthrough of what the chart shows + how to drive it
        if res["kind"] == "false_position":
            st.markdown("---")
            st.markdown(
                "**Reading the chart**  ·  the **cyan curve** is $f(x)$; the "
                "**blue / violet dots** are the bracket endpoints $a$ and $b$; "
                "the **amber dashed chord** connects $\\bigl(a, f(a)\\bigr)$ and "
                "$\\bigl(b, f(b)\\bigr)$; the **red diamond** on the x-axis is "
                "the new estimate $c = \\dfrac{a\\,f(b) - b\\,f(a)}{f(b) - f(a)}$.\n\n"
                "**Animation controls**  ·  press  **▶ Play**  to step through "
                "every Regula-Falsi iteration, or drag the slider to jump to "
                "a specific step.  The annotation above the chart shows the "
                "current $a$, $b$, $c$, and step error $|\\Delta c|$."
            )

        elif res["kind"] == "newton_raphson":
            st.markdown("---")
            st.markdown(
                "**Reading the chart**  ·  the **cyan curve** is $f(x)$; the "
                "**violet dot** is the current iterate $x_{n-1}$; the "
                "**amber dashed line** is the tangent "
                "$y = f(x_{n-1}) + f'(x_{n-1})(x - x_{n-1})$; the "
                "**red diamond** marks where the tangent crosses the x-axis — "
                "that is the next iterate $x_n$.\n\n"
                "**Animation controls**  ·  use  **▶ Play**  or the slider to "
                "watch the tangent rotate and the estimate refine each step.  "
                "Quadratic convergence means the visible step size shrinks "
                "very fast — usually 3 – 4 frames is enough."
            )

        elif res["kind"] == "gauss_seidel":
            st.markdown("---")
            st.markdown(
                "**Top panel (log scale)**  ·  $\\|A\\mathbf{x} - \\mathbf{b}\\|_\\infty$ "
                "at every iteration; a straight downward line means geometric "
                "(linear) convergence.\n\n"
                "**Bottom panel**  ·  one trace per solution component "
                "$x_i$ — you can watch each component settle towards its "
                "fixed point.  Click items in the legend to focus on one "
                "component at a time.\n\n"
                "**Animation controls**  ·  drag the slider or press "
                "**▶ Play** to move the **red diamond cursor** through the "
                "iterations on both panels simultaneously."
            )

        elif res["kind"] == "interpolation":
            algo = res["algo"]
            base_msg = (
                f"**Reading the chart**  ·  the **cyan curve** is the {algo} "
                "polynomial; the **green dots** are your input data; the "
                "**red star** is the requested interpolated value "
                f"$f({res['t']}) \\approx {res['result']:.6f}$.  Hover any "
                "point or curve segment to see exact coordinates."
            )
            extras = ""
            if "Stirling" in algo or "Forward" in algo:
                extras = (
                    "  The **gold ring** highlights $x_0$ — the table origin "
                    "used by the formula (first point for Newton-Forward, "
                    "central point for Stirling)."
                )
            elif "Lagrange" in algo:
                extras = (
                    "  Click **L₀(x), L₁(x), …** in the legend to overlay the "
                    "individual basis polynomials and see how they sum, "
                    "weighted by $y_i$, into the final interpolant."
                )
            st.markdown("---")
            st.markdown(base_msg + extras)

        elif res["kind"] == "doolittle":
            st.markdown("---")
            st.markdown(
                "**Heatmap**  ·  cell shade encodes $|L_{ij}|$ / $|U_{ij}|$; "
                "every cell is also labelled with its exact numeric value.  "
                "**L** is lower-triangular (zeros above the diagonal, ones on "
                "the diagonal — Doolittle convention) and **U** is "
                "upper-triangular.  Hover a cell to read its full-precision "
                "value."
            )