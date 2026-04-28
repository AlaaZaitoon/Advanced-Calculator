"""
app.py — Advanced Numerical Methods Calculator (Streamlit Edition)
===================================================================
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
import traceback
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import calculator
except ImportError as exc:
    st.error(
        "❌  **calculator.py not found.**  "
        "Place `calculator.py` in the same directory as `app.py`."
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be the first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Calculus Engine",
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
    /* ── Import Fira Code for monospace output ── */
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&display=swap');

    /* ── Root accent colour ── */
    :root { --accent: #007ACC; --accent-dim: #004F88; --ok: #4EC994; --warn: #FF9800; --err: #FF5555; }

    /* ── Sidebar refinements ── */
    [data-testid="stSidebar"] { border-right: 1px solid #2B2B2B; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: #A0A0A0; }

    /* ── Tab strip ── */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #2B2B2B; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 6px 18px; font-size: 0.85rem; }
    .stTabs [aria-selected="true"] { background: #007ACC22; border-bottom: 2px solid #007ACC !important; }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #1E1E2E;
        border: 1px solid #2B2B2B;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="metric-container"] [data-testid="stMetricLabel"] { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: #A0A0A0; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; }

    /* ── Data editor / dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    /* ── Code / pre blocks in step-by-step ── */
    .calc-steps {
        font-family: 'Fira Code', monospace;
        font-size: 0.83rem;
        line-height: 1.65;
        background: #141420;
        border: 1px solid #2B2B2B;
        border-radius: 10px;
        padding: 18px 22px;
        overflow-x: auto;
        white-space: pre;
        color: #D4D4D4;
    }
    .calc-steps .hdr  { color: #007ACC; font-weight: 700; }
    .calc-steps .val  { color: #4EC994; }
    .calc-steps .num  { color: #B5CEA8; }
    .calc-steps .key  { color: #9CDCFE; }
    .calc-steps .warn { color: #FF9800; }
    .calc-steps .err  { color: #FF5555; font-weight: 700; }
    .calc-steps .dim  { color: #606060; }
    .calc-steps .form { color: #C586C0; font-style: italic; }
    .calc-steps .yel  { color: #DCDCAA; font-weight: 700; }

    /* ── Method info pill ── */
    .method-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        margin-right: 8px;
    }
    .pill-direct        { background: #4EC99420; border: 1px solid #4EC994; color: #4EC994; }
    .pill-iterative     { background: #FF980020; border: 1px solid #FF9800; color: #FF9800; }
    .pill-bracketing    { background: #00FFFF20; border: 1px solid #00FFFF; color: #00FFFF; }
    .pill-open          { background: #C586C020; border: 1px solid #C586C0; color: #C586C0; }
    .pill-interpolation { background: #007ACC20; border: 1px solid #007ACC; color: #007ACC; }

    /* ── Calculate button ── */
    .stButton > button[kind="primary"] {
        width: 100%;
        background: linear-gradient(135deg, #007ACC, #005999);
        border: none;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        padding: 12px 0;
        transition: opacity 0.15s;
    }
    .stButton > button[kind="primary"]:hover { opacity: 0.88; }

    /* ── Section headers ── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #A0A0A0;
        margin: 18px 0 6px 0;
        padding-bottom: 4px;
        border-bottom: 1px solid #2B2B2B;
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
        r"A = LU \;\Rightarrow\; Ly = b \;\Rightarrow\; Ux = y",
        "Factorises **A = LU** then solves by forward & back substitution.",
    ),
    METHODS[1]: (
        "pill-iterative", "O(n²)/iter",
        r"x_i^{(k+1)} = \frac{b_i - \sum_{j \neq i} a_{ij} x_j^{(k)}}{a_{ii}}",
        "Iteratively refines **x** until ‖residual‖ < ε.  Best for SDD matrices.",
    ),
    METHODS[2]: (
        "pill-bracketing", "Linear conv.",
        r"c = \frac{a \cdot f(b) - b \cdot f(a)}{f(b) - f(a)}",
        "Draws a secant across a sign-change interval — also called **Regula Falsi**.",
    ),
    METHODS[3]: (
        "pill-open", "Quadratic conv.",
        r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}",
        "Tangent-line iteration.  Requires **f(x)** and **f′(x)**.  Fast near the root.",
    ),
    METHODS[4]: (
        "pill-interpolation", "O(n²) setup",
        r"P(x) = y_0 + u\Delta y_0 + \frac{u(u-1)}{2!}\Delta^2 y_0 + \cdots",
        "Newton's **forward difference table** — equally spaced nodes only.",
    ),
    METHODS[5]: (
        "pill-interpolation", "O(n²) setup",
        r"P(x) = y_0 + u\mu\delta y_0 + \frac{u^2}{2!}\delta^2 y_0 + \cdots",
        "**Stirling central differences** — best accuracy near the table midpoint.",
    ),
    METHODS[6]: (
        "pill-interpolation", "O(n²) eval",
        r"P(x) = \sum_{i=0}^{n} y_i \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}",
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
    paper_bgcolor="#181818",
    plot_bgcolor="#181818",
    font=dict(family="Fira Code, monospace", size=12, color="#D4D4D4"),
    xaxis=dict(gridcolor="#2B2B2B", zerolinecolor="#3A3A3A", linecolor="#3A3A3A"),
    yaxis=dict(gridcolor="#2B2B2B", zerolinecolor="#3A3A3A", linecolor="#3A3A3A"),
    legend=dict(bgcolor="#1E1E2E", bordercolor="#2B2B2B", borderwidth=1),
    margin=dict(l=50, r=30, t=60, b=50),
    hoverlabel=dict(bgcolor="#1E1E2E", font_size=12),
)


def _fig_root(f, x_lo: float, x_hi: float,
              root: float | None, title: str, eq: str) -> go.Figure:
    """Interactive function-curve plot with optional root marker."""
    xs = np.linspace(x_lo, x_hi, 600)
    try:
        try:
            ys = f(xs)
        except Exception:
            ys = np.array([f(float(x)) for x in xs])
    except Exception:
        ys = np.zeros_like(xs)

    fig = go.Figure()

    # Function curve
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        name=f"f(x) = {eq}",
        line=dict(color="#00FFFF", width=2.5),
        hovertemplate="x = %{x:.5f}<br>f(x) = %{y:.5f}<extra></extra>",
    ))

    # Zero line
    fig.add_hline(y=0, line=dict(color="#606060", width=1, dash="dot"))

    # Root marker
    if root is not None:
        fig.add_vline(x=root, line=dict(color="#FF9800", width=1, dash="dash"),
                      opacity=0.6)
        fig.add_trace(go.Scatter(
            x=[root], y=[0.0],
            mode="markers+text",
            name=f"Root ≈ {root:.6g}",
            marker=dict(color="#FF5555", size=12, symbol="circle",
                        line=dict(color="#FFFFFF", width=1.5)),
            text=[f"  x ≈ {root:.6g}"],
            textposition="top right",
            textfont=dict(color="#FF5555", size=11),
            hovertemplate=f"Root ≈ {root:.10f}<extra></extra>",
        ))

    fig.update_layout(title=dict(text=title, font=dict(size=14)), **_PLOTLY_LAYOUT)
    return fig


def _fig_convergence(residuals: list, title: str) -> go.Figure:
    """Log-scale residual convergence plot."""
    iters = list(range(1, len(residuals) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=residuals,
        mode="lines+markers",
        name="‖residual‖∞",
        line=dict(color="#00FFFF", width=2),
        marker=dict(color="#007ACC", size=6,
                    line=dict(color="#00FFFF", width=1)),
        hovertemplate="Iteration %{x}<br>‖residual‖∞ = %{y:.3e}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Iteration",
        yaxis_title="‖Ax − b‖∞",
        yaxis_type="log",
        **_PLOTLY_LAYOUT,
    )
    return fig


def _fig_interpolation(xs: list, ys: list, t: float,
                       result: float, interp_fn, title: str) -> go.Figure:
    """Interpolant curve + data scatter + target star."""
    margin  = (max(xs) - min(xs)) * 0.12 + 0.5
    x_curve = np.linspace(min(xs) - margin, max(xs) + margin, 400)
    y_curve = []
    for xv in x_curve:
        try:
            y_curve.append(float(interp_fn(float(xv))))
        except Exception:
            y_curve.append(float("nan"))

    fig = go.Figure()

    # Interpolant curve
    fig.add_trace(go.Scatter(
        x=x_curve, y=y_curve,
        mode="lines",
        name="Interpolant",
        line=dict(color="#00FFFF", width=2.5),
        hovertemplate="x = %{x:.4f}<br>f(x) = %{y:.4f}<extra></extra>",
    ))

    # Data points
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        name="Data points",
        marker=dict(color="#4EC994", size=10,
                    line=dict(color="#FFFFFF", width=1.5)),
        hovertemplate="(%{x}, %{y})<extra>Data</extra>",
    ))

    # Target interpolated point (star)
    fig.add_trace(go.Scatter(
        x=[t], y=[result],
        mode="markers+text",
        name=f"f({t}) ≈ {result:.4f}",
        marker=dict(color="#FF5555", size=16, symbol="star",
                    line=dict(color="#FFFFFF", width=1)),
        text=[f"  f({t}) ≈ {result:.4f}"],
        textposition="top right",
        textfont=dict(color="#FF5555", size=11),
        hovertemplate=f"f({t}) ≈ {result:.8f}<extra>Target</extra>",
    ))

    fig.add_vline(x=t, line=dict(color="#FF5555", width=1, dash="dash"),
                  opacity=0.5)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="x",
        yaxis_title="f(x)",
        **_PLOTLY_LAYOUT,
    )
    return fig


def _fig_lu_heatmap(L: np.ndarray, U: np.ndarray) -> go.Figure:
    """Side-by-side heatmaps for L and U factor matrices."""
    from plotly.subplots import make_subplots

    n   = L.shape[0]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Lower Triangular  L", "Upper Triangular  U"],
    )

    kw = dict(
        colorscale="Blues",
        showscale=False,
        text_auto=".3g",
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

    fig.update_layout(
        title="LU Factor Matrices",
        height=320,
        paper_bgcolor="#181818",
        plot_bgcolor="#181818",
        font=dict(color="#D4D4D4"),
        margin=dict(l=30, r=30, t=70, b=30),
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
        "fp_eq":      "x**3 - x - 2",
        "fp_a":       1.0,
        "fp_b":       2.0,
        "fp_tol":     1e-6,
        "fp_max":     100,
        # newton-raphson
        "nr_f":       "x**3 - 2*x - 5",
        "nr_fp":      "3*x**2 - 2",
        "nr_x0":      2.0,
        "nr_tol":     1e-6,
        "nr_max":     100,
        # interpolation
        "interp_xs":  "1981, 1991, 2001, 2011",
        "interp_ys":  "46, 66, 81, 93",
        "interp_t":   1985.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── App identity
    st.markdown(
        """
        <div style='text-align:center; padding: 8px 0 4px 0;'>
          <span style='font-size:2.4rem;'>∑</span><br>
          <span style='font-size:1.05rem; font-weight:700; color:#D4D4D4;'>Calculus Engine</span><br>
          <span style='font-size:0.75rem; color:#A0A0A0;'>Advanced Numerical Methods</span>
        </div>
        <hr style='border-color:#2B2B2B; margin:14px 0;'>
        """,
        unsafe_allow_html=True,
    )

    # ── Method selector
    st.markdown('<div class="section-label">Numerical Method</div>',
                unsafe_allow_html=True)
    method = st.selectbox(
        label="Method",
        options=METHODS,
        index=METHODS.index(st.session_state["method"]),
        label_visibility="collapsed",
        key="method",
    )

    # ── Method info card
    pill_cls, complexity, formula_latex, desc = METHOD_META[method]
    st.markdown(
        f"""
        <div style='background:#1A1A28; border:1px solid #2B2B2B;
                    border-radius:10px; padding:12px 14px; margin:10px 0;'>
          <span class="method-pill {pill_cls}">{pill_cls.replace("pill-", "").title()}</span>
          <span style='font-size:0.75rem; color:#606060;'>complexity: {complexity}</span>
          <p style='margin:8px 0 0 0; font-size:0.8rem; color:#A0A0A0;'>{desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.latex(formula_latex)

    st.markdown('<hr style="border-color:#2B2B2B; margin:14px 0;">', unsafe_allow_html=True)

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

        if "Gauss" in method:
            st.markdown('<div class="section-label">Solver Settings</div>',
                        unsafe_allow_html=True)
            gs_tol = st.number_input(
                "Tolerance", value=st.session_state["gs_tol"],
                format="%.2e", min_value=1e-15, max_value=1e-1, key="gs_tol",
            )
            gs_max = st.number_input(
                "Max Iterations", value=st.session_state["gs_max"],
                min_value=10, max_value=10000, step=50, key="gs_max",
            )

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

        st.markdown('<div class="section-label">Solver Settings</div>',
                    unsafe_allow_html=True)
        fp_tol = st.number_input("Tolerance", value=st.session_state["fp_tol"],
                                 format="%.2e", min_value=1e-15, max_value=1e-1,
                                 key="fp_tol")
        fp_max = st.number_input("Max Iterations", value=st.session_state["fp_max"],
                                 min_value=5, max_value=1000, step=10, key="fp_max")

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
        nr_tol = st.number_input("Tolerance", value=st.session_state["nr_tol"],
                                  format="%.2e", min_value=1e-15, max_value=1e-1,
                                  key="nr_tol")
        nr_max = st.number_input("Max Iterations", value=st.session_state["nr_max"],
                                  min_value=5, max_value=1000, step=10, key="nr_max")

    else:  # interpolation family
        st.markdown('<div class="section-label">Dataset</div>',
                    unsafe_allow_html=True)
        st.write("$X$ values (comma-separated):")
        interp_xs = st.text_input("X", value=st.session_state["interp_xs"],
                                   label_visibility="collapsed", key="interp_xs")
        st.write("$Y = f(X)$ values (comma-separated):")
        interp_ys = st.text_input("Y", value=st.session_state["interp_ys"],
                                   label_visibility="collapsed", key="interp_ys")

        st.markdown('<div class="section-label">Target Point</div>',
                    unsafe_allow_html=True)
        st.write("Interpolate at $x =$")
        interp_t = st.number_input("x", value=st.session_state["interp_t"],
                                    label_visibility="collapsed",
                                    key="interp_t", format="%.4f")

    # ── Calculate button
    st.markdown("<br>", unsafe_allow_html=True)
    calculate = st.button("▶  Calculate", type="primary", use_container_width=True)

    st.markdown(
        '<div style="text-align:center; font-size:0.68rem; color:#606060; '
        'margin-top:18px;">v2.1 · Antigravity · Streamlit Edition</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA — page title + method header
# ══════════════════════════════════════════════════════════════════════════════
col_title, col_method = st.columns([2, 3])
with col_title:
    st.markdown(
        "<h1 style='font-size:1.7rem; font-weight:800; margin-bottom:4px;'>"
        "Calculus Engine</h1>"
        "<p style='color:#A0A0A0; font-size:0.85rem; margin-top:0;'>"
        "Advanced Numerical Methods · Web Dashboard</p>",
        unsafe_allow_html=True,
    )
with col_method:
    pill_cls, complexity, _, _ = METHOD_META[method]
    st.markdown(
        f"""
        <div style='background:#1A1A28; border:1px solid #2B2B2B;
                    border-radius:10px; padding:10px 16px; margin-top:8px;'>
          <span style='font-size:0.82rem; font-weight:600; color:#D4D4D4;'>{method}</span>
          <span class="method-pill {pill_cls}" style='float:right; margin-top:2px;'>
            {complexity}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  MATRIX INPUT AREA  (only for Doolittle / Gauss-Seidel)
# ══════════════════════════════════════════════════════════════════════════════
A_arr = B_arr = X0_arr = None

if "Doolittle" in method or "Gauss" in method:
    n = st.session_state["matrix_n"]

    # ── Pre-fill data based on method
    if "Doolittle" in method:
        default_A = [[2, -1, -2], [-4, 6, 3], [-4, -2, 8]]
        default_B = [3, -8, 12]
        default_X0 = None
    else:
        default_A  = [[4, 1, -1], [2, 7, 1], [1, -3, 12]]
        default_B  = [3, 19, 31]
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
            # ─────────────── Doolittle ───────────────────────────────────
            if "Doolittle" in method:
                L, U, x = calculator.doolittle_lu_decomposition(A_arr, B_arr)
                y   = np.linalg.solve(L, B_arr)
                res = np.linalg.norm(A_arr @ x - B_arr, np.inf)

                rows: list[tuple[str, str]] = []
                rows += _hdr("Doolittle LU Decomposition") + [("", "")]
                rows += [("  Strategy:  A = LU  →  Ly = b  →  Ux = y", "key"), ("", "")]
                rows += _hdr("L  (lower triangular)")
                rows += [(_fmt_matrix(L), "num"), ("", "")]
                rows += _hdr("U  (upper triangular)")
                rows += [(_fmt_matrix(U), "num"), ("", "")]
                rows += _hdr("y  (forward sub)")
                rows += [(_fmt_matrix(y), "num"), ("", "")]
                rows += _hdr("x  (solution)")
                rows += [(_fmt_matrix(x), "val"), ("", "")]
                rows += [_div(), (f"  ‖Ax − b‖∞  =  {res:.6e}", "num")]
                for i, v in enumerate(x):
                    rows.append((f"  x{i+1}  =  {v:+.10f}", "val"))

                result_payload = dict(
                    kind="doolittle",
                    x=x, L=L, U=U, res=res,
                    steps_html=_steps_html(rows),
                    fig=_fig_lu_heatmap(L, U),
                    n=len(x),
                )

            # ─────────────── Gauss-Seidel ────────────────────────────────
            elif "Gauss" in method:
                tol = float(st.session_state["gs_tol"])
                mx  = int(st.session_state["gs_max"])
                sol, iters, converged = calculator.gauss_seidel(
                    A_arr, B_arr, X0_arr, tol, mx)
                sdd = calculator.is_strictly_diagonally_dominant(A_arr)
                n   = len(B_arr)

                # Collect residuals
                xc, residuals = (X0_arr if X0_arr is not None else np.zeros(n)).copy(), []
                diag_inv = 1.0 / np.diag(A_arr)
                for _ in range(min(iters + 1, mx)):
                    xn = xc.copy()
                    for i in range(n):
                        sigma = A_arr[i, :i] @ xn[:i] + A_arr[i, i+1:] @ xn[i+1:]
                        xn[i] = (B_arr[i] - sigma) * diag_inv[i]
                    rv = np.linalg.norm(A_arr @ xn - B_arr, np.inf)
                    residuals.append(float(rv))
                    if rv < tol:
                        break
                    xc = xn

                res_fin = np.linalg.norm(A_arr @ sol - B_arr, np.inf)

                rows = []
                rows += _hdr("Gauss-Seidel Iterative Method") + [("", "")]
                rows += [
                    (f"  SDD Check   : {'PASS ✅' if sdd else 'FAIL ⚠  (convergence not guaranteed)'}",
                     "val" if sdd else "warn"),
                    (f"  Tolerance   : {tol}", "key"),
                    (f"  Iterations  : {iters}", "num"),
                    (f"  Converged   : {'YES' if converged else 'NO'}",
                     "val" if converged else "warn"),
                    ("", ""),
                ]
                rows += _hdr("Solution  x")
                rows += [(_fmt_matrix(sol), "val"), ("", "")]
                rows += _hdr("Residual")
                rows += [
                    (f"  ‖Ax − b‖₂  =  {np.linalg.norm(A_arr@sol-B_arr):.6e}", "num"),
                    (f"  ‖Ax − b‖∞  =  {res_fin:.6e}", "num"),
                    ("", ""),
                ]
                rows += _hdr("Diagonal Dominance (row-by-row)")
                for i in range(n):
                    diag = abs(A_arr[i, i])
                    off  = sum(abs(A_arr[i, j]) for j in range(n) if j != i)
                    ok_r = diag > off
                    rows.append((
                        f"  Row {i+1}: |{A_arr[i,i]:.4f}| = {diag:.4f} "
                        f"{'>' if ok_r else '≤'} {off:.4f}  {'✓' if ok_r else '✗'}",
                        "val" if ok_r else "warn",
                    ))

                result_payload = dict(
                    kind="gauss_seidel",
                    sol=sol, iters=iters, converged=converged,
                    sdd=sdd, res_fin=res_fin,
                    steps_html=_steps_html(rows),
                    fig=_fig_convergence(
                        residuals,
                        f"Gauss-Seidel Convergence  ({len(residuals)} iters)"),
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

                if f(a) * f(b) > 0:
                    raise ValueError(
                        f"f(a)·f(b) > 0 — no sign change on [{a}, {b}].  "
                        "Choose an interval that brackets a root.")

                root, iters = calculator.false_position(f, a, b, tol, mx)
                f_root = f(root)

                log = []
                ac, bc = a, b
                fac, fbc = f(ac), f(bc)
                for k in range(min(mx, 40)):
                    if abs(fbc - fac) < 1e-14: break
                    cc  = bc - fbc * (bc - ac) / (fbc - fac)
                    fcc = f(cc)
                    log.append((k + 1, ac, bc, cc, fcc, abs(fcc)))
                    if abs(fcc) < tol: break
                    if fac * fcc < 0: bc, fbc = cc, fcc
                    else:              ac, fac = cc, fcc

                rows = []
                rows += _hdr("Method of False Position  (Regula Falsi)") + [("", "")]
                rows += [
                    (f"  f(x)      =  {eq}", "key"),
                    (f"  Interval  :  [{a},  {b}]", "key"),
                    (f"  f({a})    =  {f(a):.8f}", "num"),
                    (f"  f({b})    =  {f(b):.8f}", "num"),
                    (f"  Tolerance =  {tol}", "dim"),
                    ("", ""),
                ]
                rows += _hdr("Result")
                rows += [
                    (f"  Root    x  ≈  {root:.12f}", "val"),
                    (f"  f(root)    =  {f_root:.6e}", "num"),
                    (f"  Iterations =  {iters}", "num"),
                    ("", ""),
                    ("  Formula:  c = b − f(b)·(b−a) / (f(b)−f(a))", "form"),
                ]

                margin = (b - a) * 0.35
                result_payload = dict(
                    kind="false_position",
                    root=root, iters=iters, f_root=f_root,
                    steps_html=_steps_html(rows),
                    iter_df=_iter_df_fp(log),
                    fig=_fig_root(f, a - margin, b + margin,
                                  root, f"Regula Falsi — {eq}", eq),
                    eq=eq,
                )

            # ─────────────── Newton-Raphson ───────────────────────────────
            elif "Newton-Raph" in method:
                eq    = str(st.session_state["nr_f"])
                fp_eq = str(st.session_state["nr_fp"])
                x0    = float(st.session_state["nr_x0"])
                tol   = float(st.session_state["nr_tol"])
                mx    = int(st.session_state["nr_max"])
                f     = _make_f(eq)
                fp    = _make_f(fp_eq)

                root, iters = calculator.newton_raphson(f, fp, x0, tol, mx)

                log = []
                xc = x0
                for k in range(min(mx, 25)):
                    fxc  = f(xc)
                    fpxc = fp(xc)
                    if abs(fpxc) < 1e-14: break
                    xn  = xc - fxc / fpxc
                    err = abs(xn - xc)
                    log.append((k + 1, xc, fxc, fpxc, xn, err))
                    if err < tol: break
                    xc = xn

                rows = []
                rows += _hdr("Newton-Raphson Method") + [("", "")]
                rows += [
                    (f"  f(x)      =  {eq}", "key"),
                    (f"  f′(x)     =  {fp_eq}", "key"),
                    (f"  x₀        =  {x0}", "num"),
                    (f"  Tolerance =  {tol}", "dim"),
                    ("", ""),
                ]
                rows += _hdr("Result")
                rows += [
                    (f"  Root    x  ≈  {root:.12f}", "val"),
                    (f"  f(root)    =  {f(root):.6e}", "num"),
                    (f"  f′(root)   =  {fp(root):.6e}", "num"),
                    (f"  Iterations =  {iters}", "num"),
                    ("", ""),
                    ("  Formula:  xₙ₊₁ = xₙ − f(xₙ) / f′(xₙ)", "form"),
                ]

                span = abs(root - x0) * 1.6 + 1.5
                result_payload = dict(
                    kind="newton_raphson",
                    root=root, iters=iters,
                    steps_html=_steps_html(rows),
                    iter_df=_iter_df_nr(log),
                    fig=_fig_root(f, root - span, root + span,
                                  root, f"Newton-Raphson — {eq}", eq),
                    eq=eq, fp_eq=fp_eq,
                )

            # ─────────────── Interpolation family ────────────────────────
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
                elif "Stirling" in method:
                    result, diff = calculator.stirling_interpolation(xs, ys, t)
                    algo = "Stirling's Central Difference"
                    interp_fn = lambda xv, _xs=xs, _ys=ys: (
                        calculator.stirling_interpolation(_xs, _ys, xv)[0])
                else:
                    result    = calculator.lagrange_interpolation(xs, ys, t)
                    diff      = None
                    algo      = "Lagrange's Polynomial"
                    interp_fn = lambda xv, _xs=xs, _ys=ys: (
                        calculator.lagrange_interpolation(_xs, _ys, xv))

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
                    fig=_fig_interpolation(xs, ys, t, result, interp_fn,
                                           f"{algo}  —  f({t}) ≈ {result:.4f}"),
                    xs=xs, ys=ys,
                )

            st.session_state["result"] = result_payload

        except Exception as exc:
            st.session_state["result"] = dict(
                kind="error",
                message=str(exc),
                traceback=traceback.format_exc(),
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
        x, res_val, n_val = res["x"], res["res"], res["n"]
        st.success("✅  **LU Decomposition Solved**")
        cols = st.columns(min(n_val, 4))
        for i, v in enumerate(x):
            with cols[i % len(cols)]:
                st.metric(label=f"x{i+1}", value=f"{v:+.6f}")
        st.markdown("---")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("System size", f"{n_val} × {n_val}")
        with mc2:
            st.metric("Residual ‖Ax−b‖∞", f"{res_val:.3e}")

        st.markdown("**Strategy:**")
        st.latex(r"A = LU \quad\Rightarrow\quad Ly = b \quad\Rightarrow\quad Ux = y")

    elif res["kind"] == "gauss_seidel":
        sol, converged = res["sol"], res["converged"]
        n_val = res["n"]
        if converged:
            st.success(f"✅  **Gauss-Seidel Converged** in {res['iters']} iterations")
        else:
            st.warning(f"⚠  **Max iterations reached** ({res['iters']})")

        cols = st.columns(min(n_val, 4))
        for i, v in enumerate(sol):
            with cols[i % len(cols)]:
                st.metric(label=f"x{i+1}", value=f"{v:+.6f}")
        st.markdown("---")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Iterations", res["iters"])
        with mc2:
            st.metric("SDD", "✓ Pass" if res["sdd"] else "✗ Fail")
        with mc3:
            st.metric("‖Ax−b‖∞", f"{res['res_fin']:.3e}")

    elif res["kind"] == "false_position":
        root, iters = res["root"], res["iters"]
        st.success("✅  **Root Found — Regula Falsi**")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Root  x ≈", f"{root:.8f}")
        with mc2:
            st.metric("f(root)", f"{res['f_root']:.4e}")
        with mc3:
            st.metric("Iterations", iters)
        st.markdown("**Formula:**")
        st.latex(r"c = \frac{a \cdot f(b) - b \cdot f(a)}{f(b) - f(a)}")
        st.markdown(f"**Equation:** $f(x) = {res['eq']}$")

    elif res["kind"] == "newton_raphson":
        root, iters = res["root"], res["iters"]
        st.success("✅  **Root Found — Newton-Raphson**")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Root  x ≈", f"{root:.8f}")
        with mc2:
            st.metric("Iterations", iters)
        st.markdown("**Newton-Raphson Formula:**")
        st.latex(r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}")
        st.markdown(f"$f(x) = {res['eq']}$  ·  $f'(x) = {res['fp_eq']}$")

    elif res["kind"] == "interpolation":
        result, algo, t = res["result"], res["algo"], res["t"]
        st.success(f"✅  **{algo}**")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric(f"f({t}) ≈", f"{result:.8f}")
        with mc2:
            st.metric("Data points", len(res["xs"]))

        st.markdown(f"**Target:** $x = {t}$  →  $f({t}) \\approx {result:.6f}$")

        if "Forward" in res["algo"]:
            st.latex(r"P(x) = y_0 + u\Delta y_0 + \frac{u(u-1)}{2!}\Delta^2 y_0 + \cdots")
        elif "Stirling" in res["algo"]:
            st.latex(r"P(x) = y_0 + u\mu\delta y_0 + \frac{u^2}{2}\delta^2 y_0 + \cdots")
        else:
            st.latex(r"P(x) = \sum_{i=0}^{n} y_i \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}")

# ─── TAB 2: Step-by-Step ─────────────────────────────────────────────────────
with tab_steps:
    if res is None:
        st.info("Results will appear here after calculating.")

    elif res["kind"] == "error":
        st.error(res["message"])
        st.code(res["traceback"], language="python")

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

        # LU factor matrices as DataFrames
        if res["kind"] == "doolittle":
            st.markdown("#### Factor Matrices")
            col_l, col_u = st.columns(2)
            with col_l:
                st.markdown("**L  (lower triangular)**")
                df_L = pd.DataFrame(
                    res["L"],
                    columns=[f"c{i+1}" for i in range(res["n"])],
                )
                st.dataframe(df_L.style.format("{:.6f}"),
                             use_container_width=True)
            with col_u:
                st.markdown("**U  (upper triangular)**")
                df_U = pd.DataFrame(
                    res["U"],
                    columns=[f"c{i+1}" for i in range(res["n"])],
                )
                st.dataframe(df_U.style.format("{:.6f}"),
                             use_container_width=True)

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

        # Additional context for each method
        if res["kind"] == "false_position":
            st.markdown("---")
            st.markdown(
                "**How Regula Falsi works:** The algorithm draws a straight line (secant) "
                "between the two bracket points $(a, f(a))$ and $(b, f(b))$ and takes "
                "the x-intercept $c$ as the new estimate.  The bracket is then updated "
                "to maintain the sign change.  The **red dot** marks the converged root."
            )

        elif res["kind"] == "newton_raphson":
            st.markdown("---")
            st.markdown(
                "**How Newton-Raphson works:** At each step the method draws a **tangent** "
                "to the curve at the current point and takes its x-intercept as the next "
                "estimate.  Convergence is **quadratic** near the root — each step roughly "
                "doubles the number of correct digits."
            )

        elif res["kind"] == "gauss_seidel":
            st.markdown("---")
            st.markdown(
                "**Convergence plot (log scale):** The y-axis shows the infinity-norm "
                "residual ‖Ax − b‖∞ at each iteration.  A straight line on this plot "
                "means geometric (linear) convergence."
            )

        elif res["kind"] == "interpolation":
            st.markdown("---")
            st.markdown(
                f"**Chart:** the **cyan curve** is the {res['algo']} polynomial, "
                "**green dots** are your data points, and the **red star** marks "
                f"the interpolated value $f({res['t']}) \\approx {res['result']:.6f}$."
            )

        elif res["kind"] == "doolittle":
            st.markdown("---")
            st.markdown(
                "**Heatmap:** darker cells indicate larger absolute values.  "
                "Notice that **L** is lower-triangular (zeros above diagonal) "
                "and **U** is upper-triangular (zeros below diagonal)."
            )