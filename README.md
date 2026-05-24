# Advanced Numerical Calculator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hue-numerical-methods.streamlit.app/)

> An interactive web dashboard for seven classical numerical-methods algorithms,
> built on Streamlit + Plotly with animated, lecture-accurate visualisations.
>
> **Live Demo:** [hue-numerical-methods.streamlit.app](https://hue-numerical-methods.streamlit.app/)


The Advanced Numerical Calculator turns a textbook numerical-methods syllabus
into a hands-on, browser-based laboratory.  Enter a function or matrix, watch
each algorithm converge frame-by-frame, and step through every intermediate
quantity in a step-by-step LaTeX walkthrough — all wrapped in a calm,
academic-grade dark theme.

---

## Highlights

- **Seven full-fidelity algorithms** — direct, iterative, root-finding and
  interpolation methods, each implemented from scratch in pure NumPy.
- **Animated convergence** — Plotly `frames` with play / pause buttons and
  per-iteration sliders for every iterative method.
- **Lecture-style step-by-step view** — every intermediate matrix, difference
  table, basis polynomial and substitution is rendered with KaTeX.
- **Modern UI** — *Calm Midnight* dark theme (deep navy + soft cyan / muted
  violet accents), icon-driven sidebar navigation, glass-morphism cards.
- **Pure-Python core** — the numerical engine in `calculator.py` has zero UI
  dependencies and can be unit-tested or reused in scripts and notebooks.

---

## Methods Supported

| # | Method | Type | Formula |
|---|---|---|---|
| 1 | Doolittle's LU Decomposition | Direct linear solver | `A = LU  ⇒  Ly = b,  Ux = y` |
| 2 | Gauss-Seidel Iteration | Iterative linear solver | `xᵢ⁽ᵏ⁺¹⁾ = (bᵢ − Σⱼ≠ᵢ aᵢⱼ xⱼ) / aᵢᵢ` |
| 3 | Method of False Position | Bracketing root-finder | `c = (a·f(b) − b·f(a)) / (f(b) − f(a))` |
| 4 | Newton-Raphson | Open root-finder | `xₙ = xₙ₋₁ − f(xₙ₋₁) / f'(xₙ₋₁)` |
| 5 | Newton's Forward Interpolation | Equally-spaced data | `f(x) = y₀ + p·Δy₀ + p(p−1)/2!·Δ²y₀ + …` |
| 6 | Stirling's Central Difference Interpolation | Equally-spaced data | `f(x) = y₀ + p·μδy₀ + p²/2!·δ²y₀ + …` |
| 7 | Lagrange Interpolation | Unequally-spaced data | `f(x) = Σᵢ yᵢ · ∏ⱼ≠ᵢ (x − xⱼ) / (xᵢ − xⱼ)` |

Every method is validated against textbook examples; the regression suite
asserts the four lecture targets `f(1991) = 104.9300`,
`f(28) = 46724.0128`, `f(10) = 396.6667`, and Doolittle’s `X = [1, 3, 5]`
to at least four decimal places.

---

## Quick Start

### Prerequisites

- **Python 3.10 +** (3.11 or 3.12 recommended)
- **pip** (ships with Python)
- A modern browser (Chrome, Edge, Firefox, Safari) for the Streamlit UI

### Install

```bash
git clone https://github.com/<your-account>/advanced-numerical-calculator.git
cd advanced-numerical-calculator

python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically <http://localhost:8501>) — open it
in your browser to use the calculator.

> **Tip** &nbsp; If port 8501 is already in use, pass another:
> `streamlit run app.py --server.port 8505`

---

## Using the Application

1. **Pick a method** in the left sidebar (icons + short labels).
2. **Fill the inputs** — matrix cells, function expression, bracket
   endpoints, or comma-separated data points (defaults are pre-populated
   with lecture examples).
3. Click **▶ Calculate**.
4. Inspect the three output tabs:
   - **📊 Summary** — final values, complexity pill, formula recap.
   - **📝 Step-by-Step** — full LaTeX walkthrough, including difference
     tables, forward / back substitutions, and per-iteration logs.
   - **📈 Visualization** — interactive Plotly chart with play / pause
     animations for iterative methods, gold-ring origin markers for
     interpolation methods, and a toggleable basis-polynomial overlay
     for Lagrange.

---

## Project Structure

```
advanced-numerical-calculator/
├── .streamlit/
│   └── config.toml          # Calm Midnight theme + runtime flags
├── app.py                   # Streamlit UI: sidebar, tabs, charts, walkthroughs
├── calculator.py            # Pure-NumPy numerical engine (zero UI deps)
├── requirements.txt         # Pinned Python dependencies
├── README.md                # You are here
└── .gitignore
```

`calculator.py` exposes eight functions (`doolittle_lu_decomposition`,
`gauss_seidel`, `is_strictly_diagonally_dominant`, `false_position`,
`newton_raphson`, `newton_forward_interpolation`, `stirling_interpolation`,
`lagrange_interpolation`) that can be imported and used independently of
the Streamlit front-end.

---

## Theme — *Calm Midnight*

| Token | Hex | Usage |
|---|---|---|
| Background | `#0B1220` | Page backdrop (deep navy) |
| Elevated surface | `#0F1A2E` | Sidebar, cards, chart paper |
| Card surface | `#131F38` | Method info cards, hover labels |
| Border | `#1F2D4A` | Card outlines, axis lines |
| Primary accent | `#7DD3FC` | Buttons, interpolant curves, headings |
| Secondary accent | `#A78BFA` | Iterates, residual markers |
| Text | `#E6EDF7` | Body copy |
| Muted | `#A8B5CC` | Captions, ticks |
| Success / warn / error | `#34D399` / `#FBBF24` / `#F87171` | Data points, tangents, roots |

Typography: **Inter** for the UI, **JetBrains Mono** for code, metrics, and
iteration counters.  Both are loaded from Google Fonts via the inline CSS in
`app.py`.

---

## Dependencies

```
streamlit              ≥ 1.36, < 2
streamlit-option-menu  ≥ 0.3.13
numpy                  ≥ 1.24
pandas                 ≥ 2.0
plotly                 ≥ 5.17
```

`streamlit-option-menu` is optional — the sidebar gracefully falls back to a
plain `st.selectbox` if the package is unavailable.  No other third-party
libraries are required.

---

## Development Notes

- The numerical engine never raises raw NumPy exceptions; every public
  function validates its inputs and emits a descriptive `ValueError`
  carrying the offending shape, value, or row index.
- Iterative algorithms return a `history` payload describing every
  intermediate state so the UI can build animated frames without
  re-running the algorithm.
- The Streamlit layer never mutates `calculator.py` outputs — chart
  builders read history dictionaries and assemble Plotly figures
  declaratively.

---

## License

Released for academic and educational use.
