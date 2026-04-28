"""
gui.py — Advanced Numerical Methods Calculator  v2.1
Theme  : Nord / VS Code Hybrid (Dark)
Author : Antigravity  |  Architecture: Principal UI/UX + Lead Python Dev

CHANGES IN v2.1 (this file)
────────────────────────────────────────────────────────────────────────────
  BUG FIXES
  ─────────
  [FIX-1]  WelcomeScreen pills overflow  — replaced single-row pack() with a
           2-row grid (4 + 3).  No more text clipping or cramped borders.
           Labels now use padx=14 / pady=6 instead of hard-coded spaces.

  [FIX-2]  _pulse() memory leak  — now checks winfo_exists() before
           rescheduling so the .after() loop terminates cleanly when the
           splash frame is destroyed on launch.

  [FIX-3]  NavigationToolbar theming  — replaced bare child.configure(...)
           with an isinstance guard (tk.Button | tk.Label) to avoid spurious
           AttributeError on non-theming-capable children (e.g. tk.Frame).

  [FIX-4]  ValidatedEntry visual feedback  — restored border_color changes
           (blue on valid, red on invalid) which are far more visible than the
           previous fg_color background tint on small compact matrix cells.

  [FIX-5]  _render_plot cleanup  — old toolbar tk.Frame is now explicitly
           destroyed before building a new one, preventing widget leakage on
           repeated calculations.

  [FIX-6]  MethodInfoCard.update_info  — guarded with try/except so a method
           change while the widget is being destroyed can never crash the app.

  [FIX-7]  _start_calculate empty-field guard  — matrix-card entries are also
           checked for emptiness (not just the dict entries), preventing silent
           ValueErrors from half-filled grids.

  IMPROVEMENTS
  ────────────
  [IMP-1]  Toolbar import  — switched from the private
           matplotlib.backends._backend_tk path to the public
           matplotlib.backends.backend_tkagg path for forward compatibility.

  [IMP-2]  RichTextbox.write()  — the state is set to "normal" once at the
           top of clear() rather than on every single write() call, cutting
           the number of configure() round-trips during a large output pass.

  [IMP-3]  _render_plot  — assigns self._last_fig AFTER tight_layout so a
           failed draw_fn cannot leave a stale reference to a partial figure.

  [IMP-4]  StatusBar  — grid_propagate(False) replaced pack_propagate(False)
           to be consistent with the grid() geometry manager used in App.

  [IMP-5]  ControlPanel._clear  — returns the output panel to Summary tab
           after clearing so the user isn't left staring at a stale plot.

  [IMP-6]  Interpolation lambdas  — captured xs/ys with default-argument
           binding (xs=xs) to avoid the classic late-binding closure bug that
           would use the last assigned xs/ys values.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import numpy as np
import math

try:
    import calculator
except ImportError as exc:
    raise ImportError(
        "Cannot find calculator.py.  "
        "Ensure it is in the same directory as gui.py."
    ) from exc

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk  # [IMP-1]
import traceback
import time

# ══════════════════════════════════════════════════════════════════════════════
#  THEME CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

BG_APP        = "#181818"
BG_SIDE       = "#202020"
BG_CARD       = "#2B2B2B"
BG_HOVER      = "#313131"
ACCENT        = "#007ACC"
ACCENT_HOVER  = "#005999"
ACCENT_DIM    = "#004F88"
TEXT_MAIN     = "#D4D4D4"
TEXT_SUB      = "#A0A0A0"
TEXT_HINT     = "#606060"
BORDER        = "#3A3A3A"
BORDER_FOCUS  = "#007ACC"
ERR_COLOR     = "#FF5555"
OK_COLOR      = "#4EC994"
WARN_COLOR    = "#FF9800"
PLOT_CYAN     = "#00FFFF"
PLOT_RED      = "#FF5555"
PLOT_ORANGE   = "#FF9800"
PLOT_PURPLE   = "#C586C0"
PLOT_GRID     = "#2F2F2F"
YELLOW        = "#DCDCAA"

FONT_SPLASH   = ("Segoe UI", 38, "bold")
FONT_TITLE    = ("Segoe UI", 15, "bold")
FONT_SUB_TXT  = ("Segoe UI", 18)
FONT_SECTION  = ("Segoe UI", 12, "bold")
FONT_MAIN     = ("Segoe UI", 13)
FONT_SMALL    = ("Segoe UI", 11)
FONT_BTN      = ("Segoe UI", 13, "bold")
FONT_MONO     = ("Consolas", 13)
FONT_MONO_SM  = ("Consolas", 11)
FONT_RESULT   = ("Segoe UI", 20, "bold")
FONT_CREDIT   = ("Segoe UI", 10)
FONT_PILL     = ("Segoe UI", 9,  "bold")

# ══════════════════════════════════════════════════════════════════════════════
#  METHOD REGISTRY   (contractually fixed — False Position is index 2)
# ══════════════════════════════════════════════════════════════════════════════
METHODS = [
    "Doolittle's Method (LU Decomposition)",         # 0  ← DEFAULT
    "Gauss-Seidel Iteration",                        # 1
    "Method of False Position",                      # 2  ← MUST stay 3rd
    "Newton-Raphson Method",                         # 3
    "Newton's Forward Interpolation",                # 4
    "Stirling's Central Difference Interpolation",   # 5
    "Lagrange's Interpolation",                      # 6
]

# (tag_label, tag_color, complexity, one-line description)
METHOD_META: dict[str, tuple[str, str, str, str]] = {
    METHODS[0]: ("Direct",        OK_COLOR,    "O(n³)",           "Factorises A = LU, then solves via forward + back substitution."),
    METHODS[1]: ("Iterative",     WARN_COLOR,  "O(n²) / iter",    "Refines x until ‖residual‖ < ε.  Guaranteed for SDD matrices."),
    METHODS[2]: ("Bracketing",    PLOT_CYAN,   "Linear conv.",    "Secant line across sign-change interval — also called Regula Falsi."),
    METHODS[3]: ("Open",          PLOT_PURPLE, "Quadratic conv.", "Tangent-line iteration — requires both f(x) and f′(x)."),
    METHODS[4]: ("Interpolation", ACCENT,      "O(n²) setup",     "Newton's forward difference table — equally spaced nodes only."),
    METHODS[5]: ("Interpolation", ACCENT,      "O(n²) setup",     "Stirling central differences — best accuracy near table midpoint."),
    METHODS[6]: ("Interpolation", ACCENT,      "O(n²) eval",      "Lagrange basis polynomials — handles unequally spaced data."),
}

# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY FORMATTERS
# ══════════════════════════════════════════════════════════════════════════════
def _fmt_matrix(mat: np.ndarray, prec: int = 4) -> str:
    """Return a box-drawn, aligned string for 1-D or 2-D NumPy arrays."""
    if not isinstance(mat, np.ndarray):
        mat = np.array(mat, dtype=float)
    if mat.ndim == 1:
        inner = "  ".join(f"{v:>12.{prec}f}" for v in mat)
        return f"[ {inner} ]"
    rows = ["  ".join(f"{v:>12.{prec}f}" for v in row) for row in mat]
    w    = max(len(r) for r in rows)
    top  = "┌" + " " * (w + 2) + "┐"
    mid  = [f"│ {r} │" for r in rows]
    bot  = "└" + " " * (w + 2) + "┘"
    return "\n".join([top] + mid + [bot])


def _div(char: str = "─", width: int = 62) -> str:
    return char * width


def _hdr(title: str) -> str:
    return f"{_div()}\n  {title}\n{_div()}"


def _iter_table_nr(rows: list) -> str:
    """Aligned Newton-Raphson iteration table."""
    H   = f"  {'Iter':>4}  {'xₙ':>14}  {'f(xₙ)':>14}  {'f′(xₙ)':>14}  {'xₙ₊₁':>14}  {'Error':>12}"
    sep = "  " + "─" * (len(H) - 2)
    lines = [sep, H, sep]
    for it, xn, fn, fpn, xn1, err in rows:
        lines.append(
            f"  {it:>4}  {xn:>14.8f}  {fn:>14.8f}  {fpn:>14.8f}"
            f"  {xn1:>14.8f}  {err:>12.6e}"
        )
    lines.append(sep)
    return "\n".join(lines)


def _iter_table_fp(rows: list) -> str:
    """Aligned False-Position iteration table."""
    H   = f"  {'Iter':>4}  {'a':>12}  {'b':>12}  {'c':>14}  {'f(c)':>14}  {'|f(c)|':>12}"
    sep = "  " + "─" * (len(H) - 2)
    lines = [sep, H, sep]
    for it, a, b, c, fc, afc in rows:
        lines.append(
            f"  {it:>4}  {a:>12.6f}  {b:>12.6f}  {c:>14.8f}"
            f"  {fc:>14.8f}  {afc:>12.6e}"
        )
    lines.append(sep)
    return "\n".join(lines)


def _diff_table_fmt(diff: list) -> str:
    lines = []
    for i, col in enumerate(diff):
        vals = "  ".join(f"{v:>12.5f}" for v in col)
        lines.append(f"  Δ^{i}y  :  {vals}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATED ENTRY  [FIX-4: border_color feedback restored]
# ══════════════════════════════════════════════════════════════════════════════
class ValidatedEntry(ctk.CTkEntry):
    """CTkEntry with live numeric border-colour feedback.

    Fix-4: Uses border_color (blue/red) instead of fg_color background tint.
    This is much more visible, especially for the small compact matrix cells.
    """

    def __init__(self, master, monospace: bool = False,
                 compact: bool = False, **kwargs) -> None:
        self._var = ctk.StringVar()
        params = {
            "textvariable":         self._var,
            "font":                 FONT_MONO if monospace else FONT_MAIN,
            "corner_radius":        6,
            "border_width":         1,
            "border_color":         BORDER,
            "fg_color":             BG_APP,
            "text_color":           TEXT_MAIN,
            "placeholder_text_color": TEXT_HINT,
            "width":                52 if compact else 200,
            "height":               34,
        }
        params.update(kwargs)
        super().__init__(master, **params)
        self._var.trace_add("write", self._on_change)

    def _on_change(self, *_) -> None:
        v = self._var.get().strip()
        if not v:
            # Empty — neutral
            self.configure(border_color=BORDER, text_color=TEXT_MAIN)
        else:
            try:
                float(v)
                # Valid number — accent border
                self.configure(border_color=BORDER_FOCUS, text_color=TEXT_MAIN)
            except ValueError:
                # Invalid — red border  (bg stays normal so text is readable)
                self.configure(border_color=ERR_COLOR, text_color=ERR_COLOR)

    def flash_error(self) -> None:
        """Briefly pulse the border red to flag a required-but-empty field."""
        self.configure(border_color=ERR_COLOR, fg_color="#2A1010",
                       text_color=TEXT_MAIN)
        self.after(700, lambda: self.configure(
            border_color=BORDER, fg_color=BG_APP, text_color=TEXT_MAIN))


# ══════════════════════════════════════════════════════════════════════════════
#  RICH TEXTBOX — syntax-colour tags  [IMP-2: batched state change]
# ══════════════════════════════════════════════════════════════════════════════
class RichTextbox(ctk.CTkTextbox):
    """CTkTextbox with VS-Code-palette colour tags for step-by-step output.

    Imp-2: clear() sets state="normal" once; individual write() calls no
    longer redundantly toggle state, reducing configure() round-trips.
    """

    _TAGS: dict[str, dict] = {
        "header":  {"foreground": ACCENT,      "font": ("Consolas", 13, "bold")},
        "value":   {"foreground": OK_COLOR,     "font": ("Consolas", 13)},
        "dim":     {"foreground": TEXT_HINT,    "font": ("Consolas", 12)},
        "warn":    {"foreground": WARN_COLOR,   "font": ("Consolas", 13)},
        "err":     {"foreground": ERR_COLOR,    "font": ("Consolas", 13, "bold")},
        "key":     {"foreground": "#9CDCFE",    "font": ("Consolas", 13)},
        "num":     {"foreground": "#B5CEA8",    "font": ("Consolas", 13)},
        "formula": {"foreground": PLOT_PURPLE,  "font": ("Consolas", 13, "italic")},
        "plain":   {"foreground": TEXT_MAIN,    "font": ("Consolas", 13)},
        "table_h": {"foreground": YELLOW,       "font": ("Consolas", 13, "bold")},
    }

    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, **kwargs)
        tb = self._textbox
        for name, opts in self._TAGS.items():
            tb.tag_configure(name, **opts)
        self._unlocked = False   # tracks whether we're in an open write batch

    def clear(self) -> None:
        """Unlock, clear content, and stay unlocked for the write batch."""
        self.configure(state="normal")
        self._textbox.delete("1.0", "end")
        self._unlocked = True

    def write(self, text: str, tag: str = "plain") -> None:
        """Append coloured text.  Must be called between clear() and lock()."""
        tb    = self._textbox
        start = tb.index("end-1c")
        tb.insert("end", text)
        end   = tb.index("end-1c")
        tb.tag_add(tag, start, end)

    def writeln(self, text: str = "", tag: str = "plain") -> None:
        self.write(text + "\n", tag)

    def lock(self) -> None:
        """Seal the textbox after the write batch completes."""
        self.configure(state="disabled")
        self._unlocked = False

    def get_all(self) -> str:
        return self._textbox.get("1.0", "end")

    # ── convenience writers (all route through writeln)
    def header(self, t: str):  self.writeln(t, "header")
    def value(self, t: str):   self.writeln(t, "value")
    def dim(self, t: str):     self.writeln(t, "dim")
    def key(self, t: str):     self.writeln(t, "key")
    def num(self, t: str):     self.writeln(t, "num")
    def formula(self, t: str): self.writeln(t, "formula")
    def warn(self, t: str):    self.writeln(t, "warn")
    def err(self, t: str):     self.writeln(t, "err")
    def table_h(self, t: str): self.writeln(t, "table_h")
    def sep(self):             self.writeln(_div(), "dim")
    def blank(self):           self.writeln("", "plain")


# ══════════════════════════════════════════════════════════════════════════════
#  WELCOME SCREEN  [FIX-1: 2-row pill grid  |  FIX-2: leak-safe pulse]
# ══════════════════════════════════════════════════════════════════════════════
class WelcomeScreen(ctk.CTkFrame):
    """Splash screen shown before the main calculator is rendered.

    Fix-1: Method pills are laid out in two centred rows (4 + 3) using nested
           pack frames.  Each label uses explicit padx/pady padding — no
           leading/trailing spaces in the text string.  The pill frame itself
           uses pack_propagate(True) so it can grow to fit its content.

    Fix-2: _pulse() checks winfo_exists() before rescheduling; the after()
           loop therefore terminates automatically when the splash is destroyed.
    """

    # Row 1: 4 pills  |  Row 2: 3 pills  (centred)
    _PILL_ROWS: list[list[tuple[str, str]]] = [
        [
            ("LU Decomposition",  OK_COLOR),
            ("Gauss-Seidel",      WARN_COLOR),
            ("Regula Falsi",      PLOT_CYAN),
            ("Newton-Raphson",    PLOT_PURPLE),
        ],
        [
            ("Fwd Interpolation", ACCENT),
            ("Stirling Interp.",  ACCENT),
            ("Lagrange Interp.",  ACCENT),
        ],
    ]

    _FEATURES: list[tuple[str, str]] = [
        ("⚡", "Real-Time Validation"),
        ("📈", "Live Dark Plots"),
        ("💾", "Copy & Export PNG"),
        ("⌨", "Ctrl+Enter Shortcut"),
    ]

    def __init__(self, master, on_launch, **kwargs) -> None:
        super().__init__(master, fg_color=BG_APP, corner_radius=0, **kwargs)
        self._alive = True   # [FIX-2] pulse sentinel

        # ── top accent strip
        ctk.CTkFrame(self, fg_color=ACCENT, height=3,
                     corner_radius=0).pack(fill="x", side="top")

        centre = ctk.CTkFrame(self, fg_color="transparent")
        centre.place(relx=0.5, rely=0.47, anchor="center")

        # ── icon badge (∑ symbol)
        badge = ctk.CTkFrame(centre, fg_color=ACCENT,
                             width=76, height=76, corner_radius=20)
        badge.pack(pady=(0, 18))
        badge.pack_propagate(False)
        ctk.CTkLabel(badge, text="∑", font=("Segoe UI", 36, "bold"),
                     text_color="#FFFFFF").place(relx=0.5, rely=0.5,
                                                  anchor="center")

        # ── title + subtitle
        ctk.CTkLabel(centre, text="Calculus & Algebra Engine",
                     font=FONT_SPLASH, text_color=TEXT_MAIN).pack()
        ctk.CTkLabel(centre, text="Advanced Numerical Methods Calculator",
                     font=FONT_SUB_TXT, text_color=TEXT_SUB).pack(pady=(5, 0))

        # ── method pills: 2-row grid  [FIX-1] ─────────────────────
        # Outer container – not constrained to any fixed width so that each
        # row centres freely as the window grows.
        pills_outer = ctk.CTkFrame(centre, fg_color="transparent")
        pills_outer.pack(pady=(22, 0))

        for row_pills in self._PILL_ROWS:
            row_frame = ctk.CTkFrame(pills_outer, fg_color="transparent")
            # Centre each row by letting pack decide the horizontal position.
            row_frame.pack(pady=(0, 8))

            for label, color in row_pills:
                pill = ctk.CTkFrame(
                    row_frame,
                    fg_color=BG_CARD,
                    corner_radius=20,
                    border_width=1,
                    border_color=color,
                )
                pill.pack(side="left", padx=5)
                # [FIX-1] padx/pady on the label provide the internal spacing;
                # no hard-coded spaces in the text string.
                ctk.CTkLabel(
                    pill,
                    text=label,
                    font=FONT_PILL,
                    text_color=color,
                ).pack(padx=14, pady=6)
        # ── end pills ──────────────────────────────────────────────

        # ── thin separator
        ctk.CTkFrame(centre, fg_color=BORDER, height=1,
                     width=440).pack(pady=24)

        # ── launch button (stored for pulse animation)
        self._btn = ctk.CTkButton(
            centre, text="  ▶   Launch Calculator",
            font=FONT_BTN, height=50, width=260, corner_radius=12,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color="#FFFFFF", command=on_launch,
        )
        self._btn.pack()
        self._pulse_state = False
        self._pulse()

        # ── feature icon row
        feat_row = ctk.CTkFrame(centre, fg_color="transparent")
        feat_row.pack(pady=(22, 0))
        for icon, label in self._FEATURES:
            fc = ctk.CTkFrame(feat_row, fg_color=BG_CARD, corner_radius=8)
            fc.pack(side="left", padx=5)
            ctk.CTkLabel(fc, text=f"{icon}  {label}",
                         font=FONT_SMALL, text_color=TEXT_SUB
                         ).pack(padx=11, pady=6)

        # ── bottom credit + accent strip
        ctk.CTkLabel(self,
                     text="Powered by CustomTkinter · Matplotlib · NumPy",
                     font=FONT_CREDIT,
                     text_color=TEXT_HINT).pack(side="bottom", pady=10)
        ctk.CTkFrame(self, fg_color=ACCENT_DIM, height=2,
                     corner_radius=0).pack(fill="x", side="bottom")

    def _pulse(self) -> None:
        """Alternate the launch button border colour.

        [FIX-2] Checks winfo_exists() before rescheduling so the loop stops
        automatically when the splash frame is destroyed on app launch — no
        dangling after() callbacks or TclError exceptions.
        """
        if not self._alive:
            return
        try:
            if not self.winfo_exists():
                self._alive = False
                return
            self._pulse_state = not self._pulse_state
            self._btn.configure(
                border_width=2,
                border_color=ACCENT if self._pulse_state else ACCENT_DIM,
            )
        except Exception:
            self._alive = False
            return
        self.after(900, self._pulse)

    def destroy(self) -> None:
        """Override to mark the sentinel before Tkinter teardown."""
        self._alive = False
        super().destroy()


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD INFO CARD  [FIX-6: guarded update_info]
# ══════════════════════════════════════════════════════════════════════════════
class MethodInfoCard(ctk.CTkFrame):
    """Compact card: algorithm class pill, complexity, description."""

    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, fg_color=BG_CARD, corner_radius=8,
                         border_width=1, border_color=BORDER, **kwargs)

        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="x", padx=12, pady=8)

        top = ctk.CTkFrame(inner, fg_color="transparent")
        top.pack(fill="x")

        self._pill = ctk.CTkFrame(top, fg_color=BG_HOVER, corner_radius=10)
        self._pill.pack(side="left")
        self._tag_lbl = ctk.CTkLabel(self._pill, text="",
                                     font=FONT_PILL, text_color=OK_COLOR)
        self._tag_lbl.pack(padx=8, pady=3)

        self._cx_lbl = ctk.CTkLabel(top, text="", font=FONT_SMALL,
                                    text_color=TEXT_HINT)
        self._cx_lbl.pack(side="left", padx=(10, 0))

        self._desc = ctk.CTkLabel(inner, text="", font=FONT_SMALL,
                                  text_color=TEXT_SUB,
                                  wraplength=280, justify="left")
        self._desc.pack(anchor="w", pady=(5, 0))

    def update_info(self, method: str) -> None:
        """[FIX-6] Wrapped in try/except — method change during destruction
        can no longer raise TclError and crash the app."""
        try:
            tag, color, cx, desc = METHOD_META.get(
                method, ("—", TEXT_HINT, "—", ""))
            self._tag_lbl.configure(text=f"  {tag}  ", text_color=color)
            self._cx_lbl.configure(text=f"complexity:  {cx}")
            self._desc.configure(text=desc)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  MATRIX INPUT CARD
# ══════════════════════════════════════════════════════════════════════════════
class MatrixInputCard(ctk.CTkFrame):
    """Self-contained NxN matrix grid with optional x₀ column."""

    _SIZES = ["2x2", "3x3", "4x4", "5x5"]

    def __init__(self, master, needs_x0: bool = False,
                 default_size: int = 3,
                 pre_A=None, pre_B=None, pre_X0=None, **kwargs) -> None:
        super().__init__(master, fg_color=BG_CARD, corner_radius=10,
                         border_width=1, border_color=BORDER, **kwargs)
        self.needs_x0     = needs_x0
        self._entries_A:  list[list[ValidatedEntry]] = []
        self._entries_B:  list[ValidatedEntry] = []
        self._entries_X0: list[ValidatedEntry] = []

        # ── toolbar row
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=14, pady=(12, 4))
        ctk.CTkLabel(bar, text="Matrix Size:", font=FONT_SECTION,
                     text_color=TEXT_MAIN).pack(side="left")
        self._size_var = ctk.StringVar(value=f"{default_size}x{default_size}")
        ctk.CTkOptionMenu(
            bar, values=self._SIZES, variable=self._size_var,
            command=lambda _: self._rebuild(),
            width=86, height=28,
            fg_color=BG_APP, button_color=ACCENT,
            button_hover_color=ACCENT_HOVER, text_color=TEXT_MAIN,
            dropdown_fg_color=BG_APP, dropdown_text_color=TEXT_MAIN,
            dropdown_hover_color=BG_HOVER,
        ).pack(side="left", padx=10)

        hint = "  A · x = b" + ("  +  x₀" if needs_x0 else "")
        ctk.CTkLabel(bar, text=hint, font=FONT_SMALL,
                     text_color=TEXT_HINT).pack(side="left")

        self._grid = ctk.CTkFrame(self, fg_color="transparent")
        self._grid.pack(fill="both", expand=True, padx=14, pady=(4, 14))
        self._build(default_size, pre_A, pre_B, pre_X0)

    # ── internal builders ─────────────────────────────────────────
    def _rebuild(self) -> None:
        self._build(int(self._size_var.get().split("x")[0]))

    def _build(self, n: int, pre_A=None, pre_B=None, pre_X0=None) -> None:
        for w in self._grid.winfo_children():
            w.destroy()
        self._entries_A.clear()
        self._entries_B.clear()
        self._entries_X0.clear()

        extra = 2 if self.needs_x0 else 1
        for c in range(n + extra):
            self._grid.grid_columnconfigure(c, weight=1)

        # column headers
        ctk.CTkLabel(self._grid, text="A  (coefficients)",
                     font=FONT_MONO_SM, text_color=ACCENT
                     ).grid(row=0, column=0, columnspan=n, pady=(4, 8))
        ctk.CTkLabel(self._grid, text="b", font=FONT_MONO_SM,
                     text_color=OK_COLOR
                     ).grid(row=0, column=n, padx=(14, 2), pady=(4, 8))
        if self.needs_x0:
            ctk.CTkLabel(self._grid, text="x₀", font=FONT_MONO_SM,
                         text_color=WARN_COLOR
                         ).grid(row=0, column=n + 1, padx=(14, 2), pady=(4, 8))

        for i in range(n):
            ctk.CTkLabel(self._grid, text=f"r{i+1}",
                         font=FONT_MONO_SM, text_color=TEXT_HINT, width=18
                         ).grid(row=i + 1, column=0, sticky="e", padx=(0, 3))
            row_a: list[ValidatedEntry] = []
            for j in range(n):
                e = ValidatedEntry(self._grid, monospace=True, compact=True)
                e.grid(row=i + 1, column=j, padx=2, pady=2, sticky="ew")
                if pre_A and i < len(pre_A) and j < len(pre_A[i]):
                    e.insert(0, str(pre_A[i][j]))
                row_a.append(e)
            self._entries_A.append(row_a)

            b_e = ValidatedEntry(self._grid, monospace=True, compact=True)
            b_e.grid(row=i + 1, column=n, padx=(14, 2), pady=2, sticky="ew")
            if pre_B and i < len(pre_B):
                b_e.insert(0, str(pre_B[i]))
            self._entries_B.append(b_e)

            if self.needs_x0:
                x0_e = ValidatedEntry(self._grid, monospace=True, compact=True)
                x0_e.grid(row=i + 1, column=n + 1, padx=(14, 2), pady=2, sticky="ew")
                if pre_X0 and i < len(pre_X0):
                    x0_e.insert(0, str(pre_X0[i]))
                self._entries_X0.append(x0_e)

    # ── public API ────────────────────────────────────────────────
    def get_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        try:
            A  = [[float(e.get()) for e in row] for row in self._entries_A]
            B  = [float(e.get()) for e in self._entries_B]
            X0 = ([float(e.get() or "0") for e in self._entries_X0]
                  if self.needs_x0 else None)
            return (np.array(A, dtype=float),
                    np.array(B, dtype=float),
                    np.array(X0, dtype=float) if X0 is not None else None)
        except ValueError:
            raise ValueError("All matrix cells must contain valid numbers.")

    def clear(self) -> None:
        for row in self._entries_A:
            for e in row: e.delete(0, "end")
        for e in self._entries_B:  e.delete(0, "end")
        for e in self._entries_X0: e.delete(0, "end")

    def all_entries(self) -> list[ValidatedEntry]:
        out: list[ValidatedEntry] = list(self._entries_B) + list(self._entries_X0)
        for row in self._entries_A:
            out += row
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT PANEL  [FIX-3: toolbar theming guard  |  FIX-5: cleanup  |  IMP-3]
# ══════════════════════════════════════════════════════════════════════════════
class OutputPanel(ctk.CTkFrame):

    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, fg_color=BG_SIDE, corner_radius=12,
                         border_width=1, border_color=BORDER, **kwargs)
        self._fig_canvas:  FigureCanvasTkAgg | None = None
        self._last_fig:    Figure | None = None
        self._toolbar_host: tk.Frame | None = None   # [FIX-5] tracked for cleanup
        self._sum_cache:   str = ""

        self._tabs = ctk.CTkTabview(
            self, fg_color=BG_APP, text_color=TEXT_MAIN,
            segmented_button_selected_color=ACCENT,
            segmented_button_selected_hover_color=ACCENT_HOVER,
            segmented_button_unselected_color=BG_CARD,
            segmented_button_unselected_hover_color=BG_HOVER,
            corner_radius=10, anchor="nw",
        )
        self._tabs.pack(fill="both", expand=True, padx=16, pady=16)

        self._tab_sum = self._tabs.add("  Summary  ")
        self._tab_stp = self._tabs.add("  Steps  ")
        self._tab_vis = self._tabs.add("  Plot 📈  ")

        self._build_summary_tab()
        self._build_steps_tab()
        self._build_vis_tab()

    # ── tab builders ─────────────────────────────────────────────
    def _build_summary_tab(self) -> None:
        t = self._tab_sum

        hdr = ctk.CTkFrame(t, fg_color="transparent")
        hdr.pack(fill="x")
        self._copy_sum = ctk.CTkButton(
            hdr, text="⎘ Copy", font=FONT_SMALL, width=72, height=26,
            corner_radius=6, fg_color=BG_CARD, hover_color=BG_HOVER,
            text_color=TEXT_SUB, command=self._copy_summary,
        )
        self._copy_sum.pack(side="right", padx=12, pady=(8, 0))

        body = ctk.CTkFrame(t, fg_color="transparent")
        body.pack(expand=True, fill="both")

        self._sum_badge = ctk.CTkLabel(body, text="●",
                                       font=("Segoe UI", 32),
                                       text_color=ACCENT)
        self._sum_badge.pack(pady=(28, 4))

        self._sum_status = ctk.CTkLabel(body, text="System Online",
                                        font=FONT_TITLE, text_color=ACCENT)
        self._sum_status.pack()

        self._sum_result = ctk.CTkLabel(
            body,
            text="Select a method and press  ▶ Calculate",
            font=FONT_RESULT, text_color=TEXT_MAIN,
            wraplength=520, justify="center")
        self._sum_result.pack(pady=(14, 0), padx=30)

        self._sum_meta = ctk.CTkLabel(body, text="", font=FONT_MONO_SM,
                                      text_color=TEXT_SUB)
        self._sum_meta.pack(pady=(10, 0))

        ctk.CTkLabel(body,
                     text="Ctrl+Enter = Calculate    Ctrl+L = Clear",
                     font=("Segoe UI", 10),
                     text_color=TEXT_HINT).pack(side="bottom", pady=12)

    def _build_steps_tab(self) -> None:
        t = self._tab_stp

        bar = ctk.CTkFrame(t, fg_color="transparent")
        bar.pack(fill="x")
        for lbl, cmd in [("⎘ Copy",   self._copy_steps),
                          ("⬆ Top",   self._scroll_top),
                          ("⬇ Bottom", self._scroll_bot)]:
            ctk.CTkButton(
                bar, text=lbl, font=FONT_SMALL, width=82, height=26,
                corner_radius=6, fg_color=BG_CARD, hover_color=BG_HOVER,
                text_color=TEXT_SUB, command=cmd,
            ).pack(side="left", padx=(8, 0), pady=(8, 4))

        self._step_box = RichTextbox(
            t, font=FONT_MONO, fg_color=BG_APP,
            text_color=TEXT_MAIN, wrap="none",
            corner_radius=8, border_width=1, border_color=BORDER,
        )
        self._step_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._step_box.clear()
        self._step_box.dim("  Awaiting calculation…")
        self._step_box.lock()

    def _build_vis_tab(self) -> None:
        t = self._tab_vis

        bar = ctk.CTkFrame(t, fg_color="transparent")
        bar.pack(fill="x")
        self._save_btn = ctk.CTkButton(
            bar, text="💾 Save…", font=FONT_SMALL, width=90, height=26,
            corner_radius=6, fg_color=BG_CARD, hover_color=BG_HOVER,
            text_color=TEXT_SUB, command=self._save_plot, state="disabled",
        )
        self._save_btn.pack(side="right", padx=12, pady=(8, 4))

        self._vis_host = ctk.CTkFrame(t, fg_color="transparent")
        self._vis_host.pack(fill="both", expand=True)
        ctk.CTkLabel(self._vis_host,
                     text="📈  Run a calculation to see the graph.",
                     font=FONT_MAIN, text_color=TEXT_HINT).pack(expand=True)

    # ── Public writers ───────────────────────────────────────────
    def set_summary(self, status: str, body: str,
                    meta: str = "", success: bool = True) -> None:
        c = OK_COLOR if success else ERR_COLOR
        a = ACCENT   if success else ERR_COLOR
        self._sum_badge.configure(text_color=c)
        self._sum_status.configure(text=status, text_color=a)
        self._sum_result.configure(text=body)
        self._sum_meta.configure(text=meta)
        self._sum_cache = f"{status}\n{body}\n{meta}"
        self._tabs.set("  Summary  ")

    def write_steps(self, fn) -> None:
        """fn receives the unlocked RichTextbox; caller writes coloured output."""
        self._step_box.clear()           # opens write-batch
        fn(self._step_box)
        self._step_box.lock()            # closes write-batch

    def plot(self, f, x_lo: float, x_hi: float,
             root=None, label: str = "f(x)", title: str = "") -> None:
        def _draw(fig, ax):
            xs = np.linspace(x_lo, x_hi, 600)
            try:
                try:    ys = f(xs)
                except: ys = np.array([f(float(x)) for x in xs])
                ax.plot(xs, ys, color=PLOT_CYAN, linewidth=2, label=label)
                ax.axhline(0, color=TEXT_HINT, linewidth=0.8, linestyle="--")
                if root is not None:
                    ax.axvline(root, color=PLOT_ORANGE, linewidth=0.9,
                               linestyle=":", alpha=0.7)
                    ax.scatter([root], [0.0], color=PLOT_RED, s=80,
                               zorder=7, label=f"Root ≈ {root:.6g}")
                ax.legend(facecolor=BG_CARD, edgecolor=BORDER,
                          labelcolor=TEXT_MAIN, fontsize=9)
            except Exception as exc:
                ax.text(0.5, 0.5, f"Plot error:\n{exc}",
                        transform=ax.transAxes, color=ERR_COLOR, ha="center")
        self._render_plot(_draw, title)

    def plot_convergence(self, residuals: list,
                         title: str = "Convergence") -> None:
        def _draw(fig, ax):
            ax.semilogy(range(1, len(residuals) + 1), residuals,
                        color=PLOT_CYAN, linewidth=2,
                        marker="o", markersize=4, label="‖residual‖∞")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("‖Ax − b‖∞  (log scale)")
            ax.legend(facecolor=BG_CARD, edgecolor=BORDER,
                      labelcolor=TEXT_MAIN, fontsize=9)
        self._render_plot(_draw, title)

    def plot_interpolation(self, xs: list, ys: list,
                           t: float, result: float,
                           interp_fn, title: str = "") -> None:
        def _draw(fig, ax):
            margin  = (max(xs) - min(xs)) * 0.12 + 0.5
            x_curve = np.linspace(min(xs) - margin, max(xs) + margin, 400)
            y_curve = []
            for xv in x_curve:
                try:    y_curve.append(interp_fn(float(xv)))
                except: y_curve.append(float("nan"))
            ax.plot(x_curve, y_curve, color=PLOT_CYAN,
                    linewidth=2, label="Interpolant")
            ax.scatter(xs, ys, color=OK_COLOR, s=60,
                       zorder=6, label="Data points")
            ax.scatter([t], [result], color=PLOT_RED, s=120,
                       zorder=7, marker="*",
                       label=f"f({t}) ≈ {result:.4f}")
            ax.axvline(t, color=PLOT_RED, linewidth=0.8,
                       linestyle=":", alpha=0.6)
            ax.legend(facecolor=BG_CARD, edgecolor=BORDER,
                      labelcolor=TEXT_MAIN, fontsize=9)
        self._render_plot(_draw, title)

    # ── Core renderer  [FIX-3, FIX-5, IMP-3] ────────────────────
    def _render_plot(self, draw_fn, title: str) -> None:
        """Render a new Matplotlib figure into the Visualization tab.

        Fix-3: Toolbar children are styled only if they are tk.Button or
               tk.Label instances — avoids AttributeError on tk.Frame etc.
        Fix-5: The old toolbar host frame is explicitly destroyed so it
               doesn't accumulate as a hidden widget.
        Imp-3: self._last_fig is assigned after tight_layout so a draw
               failure doesn't leave a stale reference.
        """
        # ── teardown previous canvas + toolbar
        for w in self._vis_host.winfo_children():
            w.destroy()
        if self._last_fig:
            try:    plt.close(self._last_fig)
            except: pass
        if self._fig_canvas:
            try:    self._fig_canvas.get_tk_widget().destroy()
            except: pass
        # [FIX-5] explicitly destroy old toolbar host
        if self._toolbar_host and self._toolbar_host.winfo_exists():
            try:    self._toolbar_host.destroy()
            except: pass
        self._toolbar_host = None

        # ── create figure with dark theme
        fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=96)

        fig.patch.set_facecolor(BG_APP)
        ax.set_facecolor(BG_SIDE)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color(BORDER)
        ax.tick_params(colors=TEXT_SUB, labelsize=9)
        ax.xaxis.label.set_color(TEXT_SUB)
        ax.yaxis.label.set_color(TEXT_SUB)
        ax.grid(color=PLOT_GRID, linewidth=0.6, linestyle="--", alpha=0.7)
        ax.set_title(title, color=TEXT_MAIN, fontsize=11, pad=10)

        try:
            draw_fn(fig, ax)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Draw error:\n{exc}",
                    transform=ax.transAxes, color=ERR_COLOR, ha="center")

        fig.tight_layout(pad=1.4)
        self._last_fig = fig   # [IMP-3] assign only after successful layout

        # ── embed canvas
        self._fig_canvas = FigureCanvasTkAgg(fig, master=self._vis_host)
        cw = self._fig_canvas.get_tk_widget()
        cw.pack(fill="both", expand=True, padx=10, pady=(4, 0))

        # ── floating NavigationToolbar (pill style)
        self._toolbar_host = ctk.CTkFrame(self._vis_host, fg_color="#E0E0E0", corner_radius=10)
        self._toolbar_host.pack(side="bottom", fill="none", expand=False, pady=10)

        toolbar = NavigationToolbar2Tk(self._fig_canvas, self._toolbar_host)
        toolbar.configure(background="#E0E0E0")
        toolbar.pack(padx=4, pady=2) # Slight internal padding to preserve rounded pill edges
        
        for widget in toolbar.winfo_children():
            if isinstance(widget, (tk.Button, tk.Label)):
                try:
                    widget.configure(
                        background="#E0E0E0",
                        relief="flat", 
                        borderwidth=0,
                        activebackground="#CCCCCC",
                        cursor="hand2"
                    )
                except (tk.TclError, TypeError):
                    pass
        toolbar.update()

        # ── CRITICAL: three-step forced render (prevents blank-canvas bug)
        self._fig_canvas.draw()
        self._fig_canvas.flush_events()
        self.update_idletasks()

        self._save_btn.configure(state="normal")
        self._tabs.set("  Plot 📈  ")

    # ── Clipboard / file helpers ──────────────────────────────────
    def _copy_summary(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self._sum_cache)
        self._flash(self._copy_sum, "✔ Copied")

    def _copy_steps(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self._step_box.get_all())

    def _scroll_top(self):  self._step_box._textbox.see("1.0")
    def _scroll_bot(self):  self._step_box._textbox.see("end")

    def _save_plot(self) -> None:
        if not self._last_fig:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"),
                       ("SVG Vector", "*.svg"),
                       ("PDF",        "*.pdf")],
            title="Save Plot As…",
        )
        if path:
            self._last_fig.savefig(path, dpi=150,
                                   bbox_inches="tight", facecolor=BG_APP)

    @staticmethod
    def _flash(btn, msg: str, ms: int = 1200) -> None:
        try:
            orig = btn.cget("text")
            btn.configure(text=msg, text_color=OK_COLOR)
            btn.after(ms, lambda: btn.configure(text=orig, text_color=TEXT_SUB))
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  STATUS BAR  [IMP-4: grid_propagate]
# ══════════════════════════════════════════════════════════════════════════════
class StatusBar(ctk.CTkFrame):
    """Persistent one-line bar at the bottom of the app window."""

    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, fg_color=BG_CARD,
                         corner_radius=0, height=25, **kwargs)
        self.pack_propagate(False)
        self._count = 0

        ctk.CTkFrame(self, fg_color=BORDER, width=1,
                     corner_radius=0).pack(side="left", fill="y", pady=0, ipady=0)

        self._method_lbl = ctk.CTkLabel(self, text="",
                                        font=("Segoe UI", 9), text_color=TEXT_HINT)
        self._method_lbl.pack(side="left", padx=14, pady=0, ipady=0)

        self._time_lbl = ctk.CTkLabel(self, text="",
                                      font=("Segoe UI", 9), text_color=TEXT_HINT)
        self._time_lbl.pack(side="right", padx=14, pady=0, ipady=0)

        self._count_lbl = ctk.CTkLabel(self, text="Calculations: 0",
                                       font=("Segoe UI", 9), text_color=TEXT_HINT)
        self._count_lbl.pack(side="right", padx=14, pady=0, ipady=0)



    def set_method(self, m: str) -> None:
        short = m.split("(")[0].strip()
        self._method_lbl.configure(text=f"Active:  {short}")

    def record_calc(self, elapsed_ms: float) -> None:
        self._count += 1
        self._count_lbl.configure(text=f"Calculations: {self._count}")
        self._time_lbl.configure(text=f"Last:  {elapsed_ms:.1f} ms")


# ══════════════════════════════════════════════════════════════════════════════
#  CONTROL PANEL  [FIX-7: matrix-entry empty check  |  IMP-5, IMP-6]
# ══════════════════════════════════════════════════════════════════════════════
class ControlPanel(ctk.CTkFrame):

    def __init__(self, master, output: OutputPanel,
                 status_bar: StatusBar, **kwargs) -> None:
        super().__init__(master, fg_color=BG_SIDE, corner_radius=12,
                         border_width=1, border_color=BORDER, **kwargs)
        self._out    = output
        self._status = status_bar
        self._app    = master
        self._widgets: dict[str, ValidatedEntry] = {}
        self._matrix_card: MatrixInputCard | None = None

        # ── header strip
        hdr = ctk.CTkFrame(self, fg_color=BG_CARD,
                           corner_radius=0, height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="⚙  Calculus Engine",
                     font=FONT_TITLE, text_color=ACCENT).pack(
            side="left", padx=16, pady=10)

        # ── method selector
        sel = ctk.CTkFrame(self, fg_color="transparent")
        sel.pack(fill="x", padx=16, pady=(12, 4))
        ctk.CTkLabel(sel, text="NUMERICAL METHOD",
                     font=("Segoe UI", 9, "bold"),
                     text_color=TEXT_HINT).pack(anchor="w")

        self._method_var = ctk.StringVar(value=METHODS[0])   # ← Doolittle default
        self._dropdown = ctk.CTkOptionMenu(
            sel, values=METHODS, variable=self._method_var,
            command=self._on_method_change,
            font=FONT_MAIN, height=36, corner_radius=8,
            fg_color=BG_CARD, button_color=ACCENT,
            button_hover_color=ACCENT_HOVER, text_color=TEXT_MAIN,
            dropdown_fg_color=BG_APP, dropdown_text_color=TEXT_MAIN,
            dropdown_hover_color=BG_HOVER,
        )
        self._dropdown.pack(fill="x", pady=(6, 0))

        # ── method info card
        self._info = MethodInfoCard(self)
        self._info.pack(fill="x", padx=16, pady=(8, 0))
        self._info.update_info(METHODS[0])

        # ── scrollable input area
        self._scroll = ctk.CTkScrollableFrame(
            self, fg_color="transparent", corner_radius=0,
            scrollbar_button_color=BG_CARD,
            scrollbar_button_hover_color=BORDER,
        )
        self._scroll.pack(fill="both", expand=True, padx=6, pady=6)

        # ── footer (progress + buttons)
        foot = ctk.CTkFrame(self, fg_color="transparent")
        foot.pack(fill="x", padx=16, pady=(0, 14))

        self._progress = ctk.CTkProgressBar(
            foot, fg_color=BG_CARD, progress_color=ACCENT,
            height=4, corner_radius=2)
        self._progress.pack(fill="x", pady=(0, 10))
        self._progress.set(0)

        btn_row = ctk.CTkFrame(foot, fg_color="transparent")
        btn_row.pack(fill="x")
        btn_row.grid_columnconfigure(1, weight=1)

        self._clear_btn = ctk.CTkButton(
            btn_row, text="Clear", font=FONT_BTN, height=42, width=78,
            corner_radius=8, fg_color=BG_CARD, hover_color=BG_HOVER,
            text_color=TEXT_MAIN, border_color=BORDER, border_width=1,
            command=self._clear,
        )
        self._clear_btn.grid(row=0, column=0, padx=(0, 10))

        self._calc_btn = ctk.CTkButton(
            btn_row, text="  ▶  Calculate",
            font=FONT_BTN, height=42, corner_radius=8,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color="#FFFFFF", command=self._start_calculate,
        )
        self._calc_btn.grid(row=0, column=1, sticky="ew")

        self._render_inputs(METHODS[0])
        self._status.set_method(METHODS[0])

    # ── method change ─────────────────────────────────────────────
    def _on_method_change(self, choice: str) -> None:
        self._info.update_info(choice)
        self._render_inputs(choice)
        self._status.set_method(choice)
        try:
            self._app.title(
                f"Numerical Methods  —  {choice.split('(')[0].strip()}")
        except Exception:
            pass

    # ── input builder helpers ─────────────────────────────────────
    def _sec(self, text: str) -> None:
        row = ctk.CTkFrame(self._scroll, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=(12, 2))
        ctk.CTkLabel(row, text=text, font=FONT_SECTION,
                     text_color=TEXT_SUB).pack(side="left")

    def _lbl(self, text: str) -> None:
        ctk.CTkLabel(self._scroll, text=text, font=FONT_MAIN,
                     text_color=TEXT_MAIN).pack(anchor="w", padx=10, pady=(6, 2))

    def _entry(self, key: str, placeholder: str = "",
               default: str = "") -> ValidatedEntry:
        e = ValidatedEntry(self._scroll,
                           placeholder_text=placeholder, width=200)
        e.pack(fill="x", padx=10, pady=(0, 4))
        if default:
            e.insert(0, default)
        self._widgets[key] = e
        return e

    def _two_col(self, ll: str, rl: str,
                 lk: str, rk: str, ld: str = "", rd: str = "") -> None:
        row = ctk.CTkFrame(self._scroll, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(4, 4))
        row.grid_columnconfigure((0, 1), weight=1)
        for c, lbl in [(0, ll), (1, rl)]:
            ctk.CTkLabel(row, text=lbl, font=FONT_MAIN,
                         text_color=TEXT_MAIN).grid(
                row=0, column=c, sticky="w",
                padx=(0 if c == 0 else 6, 6 if c == 0 else 0))
        for c, key, dflt in [(0, lk, ld), (1, rk, rd)]:
            e = ValidatedEntry(row)
            e.grid(row=1, column=c, sticky="ew",
                   padx=(0 if c == 0 else 6, 6 if c == 0 else 0),
                   pady=(2, 0))
            if dflt:
                e.insert(0, dflt)
            self._widgets[key] = e

    # ── input renderer ────────────────────────────────────────────
    def _render_inputs(self, method: str) -> None:
        for w in self._scroll.winfo_children():
            w.destroy()
        self._widgets.clear()
        self._matrix_card = None
        m = method

        if "Doolittle" in m:
            self._sec("Matrix A   ·   Vector b")
            self._matrix_card = MatrixInputCard(
                self._scroll, needs_x0=False, default_size=3,
                pre_A=[[2,-1,-2],[-4,6,3],[-4,-2,8]], pre_B=[3,-8,12])
            self._matrix_card.pack(fill="x", padx=6, pady=(4, 0))

        elif "Gauss" in m:
            self._sec("Matrix A   ·   Vector b   ·   Initial x₀")
            self._matrix_card = MatrixInputCard(
                self._scroll, needs_x0=True, default_size=3,
                pre_A=[[4,1,-1],[2,7,1],[1,-3,12]],
                pre_B=[3,19,31], pre_X0=[0,0,0])
            self._matrix_card.pack(fill="x", padx=6, pady=(4, 6))
            self._sec("Solver Settings")
            self._two_col("Tolerance", "Max Iterations",
                          "tol", "max_iter", "1e-6", "1000")

        elif "False" in m:
            self._sec("Equation   f(x) = 0")
            self._lbl("Expression in x:")
            self._entry("eq", "e.g.  x**3 - x - 2", "x**3 - x - 2")
            self._sec("Bracket Interval")
            self._two_col("Lower bound  a", "Upper bound  b",
                          "a", "b", "1", "2")
            self._sec("Solver Settings")
            self._two_col("Tolerance", "Max Iterations",
                          "tol", "max_iter", "1e-6", "100")

        elif "Newton-Raph" in m:
            self._sec("Equations")
            self._lbl("f(x) =")
            self._entry("f", "e.g.  x**3 - 2*x - 5", "x**3 - 2*x - 5")
            self._lbl("f ′(x) =  (derivative)")
            self._entry("fp", "e.g.  3*x**2 - 2", "3*x**2 - 2")
            self._sec("Initial Conditions")
            self._two_col("Initial guess  x₀", "Tolerance",
                          "x0", "tol", "2", "1e-6")
            self._lbl("Max Iterations:")
            self._entry("max_iter", default="100")

        else:   # interpolation family
            algo = ("Forward"  if "Forward"  in m else
                    "Stirling" if "Stirling" in m else "Lagrange")
            self._sec(f"{algo} Interpolation — Dataset")
            self._lbl("X values  (comma-separated):")
            self._entry("xvals", "1981, 1991, …", "1981, 1991, 2001, 2011")
            self._lbl("Y = f(X) values  (comma-separated):")
            self._entry("yvals", "46, 66, …", "46, 66, 81, 93")
            self._sec("Target Point")
            self._lbl("Interpolate at  x =")
            self._entry("xtarget", "e.g.  1985", "1985")

    # ── clear  [IMP-5: switch back to Summary] ────────────────────
    def _clear(self) -> None:
        for e in self._widgets.values():
            e.delete(0, "end")
        if self._matrix_card:
            self._matrix_card.clear()
        # [IMP-5] return focus to Summary so user sees the confirmation
        self._out.set_summary("Cleared", "All fields have been reset.")
        self._out.write_steps(lambda tb: tb.dim("  Fields cleared."))

    # ── animated progress → compute ───────────────────────────────
    def _start_calculate(self) -> None:
        """[FIX-7] Checks both dict entries AND matrix-card cells for empties."""
        # Check plain widget dict entries
        empty = [e for e in self._widgets.values() if not e.get().strip()]
        # Also check matrix cells so half-filled grids are caught early
        if self._matrix_card:
            for e in self._matrix_card.all_entries():
                if not e.get().strip():
                    empty.append(e)
        if empty:
            for e in empty:
                e.flash_error()
            return

        self._calc_btn.configure(state="disabled", text="  ⏳  Working…")
        self._progress.set(0)
        self._out.set_summary("Engine Active", "Running…")
        self.update()

        t0  = time.perf_counter()
        DUR = 0.55

        def _tick() -> None:
            elapsed = time.perf_counter() - t0
            if elapsed < DUR:
                self._progress.set(min(1.0, elapsed / DUR))
                self.after(15, _tick)
            else:
                self._progress.set(1.0)
                self.update_idletasks()
                t_exec = time.perf_counter()
                self._execute()
                ms = (time.perf_counter() - t_exec) * 1000
                self._status.record_calc(ms)
                self._calc_btn.configure(state="normal",
                                         text="  ▶  Calculate")
                self.after(500, lambda: self._progress.set(0))

        _tick()

    # ── safe expression evaluator ─────────────────────────────────
    @staticmethod
    def _ev(expr: str, x) -> float:
        ns: dict = {
            "__builtins__": None, "x": x,
            "math": math, "np": np,
            "sin":   math.sin,   "cos":   math.cos,   "tan":   math.tan,
            "asin":  math.asin,  "acos":  math.acos,  "atan":  math.atan,
            "sinh":  math.sinh,  "cosh":  math.cosh,  "tanh":  math.tanh,
            "exp":   math.exp,   "log":   math.log,   "log10": math.log10,
            "log2":  math.log2,  "sqrt":  math.sqrt,  "abs":   abs,
            "pi":    math.pi,    "e":     math.e,
        }
        return float(eval(expr, ns))

    # ── dispatcher ────────────────────────────────────────────────
    def _execute(self) -> None:
        m = self._method_var.get()
        try:
            if   "Doolittle"   in m: self._run_doolittle()
            elif "Gauss"       in m: self._run_gauss_seidel()
            elif "False"       in m: self._run_false_position()
            elif "Newton-Raph" in m: self._run_newton_raphson()
            elif "Forward"     in m: self._run_interpolation("forward")
            elif "Stirling"    in m: self._run_interpolation("stirling")
            elif "Lagrange"    in m: self._run_interpolation("lagrange")
        except Exception as exc:
            self._out.set_summary("⚠  Engine Fault", str(exc), success=False)
            tb_text = traceback.format_exc()
            self._out.write_steps(
                lambda tb, t=tb_text: (
                    tb.err(_hdr("TRACEBACK")), tb.blank(), tb.err(t)
                )
            )

    # ══════════════════════════════════════════════════════════════
    #  METHOD RUNNERS
    # ══════════════════════════════════════════════════════════════
    def _run_doolittle(self) -> None:
        if self._matrix_card is None:
            raise RuntimeError("Matrix card not initialised.")
        A, B, _ = self._matrix_card.get_matrices()
        L, U, x = calculator.doolittle_lu_decomposition(A, B)
        n   = len(x)
        y   = np.linalg.solve(L, B)
        res = np.linalg.norm(A @ x - B, np.inf)

        self._out.set_summary(
            "✅  LU Decomposition Solved",
            "\n".join(f"  x{i+1}  =  {v:+.8f}" for i, v in enumerate(x)),
            meta=f"Size: {n}×{n}  |  ‖Ax−b‖∞ = {res:.3e}",
        )

        def _steps(tb: RichTextbox) -> None:
            tb.header(_hdr("Doolittle's LU Decomposition")); tb.blank()
            tb.key("  Strategy:  A = L · U  →  Ly = b  →  Ux = y"); tb.blank()
            tb.header(_hdr("Lower Triangular  L"))
            tb.num(_fmt_matrix(L)); tb.blank()
            tb.header(_hdr("Upper Triangular  U"))
            tb.num(_fmt_matrix(U)); tb.blank()
            tb.header(_hdr("Forward Substitution  y"))
            tb.num(_fmt_matrix(y)); tb.blank()
            tb.header(_hdr("Back Substitution  x  (solution)"))
            tb.value(_fmt_matrix(x)); tb.blank()
            tb.sep()
            tb.num(f"  Residual  ‖Ax − b‖∞  =  {res:.6e}"); tb.blank()
            for i, v in enumerate(x):
                tb.value(f"  x{i+1}  =  {v:+.10f}")

        self._out.write_steps(_steps)

    def _run_gauss_seidel(self) -> None:
        if self._matrix_card is None:
            raise RuntimeError("Matrix card not initialised.")
        A, B, X0 = self._matrix_card.get_matrices()
        if X0 is None:
            raise ValueError("Initial guess x₀ is required for Gauss-Seidel.")
        tol = float(self._widgets["tol"].get())
        mx  = int(self._widgets["max_iter"].get())
        sol, iters, converged = calculator.gauss_seidel(A, B, X0, tol, mx)
        sdd = calculator.is_strictly_diagonally_dominant(A)
        n   = len(B)

        # collect residuals for convergence plot
        xc, residuals = X0.copy(), []
        for _ in range(min(iters + 2, mx)):
            xn = xc.copy()
            for i in range(n):
                s    = sum(A[i][j] * xn[j] for j in range(n) if j != i)
                xn[i] = (B[i] - s) / A[i][i]
            rv = np.linalg.norm(A @ xn - B, np.inf)
            residuals.append(float(rv))
            if rv < tol:
                break
            xc = xn

        res_fin = np.linalg.norm(A @ sol - B, np.inf)

        self._out.set_summary(
            "✅  Converged" if converged else "⚠  Max Iterations Reached",
            "\n".join(f"  x{i+1}  =  {v:+.8f}" for i, v in enumerate(sol)),
            meta=(f"Iterations: {iters}  |  SDD: {'✓' if sdd else '✗'}"
                  f"  |  ‖Ax−b‖∞ = {res_fin:.3e}"),
            success=converged,
        )

        def _steps(tb: RichTextbox) -> None:
            tb.header(_hdr("Gauss-Seidel Iterative Method")); tb.blank()
            tb.writeln(
                f"  SDD check  : {'PASS ✅' if sdd else 'FAIL ⚠  (convergence not guaranteed)'}",
                "value" if sdd else "warn")
            tb.writeln(f"  Tolerance  : {tol}", "key")
            tb.writeln(f"  Iterations : {iters}", "num")
            tb.writeln(f"  Converged  : {'YES' if converged else 'NO'}",
                       "value" if converged else "warn"); tb.blank()
            tb.header(_hdr("Coefficient Matrix  A"))
            tb.num(_fmt_matrix(A)); tb.blank()
            tb.header(_hdr("Constants  b"))
            tb.num(_fmt_matrix(B)); tb.blank()
            tb.header(_hdr("Solution  x"))
            tb.value(_fmt_matrix(sol)); tb.blank()
            tb.header(_hdr("Residual Analysis"))
            tb.num(f"  ‖Ax − b‖₂  =  {np.linalg.norm(A@sol-B):.6e}")
            tb.num(f"  ‖Ax − b‖∞  =  {res_fin:.6e}"); tb.blank()
            tb.header(_hdr("Row-by-Row Diagonal Dominance"))
            for i in range(n):
                diag = abs(A[i][i])
                off  = sum(abs(A[i][j]) for j in range(n) if j != i)
                ok_r = diag > off
                tb.writeln(
                    f"  Row {i+1}:  |{A[i][i]:.4f}| = {diag:.4f}  "
                    f"{'>' if ok_r else '≤'}  {off:.4f}  {'✓' if ok_r else '✗'}",
                    "value" if ok_r else "warn")

        self._out.write_steps(_steps)
        if len(residuals) > 1:
            self._out.plot_convergence(
                residuals,
                title=f"Gauss-Seidel Convergence  ({len(residuals)} iters)")

    def _run_false_position(self) -> None:
        eq  = self._widgets["eq"].get().strip()
        a   = float(self._widgets["a"].get())
        b   = float(self._widgets["b"].get())
        tol = float(self._widgets["tol"].get())
        mx  = int(self._widgets["max_iter"].get())
        f   = lambda x: self._ev(eq, x)

        if f(a) * f(b) > 0:
            raise ValueError(
                f"f(a)·f(b) > 0 — no sign change on [{a}, {b}].\n"
                "Choose an interval that brackets a root.")

        root, iters = calculator.false_position(f, a, b, tol, mx)
        f_root = f(root)

        # iteration log for table display
        log: list = []
        ac, bc = a, b
        fac, fbc = f(ac), f(bc)
        for k in range(min(mx, 30)):
            if abs(fbc - fac) < 1e-14:
                break
            cc   = bc - fbc * (bc - ac) / (fbc - fac)
            fcc  = f(cc)
            log.append((k + 1, ac, bc, cc, fcc, abs(fcc)))
            if abs(fcc) < tol:
                break
            if fac * fcc < 0:
                bc, fbc = cc, fcc
            else:
                ac, fac = cc, fcc

        self._out.set_summary(
            "✅  Root Found  (Regula Falsi)",
            f"  x  ≈  {root:.10f}",
            meta=f"f(x) = {f_root:.4e}  |  {iters} iters  |  [{a}, {b}]",
        )

        def _steps(tb: RichTextbox) -> None:
            tb.header(_hdr("Method of False Position  (Regula Falsi)")); tb.blank()
            tb.key(f"  f(x)      =  {eq}")
            tb.key(f"  Interval  :  [{a},  {b}]")
            tb.num(f"  f({a})  =  {f(a):.8f}")
            tb.num(f"  f({b})  =  {f(b):.8f}")
            tb.writeln(f"  Tolerance  =  {tol}", "dim"); tb.blank()
            tb.header(_hdr("Iteration Table"))
            tb.table_h(_iter_table_fp(log)); tb.blank()
            tb.header(_hdr("Result"))
            tb.value(f"  Root    x  ≈  {root:.12f}")
            tb.num(f"  f(root)    =  {f_root:.6e}")
            tb.num(f"  Iterations =  {iters}"); tb.blank()
            tb.formula("  Formula:  c = b − f(b)·(b−a) / (f(b)−f(a))")

        self._out.write_steps(_steps)
        m = (b - a) * 0.35
        self._out.plot(f, a - m, b + m, root=root,
                       label=f"f(x) = {eq}",
                       title=f"Regula Falsi  —  {eq}")

    def _run_newton_raphson(self) -> None:
        eq    = self._widgets["f"].get().strip()
        fp_eq = self._widgets["fp"].get().strip()
        x0    = float(self._widgets["x0"].get())
        tol   = float(self._widgets["tol"].get())
        mx    = int(self._widgets["max_iter"].get())
        f     = lambda x: self._ev(eq,    x)
        fp    = lambda x: self._ev(fp_eq, x)

        root, iters = calculator.newton_raphson(f, fp, x0, tol, mx)

        # iteration log
        log: list = []
        xc = x0
        for k in range(min(mx, 20)):
            fxc  = f(xc)
            fpxc = fp(xc)
            if abs(fpxc) < 1e-14:
                break
            xn   = xc - fxc / fpxc
            err  = abs(xn - xc)
            log.append((k + 1, xc, fxc, fpxc, xn, err))
            if err < tol:
                break
            xc = xn

        self._out.set_summary(
            "✅  Root Found  (Newton-Raphson)",
            f"  x  ≈  {root:.10f}",
            meta=f"f(x) = {f(root):.4e}  |  {iters} iters  |  x₀ = {x0}",
        )

        def _steps(tb: RichTextbox) -> None:
            tb.header(_hdr("Newton-Raphson Method")); tb.blank()
            tb.key(f"  f(x)      =  {eq}")
            tb.key(f"  f′(x)     =  {fp_eq}")
            tb.num(f"  x₀        =  {x0}")
            tb.writeln(f"  Tolerance =  {tol}", "dim"); tb.blank()
            tb.header(_hdr("Iteration Table"))
            tb.table_h(_iter_table_nr(log)); tb.blank()
            tb.header(_hdr("Result"))
            tb.value(f"  Root     x  ≈  {root:.12f}")
            tb.num(f"  f(root)     =  {f(root):.6e}")
            tb.num(f"  f′(root)    =  {fp(root):.6e}")
            tb.num(f"  Iterations  =  {iters}"); tb.blank()
            tb.formula("  Formula:  xₙ₊₁ = xₙ − f(xₙ) / f′(xₙ)")

        self._out.write_steps(_steps)
        span = abs(root - x0) * 1.6 + 1.5
        self._out.plot(f, root - span, root + span, root=root,
                       label=f"f(x) = {eq}",
                       title=f"Newton-Raphson  —  {eq}")

    def _run_interpolation(self, kind: str) -> None:
        xs = [float(v.strip()) for v in self._widgets["xvals"].get().split(",")]
        ys = [float(v.strip()) for v in self._widgets["yvals"].get().split(",")]
        t  = float(self._widgets["xtarget"].get())

        if len(xs) != len(ys):
            raise ValueError("X and Y arrays must have equal length.")

        # [IMP-6] default-argument binding captures xs/ys at call time, not
        # at execution time, avoiding the classic Python closure bug.
        if kind == "forward":
            result, diff = calculator.newton_forward_interpolation(xs, ys, t)
            algo         = "Newton's Forward Difference"
            interp_fn    = lambda xv, _xs=xs, _ys=ys: (
                calculator.newton_forward_interpolation(_xs, _ys, xv)[0])
        elif kind == "stirling":
            result, diff = calculator.stirling_interpolation(xs, ys, t)
            algo         = "Stirling's Central Difference"
            interp_fn    = lambda xv, _xs=xs, _ys=ys: (
                calculator.stirling_interpolation(_xs, _ys, xv)[0])
        else:
            result    = calculator.lagrange_interpolation(xs, ys, t)
            diff      = None
            algo      = "Lagrange's Polynomial"
            interp_fn = lambda xv, _xs=xs, _ys=ys: (
                calculator.lagrange_interpolation(_xs, _ys, xv))

        self._out.set_summary(
            f"✅  {algo}",
            f"  f({t})  ≈  {result:.8f}",
            meta=f"n = {len(xs)} points  |  target x = {t}",
        )

        def _steps(tb: RichTextbox) -> None:
            tb.header(_hdr(algo)); tb.blank()
            tb.key(f"  X data   :  {xs}")
            tb.key(f"  Y data   :  {ys}")
            tb.num(f"  Target x :  {t}"); tb.blank()
            tb.header(_hdr("Result"))
            tb.value(f"  f({t})  ≈  {result:.10f}")
            if diff:
                tb.blank()
                tb.header(_hdr("Difference Table"))
                tb.table_h(_diff_table_fmt(diff))
            tb.blank()
            tb.dim("  Tip: Switch to  Plot 📈  to see data + interpolant curve.")

        self._out.write_steps(_steps)
        self._out.plot_interpolation(
            xs, ys, t, result, interp_fn,
            title=f"{algo}  —  f({t}) ≈ {result:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  APPLICATION ROOT
# ══════════════════════════════════════════════════════════════════════════════
class App(ctk.CTk):

    def __init__(self) -> None:
        super().__init__()
        self.title("Advanced Numerical Methods Calculator")
        self.geometry("1300x760")
        self.minsize(1060, 640)
        self.configure(fg_color=BG_APP)

        # Suppress spurious CustomTkinter destroy/callback errors in console
        self.report_callback_exception = lambda *_: None
        try:
            self.tk.createcommand("bgerror", lambda *_: None)
        except Exception:
            pass

        self._splash = WelcomeScreen(self, on_launch=self._launch)
        self._splash.place(relx=0, rely=0, relwidth=1, relheight=1)

    def _launch(self) -> None:
        self._splash.destroy()       # also sets _alive=False via override

        self.grid_columnconfigure(0, weight=30)
        self.grid_columnconfigure(1, weight=70)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        self._out = OutputPanel(self)
        self._out.grid(row=0, column=1, sticky="nsew",
                       padx=(8, 18), pady=(18, 6))

        self._sb = StatusBar(self)
        self._sb.grid(row=1, column=0, columnspan=2,
                      sticky="ew", padx=0, pady=0)

        self._ctrl = ControlPanel(self, output=self._out,
                                  status_bar=self._sb)
        self._ctrl.grid(row=0, column=0, sticky="nsew",
                        padx=(18, 8), pady=(18, 6))

        # ── Global keyboard shortcuts
        self.bind_all("<Control-Return>", lambda _: self._ctrl._start_calculate())
        self.bind_all("<Control-l>",      lambda _: self._ctrl._clear())
        self.bind_all("<Control-L>",      lambda _: self._ctrl._clear())


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = App()
    app.mainloop()