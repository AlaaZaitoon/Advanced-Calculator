"""
calculator.py — Numerical Methods Engine
=========================================
Backend math engine for the Advanced Numerical Methods Calculator.

Methods implemented
───────────────────
  1. Doolittle's LU Decomposition         (direct solver)
  2. Gauss-Seidel Iteration               (iterative solver)
  3. Method of False Position / Regula Falsi (root-finding, bracketed)
  4. Newton-Raphson                        (root-finding, open)
  5. Newton's Forward Difference Interpolation
  6. Stirling's Central Difference Interpolation
  7. Lagrange Interpolation

All public functions are fully type-annotated and validate their inputs
before any numerical work begins so the GUI receives clean, descriptive
exceptions rather than raw NumPy/Python tracebacks.
"""

from __future__ import annotations

__all__ = [
    "doolittle_lu_decomposition",
    "gauss_seidel",
    "is_strictly_diagonally_dominant",
    "false_position",
    "newton_raphson",
    "newton_forward_interpolation",
    "stirling_interpolation",
    "lagrange_interpolation",
]

from typing import Callable
import numpy as np

# Tolerance used throughout for "effectively zero" pivot/denominator checks.
_EPS = 1e-14


# ─────────────────────────────────────────────────────────────────────────────
#  1. Doolittle LU Decomposition
# ─────────────────────────────────────────────────────────────────────────────
def doolittle_lu_decomposition(
    A: np.ndarray | list,
    B: np.ndarray | list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve **A x = B** via Doolittle LU decomposition (without pivoting).

    Decomposes A into a unit lower-triangular matrix **L** (ones on diagonal)
    and upper-triangular **U** such that A = L U, then solves:

    * **L y = B**  (forward substitution)
    * **U x = y**  (back substitution)

    Parameters
    ----------
    A : array_like, shape (n, n)
        Coefficient matrix.  Must be square and non-singular.
    B : array_like, shape (n,) or (n, 1)
        Right-hand-side vector.

    Returns
    -------
    L : ndarray, shape (n, n)   — lower-triangular factor
    U : ndarray, shape (n, n)   — upper-triangular factor
    x : ndarray, shape (n,)     — solution vector

    Raises
    ------
    ValueError
        Non-square A, shape mismatch, or singular matrix (zero pivot).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float).ravel()

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(
            f"Matrix A must be square; received shape {A.shape}."
        )
    n = A.shape[0]
    if B.shape[0] != n:
        raise ValueError(
            f"Vector B must have {n} element(s) to match A ({n}×{n}); "
            f"received {B.shape[0]}."
        )

    L = np.eye(n, dtype=float)        # Doolittle: L diagonal is always 1
    U = np.zeros((n, n), dtype=float)

    for i in range(n):
        # ── Upper row of U  (vectorised slice)
        U[i, i:] = A[i, i:] - L[i, :i] @ U[:i, i:]

        if abs(U[i, i]) < _EPS:
            raise ValueError(
                f"Zero pivot at U[{i},{i}] — matrix is singular or requires "
                "row pivoting.  Try reordering the equations."
            )

        # ── Lower column of L  (vectorised slice)
        if i < n - 1:
            L[i + 1:, i] = (A[i + 1:, i] - L[i + 1:, :i] @ U[:i, i]) / U[i, i]

    # Forward substitution  L y = B  (L has unit diagonal → L[i,i] = 1)
    y = np.zeros(n, dtype=float)
    for i in range(n):
        y[i] = B[i] - L[i, :i] @ y[:i]

    # Back substitution  U x = y
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]

    return L, U, x


# ─────────────────────────────────────────────────────────────────────────────
#  2. Gauss-Seidel Iteration
# ─────────────────────────────────────────────────────────────────────────────
def is_strictly_diagonally_dominant(A: np.ndarray) -> bool:
    """Return **True** if *A* is strictly diagonally dominant.

    A sufficient (not necessary) condition for Gauss-Seidel convergence.
    Each row must satisfy  |a_ii| > Σ_{j≠i} |a_ij|.
    """
    A = np.asarray(A, dtype=float)
    diag     = np.abs(np.diag(A))
    off_sums = np.sum(np.abs(A), axis=1) - diag
    return bool(np.all(diag > off_sums))


def gauss_seidel(
    A:             np.ndarray | list,
    b:             np.ndarray | list,
    x0:            np.ndarray | list | None = None,
    tolerance:     float = 1e-6,
    max_iterations: int  = 1000,
) -> tuple[np.ndarray, int, bool]:
    """Solve **A x = b** iteratively with the Gauss-Seidel method.

    Parameters
    ----------
    A             : array_like (n, n) — coefficient matrix
    b             : array_like (n,)   — right-hand side
    x0            : array_like (n,)   — initial guess (zeros if ``None``)
    tolerance     : float             — ‖xₙ − xₙ₋₁‖∞ convergence threshold
    max_iterations: int               — hard iteration cap

    Returns
    -------
    x          : ndarray — solution vector at termination
    iterations : int     — number of iterations performed
    converged  : bool    — True if tolerance was reached before the cap
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square; received shape {A.shape}.")
    n = A.shape[0]
    if b.shape[0] != n:
        raise ValueError(
            f"b must have {n} elements to match {n}×{n} matrix A; "
            f"received {b.shape[0]}."
        )

    zero_rows = np.flatnonzero(np.isclose(np.diag(A), 0.0))
    if zero_rows.size:
        raise ValueError(
            f"Zero diagonal element(s) at row(s) {zero_rows.tolist()}.  "
            "Reorder equations so no diagonal entry is zero."
        )

    x = np.zeros(n, dtype=float) if x0 is None else np.asarray(x0, dtype=float).copy()

    # Pre-compute reciprocals of the diagonal to avoid repeated division
    diag_inv = 1.0 / np.diag(A)

    for iteration in range(1, max_iterations + 1):
        x_prev = x.copy()
        for i in range(n):
            # Sum all terms except the diagonal
            sigma = A[i, :i] @ x[:i] + A[i, i + 1:] @ x[i + 1:]
            x[i]  = (b[i] - sigma) * diag_inv[i]

        if np.linalg.norm(x - x_prev, ord=np.inf) < tolerance:
            return x, iteration, True

    return x, max_iterations, False


# ─────────────────────────────────────────────────────────────────────────────
#  3. Method of False Position (Regula Falsi)
# ─────────────────────────────────────────────────────────────────────────────
def false_position(
    f:             Callable[[float], float],
    a:             float,
    b:             float,
    tolerance:     float = 1e-6,
    max_iterations: int  = 100,
) -> tuple[float, int]:
    """Find a root of *f* on **[a, b]** using the False-Position method.

    Requires a sign change: ``f(a) · f(b) < 0``.

    Parameters
    ----------
    f             : callable     — scalar function f(x)
    a, b          : float        — bracket endpoints (f(a) and f(b) opposite sign)
    tolerance     : float        — |f(c)| stopping criterion
    max_iterations: int          — hard iteration cap

    Returns
    -------
    root       : float — approximated root
    iterations : int   — iterations performed
    """
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise ValueError(
            f"f(a) = {fa:.6g} and f(b) = {fb:.6g} must have opposite signs.  "
            "Ensure the interval brackets a root."
        )

    c = a  # initialise so we always have a valid return value
    for iteration in range(1, max_iterations + 1):
        fa, fb = f(a), f(b)

        denom = fb - fa
        if abs(denom) < _EPS:
            # Numerically degenerate — return best estimate so far
            return c, iteration

        c  = (a * fb - b * fa) / denom
        fc = f(c)

        if abs(fc) < tolerance:
            return c, iteration

        # Update bracket (Illinois / standard Regula Falsi)
        if fa * fc < 0:
            b = c
        else:
            a = c

    return c, max_iterations


# ─────────────────────────────────────────────────────────────────────────────
#  4. Newton-Raphson Method
# ─────────────────────────────────────────────────────────────────────────────
def newton_raphson(
    f:             Callable[[float], float],
    f_prime:       Callable[[float], float],
    x0:            float,
    tolerance:     float = 1e-6,
    max_iterations: int  = 100,
) -> tuple[float, int]:
    """Find a root of *f* near *x0* using the Newton-Raphson method.

    Parameters
    ----------
    f             : callable — function f(x)
    f_prime       : callable — derivative f′(x)
    x0            : float    — initial guess
    tolerance     : float    — stopping criterion  |xₙ₊₁ − xₙ| < tolerance
    max_iterations: int      — hard iteration cap

    Returns
    -------
    root       : float — approximated root
    iterations : int   — iterations performed

    Raises
    ------
    ZeroDivisionError
        If f′(x) ≈ 0 during an iteration, making the step undefined.
    """
    x = float(x0)

    for iteration in range(1, max_iterations + 1):
        fx  = f(x)
        fpx = f_prime(x)

        if abs(fpx) < _EPS:
            raise ZeroDivisionError(
                f"Derivative f′({x:.6g}) ≈ 0 at iteration {iteration}.  "
                "Newton-Raphson fails here; try a different starting point."
            )

        x_next = x - fx / fpx

        if abs(x_next - x) < tolerance or abs(f(x_next)) < tolerance:
            return x_next, iteration

        x = x_next

    return x, max_iterations


# ─────────────────────────────────────────────────────────────────────────────
#  5. Newton's Forward Difference Interpolation
# ─────────────────────────────────────────────────────────────────────────────
def newton_forward_interpolation(
    x_vals: list[float] | np.ndarray,
    y_vals: list[float] | np.ndarray,
    x:      float,
) -> tuple[float, list[list[float]]]:
    """Interpolate *y* at *x* using Newton's Forward Difference formula.

    Requires **equally spaced** x values.

    Parameters
    ----------
    x_vals : array_like (n,) — independent variable values (equally spaced)
    y_vals : array_like (n,) — dependent variable values
    x      : float           — target interpolation point

    Returns
    -------
    result     : float               — interpolated y value
    diff_table : list[list[float]]   — full forward-difference table
                                        diff_table[k] is the k-th order differences
    """
    x_vals = list(x_vals)
    y_vals = list(y_vals)
    n = len(x_vals)

    if n < 2:
        raise ValueError("At least 2 data points are required.")
    if len(y_vals) != n:
        raise ValueError("x_vals and y_vals must have the same length.")

    h = x_vals[1] - x_vals[0]
    if abs(h) < _EPS:
        raise ValueError("x values must be distinct (step h ≠ 0).")

    # Verify equal spacing
    for i in range(1, n - 1):
        if not np.isclose(x_vals[i + 1] - x_vals[i], h, rtol=1e-5):
            raise ValueError(
                "x values must be equally spaced for Newton's Forward formula."
            )

    # Build full forward-difference table
    diff: list[list[float]] = [list(y_vals)]
    for k in range(1, n):
        prev = diff[-1]
        diff.append([prev[i + 1] - prev[i] for i in range(len(prev) - 1)])

    x0    = x_vals[0]
    u     = (x - x0) / h
    result      = diff[0][0]
    u_product   = 1.0
    factorial_k = 1.0

    for k in range(1, n):
        if not diff[k]:        # empty column — table exhausted
            break
        u_product   *= (u - (k - 1))
        factorial_k *= k
        result      += (u_product / factorial_k) * diff[k][0]

    return result, diff


# ─────────────────────────────────────────────────────────────────────────────
#  6. Stirling's Central Difference Interpolation
# ─────────────────────────────────────────────────────────────────────────────
def stirling_interpolation(
    x_vals: list[float] | np.ndarray,
    y_vals: list[float] | np.ndarray,
    x:      float,
) -> tuple[float, list[list[float]]]:
    """Interpolate *y* at *x* using Stirling's Central Difference formula.

    Best accuracy is achieved when *x* is near the **centre** of the table.
    Requires equally spaced x values.

    Parameters
    ----------
    x_vals : array_like (n,) — equally spaced independent variable values
    y_vals : array_like (n,) — dependent variable values
    x      : float           — target interpolation point

    Returns
    -------
    result     : float             — interpolated y value
    diff_table : list[list[float]] — central-difference table
    """
    x_vals = list(x_vals)
    y_vals = list(y_vals)
    n = len(x_vals)

    if n < 2:
        raise ValueError("At least 2 data points are required.")
    if len(y_vals) != n:
        raise ValueError("x_vals and y_vals must have the same length.")

    h = x_vals[1] - x_vals[0]
    if abs(h) < _EPS:
        raise ValueError("x values must be distinct (step h ≠ 0).")

    for i in range(1, n - 1):
        if not np.isclose(x_vals[i + 1] - x_vals[i], h, rtol=1e-5):
            raise ValueError(
                "x values must be equally spaced for Stirling's formula."
            )

    # Build difference table (same structure as Newton forward)
    diff: list[list[float]] = [list(y_vals)]
    for k in range(1, n):
        prev = diff[-1]
        diff.append([prev[i + 1] - prev[i] for i in range(len(prev) - 1)])

    # Nearest table point as origin
    idx = int(np.argmin([abs(x_vals[i] - x) for i in range(n)]))
    x0  = x_vals[idx]
    u   = (x - x0) / h

    result = diff[0][idx]

    # 1st-order term:  u · μδy₀  = u · (δy_{-½} + δy_{+½}) / 2
    if idx > 0 and idx < len(diff[1]):
        mu_delta1 = (diff[1][idx - 1] + diff[1][idx]) / 2.0
        result   += u * mu_delta1

    # 2nd-order term:  (u²/2) · δ²y₀
    if idx >= 1 and len(diff) > 2 and (idx - 1) < len(diff[2]):
        result += (u ** 2 / 2.0) * diff[2][idx - 1]

    # 3rd-order term:  u(u²-1)/6 · μδ³y₀
    if idx >= 2 and len(diff) > 3 and (idx - 2) < len(diff[3]) and (idx - 1) < len(diff[3]):
        mu_delta3 = (diff[3][idx - 2] + diff[3][idx - 1]) / 2.0
        result   += (u * (u ** 2 - 1) / 6.0) * mu_delta3

    # 4th-order term:  u²(u²-1)/24 · δ⁴y₀
    if idx >= 2 and len(diff) > 4 and (idx - 2) < len(diff[4]):
        result += (u ** 2 * (u ** 2 - 1) / 24.0) * diff[4][idx - 2]

    return result, diff


# ─────────────────────────────────────────────────────────────────────────────
#  7. Lagrange Interpolation
# ─────────────────────────────────────────────────────────────────────────────
def lagrange_interpolation(
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
    target_x: float,
) -> float:
    """Interpolate *y* at *target_x* using Lagrange's basis-polynomial formula.

    Handles unequally spaced data.  Complexity is O(n²) per evaluation.

    Parameters
    ----------
    x_values : array_like (n,) — independent variable values (must be unique)
    y_values : array_like (n,) — dependent variable values
    target_x : float           — point at which to interpolate

    Returns
    -------
    float — interpolated y value

    Raises
    ------
    ValueError
        Empty arrays, mismatched lengths, or duplicate x values.
    """
    x_values = list(x_values)
    y_values = list(y_values)
    n = len(x_values)

    if n == 0:
        raise ValueError("Input data arrays cannot be empty.")
    if n != len(y_values):
        raise ValueError("x_values and y_values must have the same length.")
    if len(set(x_values)) != n:
        raise ValueError(
            "All x values must be unique — duplicates cause division by zero."
        )

    # Vectorised Lagrange evaluation using NumPy broadcasting
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    t     = float(target_x)

    result = 0.0
    for i in range(n):
        # Mask out the i-th point
        mask   = np.arange(n) != i
        num    = np.prod(t       - x_arr[mask])
        den    = np.prod(x_arr[i] - x_arr[mask])
        result += y_arr[i] * num / den

    return float(result)