"""
model_selection.py
===================

Determines the reaction order n (and the corresponding rate constant
k) from concentration-time data.

Method, and why it is appropriate for this data
--------------------------------------------------
The uploaded solutions manual demonstrates Levenspiel's classic
"integral method of analysis" repeatedly (e.g. problems 3.17, 3.21,
3.25): guess a reaction order, transform (linearize) the data
according to the integrated rate law for that order, plot it, and see
whether it falls on a straight line. Whichever guessed order
straightens the data out is taken as the reaction order, and the slope
of that line gives k.

The project brief asks for TWO things the naive version of that method
does not give:

  1. Support for *continuous* (non-integer) order, not just a menu of
     hard-coded 0 / 1 / 2 / 3 formulas.
  2. A "mathematically justified" way to pick the best order, rather
     than blindly trusting whichever transform happens to have the
     highest R^2.

This module addresses both as follows.

Step 1 - continuous generalization of the linearization idea
--------------------------------------------------------------
For ANY order n, the integrated nth-order rate law (kinetics.py,
equation 2) says that the transformed variable

    y(n) = C_A^(1-n)          (n != 1)
    y(1) = ln(C_A)            (n  = 1, the removable singularity)

is a LINEAR function of t: y = a + b*t. This holds for every real n,
not just integers, so "guess n, check linearity" generalizes directly
into a continuous search over n: for each candidate n we run an
ordinary least-squares fit of y(n) against t and get a slope, from
which k follows algebraically (see `_linear_transform_fit`). This is
exactly the manual's method (problems 3.17/3.21/3.25), just made
continuous instead of a fixed menu of guesses.

Step 2 - why comparing R^2 of the raw transforms is NOT valid, and
what we do instead
---------------------------------------------------------------------
The R^2 of the linear fit in step 1 is computed in the TRANSFORMED
space y(n), and that space changes shape with n (C_A^(1-n) for
n = -1 is a very different scale/curvature than for n = 3). A high
R^2 in one transform is not on equal footing with a high R^2 in
another transform, so picking "whichever transform has the highest
R^2" - which the project brief explicitly warns against - is not
mathematically sound.

Instead, for every candidate n we:
  (a) get a k estimate from the step-1 linear transform (this is fast
      and is exactly the manual's method), then
  (b) use that (n, k) to predict C_A(t) with the ACTUAL integrated
      rate law (kinetics.integrated_concentration) - i.e. we map back
      into real, physical concentration units, and
  (c) score the fit by R^2 between predicted and measured C_A(t), IN
      CONCENTRATION SPACE, which is the same space and the same units
      for every candidate n. This makes the comparison across
      different orders fair.

We search over continuous n by first scanning a coarse grid (to avoid
missing the right neighborhood / getting stuck in a local optimum),
then polishing the best grid point with a bounded local optimizer
(`scipy.optimize.minimize_scalar`) that maximizes concentration-space
R^2 (equivalently minimizes concentration-space SSE).

Step 3 - nonlinear refinement
--------------------------------
Linearizing transforms (like ln(C_A) or C_A^(1-n)) distort the error
structure of the data: a small absolute error in C_A near the end of
the reaction (low concentration) becomes a much larger error after a
log or negative-power transform than the same absolute error at high
concentration. That means the step-1/step-2 estimate, while a good and
literature-standard starting point, is not the statistically optimal
fit in real concentration units.

So as a final step we run a genuine nonlinear least-squares
optimization directly on C_A(t) (not any transformed variable),
starting from the step-2 estimate, using
`scipy.optimize.minimize` (Nelder-Mead) to jointly refine (n, k) by
minimizing the concentration-space sum of squared errors. This gives
the final reported order and rate constant, with C_AO held fixed at
the experimental t=0 value (it is a measured initial condition, not a
fitted kinetic parameter).
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.optimize import minimize_scalar, minimize

from . import kinetics

DEFAULT_N_BOUNDS = (-1.0, 4.0)
_COARSE_GRID_STEP = 0.1


@dataclass
class OrderCandidate:
    """One tested reaction order and how well it fit the data."""
    n: float
    k: float
    r2_concentration_space: float


@dataclass
class OrderSelectionResult:
    """Final result of the order-determination procedure."""
    best_n: float
    best_k: float
    r2: float
    candidates_scanned: List[OrderCandidate]


def _linear_transform_fit(t: np.ndarray, CA: np.ndarray, n: float) -> Tuple[float, float]:
    """
    Step 1: ordinary least-squares fit of the linearized nth-order
    integrated rate law (see module docstring). Returns (k, r2) where
    r2 is the R^2 of the fit IN THE TRANSFORMED SPACE (used only as a
    fallback quality flag - the real, comparable score is computed
    later in concentration space).
    """
    if kinetics.is_first_order(n):
        y = np.log(CA)
    else:
        y = np.power(CA, 1.0 - n)

    # y = a + b*t  (ordinary least squares, free intercept - we do not
    # force the intercept through C_AO^(1-n)/ln(C_AO) because real
    # data has noise in the t=0 measurement too; this matches the
    # "plot and take the slope" approach in the source material).
    b, a = np.polyfit(t, y, deg=1)

    y_pred = a + b * t
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_transform = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    if kinetics.is_first_order(n):
        k = -b
    else:
        k = b / (1.0 - n)

    return k, r2_transform


def _r2_concentration_space(t: np.ndarray, CA: np.ndarray, CA0: float,
                             k: float, n: float) -> float:
    """Step 2: R^2 between predicted and measured C_A(t), in real
    concentration units - the fair, common basis for comparing
    different candidate orders."""
    CA_pred = kinetics.integrated_concentration(t, CA0, k, n)
    ss_res = np.sum((CA - CA_pred) ** 2)
    ss_tot = np.sum((CA - np.mean(CA)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def _score_order(t: np.ndarray, CA: np.ndarray, CA0: float, n: float) -> OrderCandidate:
    """Fit k for a given n (step 1) and score it in concentration
    space (step 2)."""
    try:
        k, _ = _linear_transform_fit(t, CA, n)
    except (ZeroDivisionError, FloatingPointError):
        return OrderCandidate(n=n, k=np.nan, r2_concentration_space=-np.inf)

    if not np.isfinite(k) or k <= 0:
        # A non-positive or non-finite rate constant is not physical
        # for an irreversible decay -r_A = k*C_A^n with k>0; reject
        # this candidate order outright rather than let it "win" on a
        # spurious fit.
        return OrderCandidate(n=n, k=k, r2_concentration_space=-np.inf)

    r2 = _r2_concentration_space(t, CA, CA0, k, n)
    return OrderCandidate(n=n, k=k, r2_concentration_space=r2)


def _nonlinear_refine(t: np.ndarray, CA: np.ndarray, CA0: float,
                       n_init: float, k_init: float,
                       n_bounds: Tuple[float, float]) -> Tuple[float, float]:
    """Step 3: joint nonlinear least-squares polish of (n, k) directly
    against measured concentration, starting from the step-2 estimate.
    """
    def sse(params):
        n, log_k = params
        k = np.exp(log_k)  # optimize in log(k) to keep k > 0 automatically
        if n_bounds[0] <= n <= n_bounds[1]:
            CA_pred = kinetics.integrated_concentration(t, CA0, k, n)
            return np.sum((CA - CA_pred) ** 2)
        return np.inf

    x0 = [n_init, np.log(max(k_init, 1e-12))]
    result = minimize(sse, x0, method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 2000})

    n_final, log_k_final = result.x
    n_final = float(np.clip(n_final, n_bounds[0], n_bounds[1]))
    k_final = float(np.exp(log_k_final))
    return n_final, k_final


def determine_best_order(t: np.ndarray, CA: np.ndarray,
                          n_bounds: Tuple[float, float] = DEFAULT_N_BOUNDS
                          ) -> OrderSelectionResult:
    """
    Full three-step order-determination procedure described in the
    module docstring.

    Parameters
    ----------
    t : np.ndarray
        Validated, strictly increasing time array, t[0] == 0.
    CA : np.ndarray
        Validated, strictly positive concentration array.
    n_bounds : (float, float)
        Search range for the reaction order. Default (-1, 4) covers
        essentially all reaction orders encountered in practice
        (Levenspiel Ch. 3 notes most real reactions fall between
        0 and 3, with a wider net cast here for safety).

    Returns
    -------
    OrderSelectionResult
    """
    CA0 = CA[0]
    candidates: List[OrderCandidate] = []

    # --- coarse grid scan (avoids missing the right neighborhood) ---
    grid = np.arange(n_bounds[0], n_bounds[1] + 1e-9, _COARSE_GRID_STEP)
    for n in grid:
        candidates.append(_score_order(t, CA, CA0, float(n)))

    best_grid = max(candidates, key=lambda c: c.r2_concentration_space)

    # --- bounded local optimizer around the best grid point ----------
    def neg_r2(n):
        return -_score_order(t, CA, CA0, n).r2_concentration_space

    lo = max(n_bounds[0], best_grid.n - _COARSE_GRID_STEP)
    hi = min(n_bounds[1], best_grid.n + _COARSE_GRID_STEP)
    opt = minimize_scalar(neg_r2, bounds=(lo, hi), method="bounded",
                           options={"xatol": 1e-6})
    n_step2 = float(opt.x)
    step2_candidate = _score_order(t, CA, CA0, n_step2)

    # If the local optimizer somehow did worse than the coarse grid
    # (can happen at the very edge of the bounds), fall back to the
    # grid winner.
    if step2_candidate.r2_concentration_space < best_grid.r2_concentration_space:
        step2_candidate = best_grid

    # --- step 3: nonlinear refinement in real concentration units ----
    n_final, k_final = _nonlinear_refine(
        t, CA, CA0, step2_candidate.n, step2_candidate.k, n_bounds
    )
    r2_final = _r2_concentration_space(t, CA, CA0, k_final, n_final)

    # Safety net: nonlinear refinement is a local optimizer and can in
    # rare cases (e.g. very noisy data) land somewhere worse than the
    # step-2 estimate. Never report a final answer worse than step 2.
    if r2_final < step2_candidate.r2_concentration_space:
        n_final, k_final, r2_final = (
            step2_candidate.n, step2_candidate.k,
            step2_candidate.r2_concentration_space,
        )

    return OrderSelectionResult(
        best_n=n_final,
        best_k=k_final,
        r2=r2_final,
        candidates_scanned=candidates,
    )
