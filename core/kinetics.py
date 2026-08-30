"""
kinetics.py
===========

Pure nth-order batch-reactor kinetics equations. No fitting, no I/O -
just the math, so it can be tested and reused independently of how
(n, k) were determined.

Reference
---------
Levenspiel, "Chemical Reaction Engineering", Chapter 3
("Interpretation of Batch Reactor Data"), constant-volume batch
reactor, single irreversible reaction:

    -r_A = -dC_A/dt = k * C_A^n                                  (1)

This single expression is what the project brief calls "general nth
order kinetics". Every quantity below (integrated law, rate, half-life,
conversion) is obtained by direct calculus on equation (1) - nothing
is hard-coded per specific integer order.

The n = 1 case is handled as a separate branch because it is a genuine
mathematical singularity of the general solution below (division by
1-n), not because we are special-casing "common" orders. This is the
same treatment Levenspiel uses in the source material.

Confirmed against the uploaded solutions manual
-------------------------------------------------
* Problem 3.23 uses (their numbering) "Eq 33a":
      (n-1)*k*t = C_AO^(1-n) * [ (C_A/C_AO)^(1-n) - 1 ]
  which is algebraically identical to equation (2) below.
* Problem 3.19 uses (their numbering) "Eq 29", the time for complete
  conversion of an nth-order reaction (n<1):
      t = C_AO^(1-n) / [ (1-n) * k ]
  which is the n<1, C_A->0 limit of equation (2) below (see
  `time_for_complete_conversion`).
* Problem 3.27 uses the differential form  -ΔC_A/Δt = k * C_bar_A^n,
  the finite-difference analogue of equation (1), used here in
  `model_selection.py` as a cross-check method.
"""

import numpy as np

# Any n closer than this to 1.0 is treated as exactly first order, to
# avoid dividing by (1-n) ~ 0 and blowing up numerically.
_FIRST_ORDER_TOLERANCE = 1e-6


def is_first_order(n: float) -> bool:
    return abs(n - 1.0) < _FIRST_ORDER_TOLERANCE


def integrated_concentration(t, CA0: float, k: float, n: float):
    """
    C_A(t) predicted by the integrated nth-order rate law.

    Derivation (Levenspiel Ch. 3, confirmed as "Eq 33a" in the source
    solutions manual, problem 3.23):

        -dC_A/dt = k * C_A^n
        => -C_A^(-n) dC_A = k dt
        => integrating from (0, C_AO) to (t, C_A):

               C_AO^(1-n) - C_A^(1-n)
               ------------------------ = k*t                      (2)
                      (1 - n)

        => C_A(t) = [ C_AO^(1-n) - (1-n)*k*t ]^(1/(1-n))            (2a)

    For n = 1 the (1-n) denominator vanishes; integrating
    -dC_A/C_A = k dt directly instead gives the standard first-order
    law:

        C_A(t) = C_AO * exp(-k*t)                                  (3)

    For n < 1, equation (2a) reaches C_A = 0 at a finite time
    (see `time_for_complete_conversion`); beyond that time the
    bracketed term would go negative, which is not physical (the
    reactant cannot un-react), so the prediction is clipped to 0.

    Parameters
    ----------
    t : float or array-like
        Time(s) at which to evaluate predicted concentration.
    CA0 : float
        Initial concentration (concentration at t=0).
    k : float
        Rate constant.
    n : float
        Reaction order.

    Returns
    -------
    float or np.ndarray
        Predicted concentration(s), same shape as `t`.
    """
    t = np.asarray(t, dtype=float)
    scalar_input = (t.ndim == 0)
    t = np.atleast_1d(t)

    if is_first_order(n):
        CA = CA0 * np.exp(-k * t)
    else:
        bracket = CA0 ** (1.0 - n) - (1.0 - n) * k * t
        # Physical clipping: concentration cannot be negative, and for
        # n < 1 the model predicts exact completion at a finite time
        # (see time_for_complete_conversion). Beyond that time the
        # bracket goes negative; clip it to 0 before the fractional
        # power to avoid complex results.
        bracket = np.clip(bracket, a_min=0.0, a_max=None)
        with np.errstate(invalid="ignore", divide="ignore"):
            CA = bracket ** (1.0 / (1.0 - n))
        CA = np.nan_to_num(CA, nan=0.0)

    return CA.item() if scalar_input else CA


def rate_of_reaction(CA, k: float, n: float):
    """
    Instantaneous rate of reaction, directly from the defining rate law
    (Levenspiel Ch. 3, equation (1) above):

        -r_A = k * C_A^n

    Parameters
    ----------
    CA : float or array-like
        Concentration(s) at which to evaluate the rate.
    k : float
    n : float

    Returns
    -------
    float or np.ndarray
        -r_A, in concentration/time units consistent with k and C_A.
    """
    CA = np.asarray(CA, dtype=float)
    return k * np.power(CA, n)


def half_life(CA0: float, k: float, n: float) -> float:
    """
    Half-life t_(1/2): the time at which C_A = C_AO / 2.

    Derived directly from equation (2) above by substituting
    C_A = C_AO/2 - this is NOT an independently hard-coded formula per
    order, it is the same general integrated law evaluated at one
    specific concentration:

        C_AO^(1-n) - (C_AO/2)^(1-n)
        ---------------------------- = k * t_(1/2)
                 (1 - n)

        => t_(1/2) = C_AO^(1-n) * (1 - 2^(n-1)) / [ (1-n) * k ]
                    = [ 2^(n-1) - 1 ] / [ (n-1) * k * C_AO^(n-1) ]   (4)

    For n = 1 this ratio is 0/0; taking the limit n->1 (L'Hopital, or
    equivalently integrating -dC_A/C_A = k dt directly) gives the
    standard result:

        t_(1/2) = ln(2) / k                                        (5)

    Parameters
    ----------
    CA0 : float
        Initial concentration.
    k : float
        Rate constant.
    n : float
        Reaction order.

    Returns
    -------
    float
        Half-life, in time units consistent with k.
    """
    if is_first_order(n):
        return np.log(2.0) / k

    numerator = (2.0 ** (n - 1.0)) - 1.0
    denominator = (n - 1.0) * k * (CA0 ** (n - 1.0))
    return numerator / denominator


def time_for_complete_conversion(CA0: float, k: float, n: float):
    """
    Time at which C_A reaches exactly 0 (X_A = 1).

    Confirmed in the uploaded solutions manual as "Eq 29", problem
    3.19, valid for n < 1:

        t_complete = C_AO^(1-n) / [ (1-n) * k ]                    (6)

    This is simply equation (2) evaluated at C_A = 0.

    For n >= 1 (within first-order tolerance, or above), C_A approaches
    0 only asymptotically as t -> infinity (the bracket in equation
    (2a) never reaches 0 in finite time), so this function returns
    None (physically: no finite time gives exactly complete conversion
    for n >= 1). Uses the same `is_first_order()` tolerance as the
    rest of this module, so a fitted n that is extremely close to but
    not exactly 1.0 (e.g. n = 0.99999997 from a nonlinear fit) is
    still correctly treated as first order here, instead of returning
    a huge, physically meaningless finite time.

    Parameters
    ----------
    CA0, k, n : as above

    Returns
    -------
    float or None
    """
    if n >= 1.0 or is_first_order(n):
        return None
    return (CA0 ** (1.0 - n)) / ((1.0 - n) * k)


def conversion(CA0: float, CA):
    """
    Fractional conversion of A, the standard Levenspiel definition
    used throughout the source material (e.g. problems 3.3, 3.9, 3.29):

        X_A = (C_AO - C_A) / C_AO

    Parameters
    ----------
    CA0 : float
        Initial concentration.
    CA : float or array-like
        Concentration(s) at which to evaluate conversion.

    Returns
    -------
    float or np.ndarray
        X_A, dimensionless, in [0, 1].
    """
    CA = np.asarray(CA, dtype=float)
    return (CA0 - CA) / CA0
