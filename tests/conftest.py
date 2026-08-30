"""Shared pytest fixtures/helpers for the CRE engine test suite."""

import numpy as np
import pytest
from core.kinetics import integrated_concentration


@pytest.fixture
def make_synthetic_data():
    """
    Returns a helper function that generates exact (noise-free)
    concentration-time data for a known (CA0, k, n), using the SAME
    integrated_concentration function the engine itself uses.

    This is intentional: it means these tests check that the fitting
    procedure (validation -> model_selection -> calculations) can
    correctly RECOVER known parameters, not that our forward model and
    our fitting model happen to independently agree by chance.
    """
    def _make(CA0, k, n, t):
        t = np.asarray(t, dtype=float)
        CA = integrated_concentration(t, CA0, k, n)
        return t, CA
    return _make
