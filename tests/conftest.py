"""
Shared fixtures for GoF test suite.

Loads injection data and real events once per session to avoid
repeated expensive I/O.
"""
import sys
import os
import pytest
import numpy as np

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def inj_df():
    """Load merged O3+O4 injection dataframe (session-scoped)."""
    from gof_core import load_injections
    return load_injections()


@pytest.fixture(scope="session")
def real_events():
    """Load real GW events with mid-mass cut (session-scoped)."""
    from gof_core import load_real_events
    return load_real_events()


@pytest.fixture(scope="session")
def bin_edges(inj_df, real_events):
    """Compute data-adaptive bin edges (session-scoped)."""
    from gof_core import compute_bin_edges
    return compute_bin_edges(inj_df, real_events)


@pytest.fixture(scope="session")
def data_vec(real_events, bin_edges):
    """Build observed histogram vector from real events."""
    from gof_core import make_histogram
    q05 = [ev['chieff_0.05'] for ev in real_events]
    q95 = [ev['chieff_0.95'] for ev in real_events]
    return make_histogram(q05, q95, bin_edges)


@pytest.fixture(scope="session")
def models():
    """Return the MODELS registry list."""
    from gof_core import MODELS
    return MODELS


# Ensure output directories exist
@pytest.fixture(scope="session", autouse=True)
def ensure_output_dirs():
    """Create output directories for diagnostic plots."""
    os.makedirs("tests/output/physics", exist_ok=True)
