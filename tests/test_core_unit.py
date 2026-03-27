"""
Unit tests for gof_core.py: binning, histograms, and edge cases.

Tests pure-function behaviour without needing real data for most tests.
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMakeHistogram:
    """Tests for make_histogram: normalization, shape, edge cases."""

    def test_output_shape(self):
        from gof_core import make_histogram
        edges = np.linspace(-1, 1, 11)  # 10 bins
        q05 = np.random.uniform(-0.5, 0.5, 50)
        q95 = np.random.uniform(-0.5, 0.5, 50)
        vec = make_histogram(q05, q95, edges)
        assert vec.shape == (20,), f"Expected shape (20,), got {vec.shape}"

    def test_each_half_sums_to_one(self):
        """Each sub-histogram (q05, q95) should be normalized to sum=1."""
        from gof_core import make_histogram
        edges = np.linspace(-1, 1, 11)
        q05 = np.random.uniform(-0.8, 0.8, 100)
        q95 = np.random.uniform(-0.8, 0.8, 100)
        vec = make_histogram(q05, q95, edges)
        n_bins = 10
        np.testing.assert_allclose(vec[:n_bins].sum(), 1.0, atol=1e-12)
        np.testing.assert_allclose(vec[n_bins:].sum(), 1.0, atol=1e-12)

    def test_empty_input(self):
        """Empty arrays should produce a zero vector (no crash)."""
        from gof_core import make_histogram
        edges = np.linspace(-1, 1, 6)
        vec = make_histogram([], [], edges)
        # All zeros because sum is 0 and division is skipped
        np.testing.assert_array_equal(vec, 0.0)

    def test_single_event(self):
        """Single event: one bin gets 1.0, rest get 0."""
        from gof_core import make_histogram
        edges = np.array([0.0, 0.5, 1.0])  # 2 bins
        vec = make_histogram([0.25], [0.75], edges)
        # q05=0.25 -> bin 0; q95=0.75 -> bin 1
        assert vec.shape == (4,)
        assert vec[0] == 1.0  # q05 in first bin
        assert vec[1] == 0.0
        assert vec[2] == 0.0  # q95 in second bin
        assert vec[3] == 1.0


class TestComputeBinEdges:
    """Tests for compute_bin_edges."""

    def test_correct_number_of_edges(self, inj_df, real_events):
        from gof_core import compute_bin_edges
        for n_bins in [5, 10, 20]:
            edges = compute_bin_edges(inj_df, real_events, n_bins=n_bins)
            assert len(edges) == n_bins + 1, (
                f"Expected {n_bins+1} edges, got {len(edges)}"
            )

    def test_edges_monotonically_increasing(self, inj_df, real_events):
        from gof_core import compute_bin_edges
        edges = compute_bin_edges(inj_df, real_events, n_bins=10)
        assert np.all(np.diff(edges) > 0), "Bin edges must be strictly increasing"

    def test_margin_expands_range(self, inj_df, real_events):
        """Non-zero margin should expand the range beyond the data extremes."""
        from gof_core import compute_bin_edges
        edges_no_margin = compute_bin_edges(inj_df, real_events,
                                             n_bins=10, margin_frac=0.0)
        edges_with_margin = compute_bin_edges(inj_df, real_events,
                                               n_bins=10, margin_frac=0.10)
        assert edges_with_margin[0] < edges_no_margin[0]
        assert edges_with_margin[-1] > edges_no_margin[-1]


class TestCallisterPriorWrapper:
    """Tests for the _callister_prior wrapper in gof_core."""

    def test_returns_positive_values(self):
        from gof_core import _callister_prior
        q = np.array([0.5, 0.5, 0.5])
        chieff = np.array([0.0, 0.3, -0.3])
        vals = _callister_prior(q, chieff)
        assert np.all(vals >= 0), "Prior values must be non-negative"
        assert np.all(np.isfinite(vals)), "Prior values must be finite"

    def test_scalar_input(self):
        """Scalar inputs should work without error."""
        from gof_core import _callister_prior
        val = _callister_prior(0.5, 0.1)
        assert np.isfinite(val).all()
        assert (val >= 0).all()

    def test_symmetric_for_equal_mass(self):
        """For q=1, the prior should be symmetric about chieff=0."""
        from gof_core import _callister_prior
        q = np.ones(100)
        chieff = np.linspace(0.01, 0.9, 100)
        pos = _callister_prior(q, chieff)
        neg = _callister_prior(q, -chieff)
        np.testing.assert_allclose(pos, neg, rtol=1e-6,
                                    err_msg="Prior should be symmetric for q=1")


class TestTruncnormPdf:
    """Tests for the _truncnorm_pdf helper."""

    def test_integrates_to_one(self):
        """Truncated normal PDF on [lo, hi] must integrate to 1."""
        from gof_core import _truncnorm_pdf
        x = np.linspace(-1, 1, 10000)
        pdf = _truncnorm_pdf(x, mu=0.0, sigma=0.3, lo=-1.0, hi=1.0)
        integral = np.trapz(pdf, x)
        np.testing.assert_allclose(integral, 1.0, atol=1e-4)

    def test_zero_outside_bounds(self):
        """PDF should be zero outside [lo, hi]."""
        from gof_core import _truncnorm_pdf
        x_out = np.array([-2.0, -1.01, 1.01, 2.0])
        pdf = _truncnorm_pdf(x_out, mu=0.0, sigma=0.3, lo=-1.0, hi=1.0)
        np.testing.assert_allclose(pdf, 0.0, atol=1e-10)


class TestModelRegistry:
    """Verify MODELS registry structure and completeness."""

    def test_five_models_registered(self, models):
        assert len(models) == 5, f"Expected 5 models, found {len(models)}"

    def test_required_keys(self, models):
        required = {"name", "weight_fn", "bounds", "theta0", "has_free_params"}
        for m in models:
            assert required.issubset(m.keys()), (
                f"Model '{m.get('name', '?')}' missing keys: "
                f"{required - set(m.keys())}"
            )

    def test_weight_fn_callable(self, models):
        for m in models:
            assert callable(m["weight_fn"]), (
                f"Model '{m['name']}' weight_fn is not callable"
            )

    def test_theta0_matches_bounds(self, models):
        """theta0 length must match bounds length for parametric models."""
        for m in models:
            if m["has_free_params"]:
                assert len(m["theta0"]) == len(m["bounds"]), (
                    f"{m['name']}: theta0 has {len(m['theta0'])} params "
                    f"but bounds has {len(m['bounds'])}"
                )
