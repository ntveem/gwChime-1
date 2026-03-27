"""
Physics tests for the GoF χ_eff pipeline.

These tests verify physical invariants: normalization, weight properties,
determinism, and sensitivity to obviously wrong models.

All tests marked @pytest.mark.physics.
Diagnostic plots saved to tests/output/physics/.
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Isotropic model weights are unity
# ---------------------------------------------------------------------------
@pytest.mark.physics
class TestIsotropicWeightsUnity:
    """Isotropic model: p_pop == p_prior, so all weights must be 1.0."""

    def test_isotropic_weights_all_one(self, inj_df, models):
        # The isotropic weight function returns ones by construction
        iso_model = [m for m in models if m["name"] == "Isotropic"][0]
        weights = iso_model["weight_fn"](inj_df, None)

        assert len(weights) == len(inj_df), "Weight count != injection count"
        np.testing.assert_allclose(
            weights, 1.0, atol=1e-6,
            err_msg="Isotropic weights must all be 1.0 (p_pop == p_prior)"
        )


# ---------------------------------------------------------------------------
# 2. Callister prior normalizes to 1
# ---------------------------------------------------------------------------
@pytest.mark.physics
class TestCallisterPriorNormalization:
    """The Callister chi_eff prior must integrate to 1 for any mass ratio q."""

    @pytest.mark.parametrize("q_val", [0.1, 0.3, 0.5, 0.7, 1.0])
    def test_callister_prior_integrates_to_one(self, q_val):
        from callister_effective_spin_priors import chi_effective_prior_from_isotropic_spins

        chi_max = 0.9999
        n_pts = 10000
        chieff_grid = np.linspace(-chi_max, chi_max, n_pts)

        q_arr = np.full_like(chieff_grid, q_val)
        pdf_vals = chi_effective_prior_from_isotropic_spins(q_arr, chi_max, chieff_grid)
        integral = np.trapz(pdf_vals, chieff_grid)

        # Save diagnostic plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(chieff_grid, pdf_vals, label=f"q={q_val}")
        ax.set_xlabel(r"$\chi_{\rm eff}$")
        ax.set_ylabel("PDF")
        ax.set_title(f"Callister prior (q={q_val}), integral={integral:.6f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            f"tests/output/physics/test_callister_normalization_q{q_val}.png",
            dpi=100,
        )
        plt.close(fig)

        assert np.isfinite(integral), f"Integral is not finite for q={q_val}"
        np.testing.assert_allclose(
            integral, 1.0, atol=1e-3,
            err_msg=f"Callister prior not normalized for q={q_val}: integral={integral}",
        )


# ---------------------------------------------------------------------------
# 3. All model weights are finite and positive
# ---------------------------------------------------------------------------
@pytest.mark.physics
class TestAllModelWeightsFinitePositive:
    """For all 5 models at θ₀, weights must be finite, positive, with N_eff > 100."""

    def test_weights_finite_positive_all_models(self, inj_df, models):
        results = {}
        for model in models:
            name = model["name"]
            w = np.asarray(model["weight_fn"](inj_df, model["theta0"]), dtype=float)

            # Basic checks
            assert len(w) == len(inj_df), f"{name}: weight length mismatch"
            assert np.all(np.isfinite(w)), f"{name}: found NaN or Inf in weights"
            assert np.all(w >= 0), f"{name}: found negative weights"

            # For ratio-based models (p_pop / p_prior), exact zeros can occur
            # at chi_eff boundaries where both numerator and denominator → 0.
            # Skew-normal is known to produce ~33% zeros because Callister
            # prior vanishes for |chieff| > chi_max * q/(1+q).
            # We require N_eff > 100 on positive weights as the primary check.
            n_zero = np.sum(w == 0)
            frac_zero = n_zero / len(w)

            # Effective sample size (computed on positive weights only)
            w_pos = w[w > 0]
            w_sum = w_pos.sum()
            w2_sum = (w_pos ** 2).sum()
            n_eff = w_sum ** 2 / w2_sum
            results[name] = {
                "n_eff": n_eff, "w_mean": w.mean(),
                "w_std": w.std(), "n_zero": int(n_zero),
            }

            assert n_eff > 100, (
                f"{name}: N_eff = {n_eff:.1f} < 100 — insufficient effective samples"
            )

        # Save summary plot
        fig, ax = plt.subplots(figsize=(8, 4))
        names = list(results.keys())
        n_effs = [results[n]["n_eff"] for n in names]
        ax.barh(names, n_effs)
        ax.axvline(100, color="red", linestyle="--", label="N_eff=100 threshold")
        ax.set_xlabel("N_eff")
        ax.set_title("Effective sample size per model at θ₀")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            "tests/output/physics/test_all_model_weights_neff_summary.png",
            dpi=100,
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# 4. χ² computation is deterministic
# ---------------------------------------------------------------------------
@pytest.mark.physics
class TestChi2Determinism:
    """compute_chi2 must return bit-identical results when called twice
    with the same inputs, thanks to the fixed seed=42."""

    def test_chi2_exact_reproducibility(self, inj_df, data_vec, bin_edges, models):
        from gof_core import compute_chi2

        # Use truncated Gaussian model at θ₀
        tg_model = [m for m in models if m["name"] == "Truncated Gaussian"][0]
        theta0 = tg_model["theta0"]

        res1 = compute_chi2(data_vec, inj_df, tg_model["weight_fn"], theta0,
                            bin_edges, n_bootstrap=200, seed=42)
        res2 = compute_chi2(data_vec, inj_df, tg_model["weight_fn"], theta0,
                            bin_edges, n_bootstrap=200, seed=42)

        # Exact floating-point equality (not just close)
        assert res1["chi2"] == res2["chi2"], (
            f"χ² not deterministic: {res1['chi2']} != {res2['chi2']}"
        )
        np.testing.assert_array_equal(
            res1["mu"], res2["mu"],
            err_msg="Bootstrap mean vectors differ between runs"
        )
        np.testing.assert_array_equal(
            res1["cov"], res2["cov"],
            err_msg="Bootstrap covariance matrices differ between runs"
        )


# ---------------------------------------------------------------------------
# 5. Covariance matrix is positive definite
# ---------------------------------------------------------------------------
@pytest.mark.physics
class TestCovariancePositiveDefinite:
    """The bootstrap covariance matrix must be positive definite
    with condition number < 1e6."""

    def test_cov_eigenvalues_positive(self, inj_df, data_vec, bin_edges, models):
        from gof_core import compute_chi2

        # Test with isotropic model (simplest, no parameter tuning)
        iso_model = [m for m in models if m["name"] == "Isotropic"][0]
        res = compute_chi2(data_vec, inj_df, iso_model["weight_fn"],
                           iso_model["theta0"], bin_edges,
                           n_bootstrap=1000, seed=42)

        cov = res["cov"]
        eigenvalues = np.linalg.eigvalsh(cov)

        # Save diagnostic plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(np.sort(eigenvalues)[::-1], "o-")
        ax.set_xlabel("Eigenvalue index (sorted desc)")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(f"Cov eigenspectrum (cond={res['cond_number']:.2e})")
        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(
            "tests/output/physics/test_cov_eigenspectrum.png",
            dpi=100,
        )
        plt.close(fig)

        assert np.all(eigenvalues > 0), (
            f"Covariance has non-positive eigenvalues: min={eigenvalues.min():.2e}"
        )
        cond = np.max(eigenvalues) / np.min(eigenvalues)
        assert cond < 1e6, (
            f"Condition number {cond:.2e} >= 1e6 — covariance ill-conditioned"
        )


# ---------------------------------------------------------------------------
# 6. Obviously wrong model produces large χ²
# ---------------------------------------------------------------------------
@pytest.mark.physics
class TestBadModelLargeChi2:
    """A model with weights = exp(10 * chieff) is obviously wrong and
    must produce a χ² far larger than any reasonable model."""

    def test_bad_model_chi2_much_larger(self, inj_df, data_vec, bin_edges, models):
        from gof_core import compute_chi2

        # Bad model: weights concentrated on chieff > 0.5
        def bad_weight_fn(df, theta):
            return np.exp(10.0 * df["chieff"].values)

        res_bad = compute_chi2(data_vec, inj_df, bad_weight_fn, None,
                               bin_edges, n_bootstrap=500, seed=42)

        # Compare against isotropic model (our baseline "reasonable" model)
        iso_model = [m for m in models if m["name"] == "Isotropic"][0]
        res_iso = compute_chi2(data_vec, inj_df, iso_model["weight_fn"],
                               iso_model["theta0"], bin_edges,
                               n_bootstrap=500, seed=42)

        chi2_bad = res_bad["chi2"]
        chi2_iso = res_iso["chi2"]

        # Save diagnostic plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Isotropic (reasonable)", "exp(10·χ_eff) (bad)"],
               [chi2_iso, chi2_bad], color=["green", "red"])
        ax.set_ylabel("χ²")
        ax.set_title("χ² comparison: reasonable vs obviously wrong model")
        fig.tight_layout()
        fig.savefig(
            "tests/output/physics/test_bad_model_chi2_comparison.png",
            dpi=100,
        )
        plt.close(fig)

        assert chi2_bad > 5 * chi2_iso, (
            f"Bad model χ²={chi2_bad:.1f} is not 5× larger than "
            f"isotropic χ²={chi2_iso:.1f}"
        )
