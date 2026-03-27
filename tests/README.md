# GoF χ_eff Pipeline — Test Suite

## Test Files

| File | Covers | Type | Notes |
|------|--------|------|-------|
| `conftest.py` | Shared fixtures: `inj_df`, `real_events`, `bin_edges`, `data_vec`, `models` | Fixtures | Session-scoped to avoid repeated I/O |
| `test_core_unit.py` | `make_histogram`, `compute_bin_edges`, `_callister_prior`, `_truncnorm_pdf`, `MODELS` registry | Unit | Pure-function tests, no data dependency for most |
| `test_physics_gof.py` | 6 professor-specified physics invariants | Physics (`@pytest.mark.physics`) | Diagnostic plots in `output/physics/` |

## Physics Tests (test_physics_gof.py)

1. **TestIsotropicWeightsUnity** — Isotropic weight_fn returns all 1.0
2. **TestCallisterPriorNormalization** — Callister prior integrates to 1 for q ∈ {0.1, 0.3, 0.5, 0.7, 1.0}
3. **TestAllModelWeightsFinitePositive** — All 5 models: no NaN/Inf/negative, N_eff > 100
4. **TestChi2Determinism** — `compute_chi2` produces bit-identical results with fixed seed
5. **TestCovariancePositiveDefinite** — Bootstrap covariance eigenvalues > 0, κ < 1e6
6. **TestBadModelLargeChi2** — exp(10·χ_eff) model yields χ² > 5× isotropic baseline

## Running

```bash
/home/teja/anaconda3/envs/cogwheel/bin/python -m pytest tests/ -v
# Physics tests only:
/home/teja/anaconda3/envs/cogwheel/bin/python -m pytest tests/ -m physics -v
```

## Diagnostic Plots

Saved to `tests/output/physics/`:
- `test_callister_normalization_q{q}.png` — PDF curves per q
- `test_all_model_weights_neff_summary.png` — N_eff bar chart
- `test_cov_eigenspectrum.png` — Covariance eigenvalue spectrum
- `test_bad_model_chi2_comparison.png` — χ² bar chart: reasonable vs bad model
