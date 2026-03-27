"""
Core library for χ_eff goodness-of-fit tests on GW population models.

Provides data loading (injections + real events), data-adaptive binning,
and five population-model weight functions as plain dicts.
"""
import logging
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, skewnorm, truncnorm

sys.path.insert(0, '/home/teja/Work/gwChime')
from callister_effective_spin_priors import chi_effective_prior_from_isotropic_spins
from events_summary import EventsSummary

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CHI_MAX = 0.9999  # injection prior support for spin magnitudes

_INJ_PATHS = {
    'o3': '/home/teja/Work/injections/injs_lvc_prior_o3/2-mid_mass',
    'o4': '/home/teja/Work/injections/injs_lvc_prior_o4/2-mid_mass',
}

_EVENTS_PATH = '/home/isha/gwChime/events_summary_o1-o4a_new.hdf5'

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_injections(inj_paths=None):
    """Load O3+O4 injection feather files, merge with PE summaries.

    Returns
    -------
    df : pd.DataFrame
        Columns include: chieff, lnq, q, s1, s2, s1z, s2z,
        cos_tilt1, cos_tilt2, chieff_0.05, chieff_0.95, and more.
    """
    if inj_paths is None:
        inj_paths = _INJ_PATHS

    frames = []
    for run, base in inj_paths.items():
        inj = pd.read_feather(f'{base}/injections.feather')
        summ = pd.read_feather(f'{base}/summary.feather')

        # Summary index is injection name; align by setting same index
        inj = inj.set_index('name')
        df = inj.join(summ, how='inner')
        df = df.reset_index()            # 'name' back as column
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # Derived columns --------------------------------------------------------
    df['q'] = np.exp(df['lnq'])
    assert (df['q'] > 0).all() and (df['q'] <= 1).all(), \
        'q = exp(lnq) must satisfy 0 < q <= 1'

    # cos(tilt) = s_iz / s_i, guarded for near-zero spin magnitude
    for idx in ('1', '2'):
        s_col = f's{idx}'
        sz_col = f's{idx}z'
        ct_col = f'cos_tilt{idx}'
        mask_low = df[s_col] < 1e-4
        df[ct_col] = np.where(mask_low, 0.0, df[sz_col] / df[s_col])

    # Precompute Callister prior (expensive, depends only on q and chieff)
    df['_callister_cache'] = _callister_prior(df['q'].values, df['chieff'].values)

    return df


def load_real_events(events_path=None, mc_lo=5.0, mc_hi=25.0):
    """Load EventsSummary, apply mid-mass cut, compute χ_eff quantiles.

    Parameters
    ----------
    events_path : str, optional
        Path to HDF5 file.
    mc_lo, mc_hi : float
        Source-frame chirp-mass bounds (solar masses).

    Returns
    -------
    events : list[dict]
        Each dict has keys: name, chieff_0.05, chieff_0.95, mc_median.
    """
    if events_path is None:
        events_path = _EVENTS_PATH

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        evs = EventsSummary.from_hdf5(events_path)

    out = []
    for name in evs.events:
        samples = evs.pe_samples[name]
        m1 = samples['m1_source'].values
        m2 = samples['m2_source'].values
        mc = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
        mc_med = np.median(mc)

        if mc_med < mc_lo or mc_med > mc_hi:
            continue

        chieff = samples['chieff'].values
        out.append({
            'name': name,
            'chieff_0.05': np.percentile(chieff, 5),
            'chieff_0.95': np.percentile(chieff, 95),
            'mc_median': mc_med,
        })

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Binning
# ─────────────────────────────────────────────────────────────────────────────

def compute_bin_edges(inj_df, real_events, n_bins=15, margin_frac=0.10):
    """Data-adaptive bin edges from union of all χ_eff quantiles.

    Parameters
    ----------
    inj_df : pd.DataFrame
        Injection dataframe (must have chieff_0.05, chieff_0.95).
    real_events : list[dict]
        Output of ``load_real_events()``.
    n_bins : int
        Approximate number of bins.
    margin_frac : float
        Fractional expansion of the data range.

    Returns
    -------
    edges : np.ndarray, shape (n_bins + 1,)
    """
    vals = np.concatenate([
        inj_df['chieff_0.05'].values,
        inj_df['chieff_0.95'].values,
        [ev['chieff_0.05'] for ev in real_events],
        [ev['chieff_0.95'] for ev in real_events],
    ])
    lo, hi = vals.min(), vals.max()
    span = hi - lo
    lo -= margin_frac * span
    hi += margin_frac * span
    return np.linspace(lo, hi, n_bins + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Callister prior wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _callister_prior(q, chieff):
    """Evaluate isotropic-spin χ_eff prior (Callister+2017).

    Parameters are broadcast to 1-D arrays of the same length.
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    chieff = np.atleast_1d(np.asarray(chieff, dtype=float))
    return chi_effective_prior_from_isotropic_spins(q, CHI_MAX, chieff)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: truncated-normal pdf
# ─────────────────────────────────────────────────────────────────────────────

def _truncnorm_pdf(x, mu, sigma, lo, hi):
    """Truncated-normal PDF on [lo, hi]."""
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    return truncnorm.pdf(x, a, b, loc=mu, scale=sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Model weight functions
# ─────────────────────────────────────────────────────────────────────────────

def _w_isotropic(df, theta):
    """Isotropic spins — injection prior IS the population."""
    return np.ones(len(df))


def _w_skewnorm(df, theta):
    """Skew-normal chi_eff distribution (scipy parameterization)."""
    a, loc, scale = theta
    chieff = df['chieff'].values
    numerator = skewnorm.pdf(chieff, a, loc=loc, scale=scale)
    denominator = df['_callister_cache'].values
    mask = denominator > 0
    out = np.zeros_like(numerator)
    out[mask] = numerator[mask] / denominator[mask]
    return out


def _w_lvk_default(df, theta):
    """LVK Default spin model (GWTC-3 parameterization).

    Population: TruncNorm on spin magnitudes × (aligned + isotropic tilts).
    Injection prior: uniform s∈[0,1), uniform cos(tilt)∈[-1,1]
        → p_prior = 1 / (chi_max^2 * 4)
    """
    mu_chi, sig_chi, mu_t, sig_t, zeta = theta

    s1 = df['s1'].values
    s2 = df['s2'].values
    ct1 = df['cos_tilt1'].values
    ct2 = df['cos_tilt2'].values

    # Spin magnitude PDFs (truncated to [0, 1])
    p_s1 = _truncnorm_pdf(s1, mu_chi, sig_chi, 0.0, 1.0)
    p_s2 = _truncnorm_pdf(s2, mu_chi, sig_chi, 0.0, 1.0)

    # Tilt PDFs (truncated to [-1, 1])
    p_ct1 = _truncnorm_pdf(ct1, mu_t, sig_t, -1.0, 1.0)
    p_ct2 = _truncnorm_pdf(ct2, mu_t, sig_t, -1.0, 1.0)

    # Mixture: aligned component + isotropic component
    tilt_factor = zeta * p_ct1 * p_ct2 + (1.0 - zeta) / 4.0

    # Population PDF
    p_pop = p_s1 * p_s2 * tilt_factor

    # Injection prior: uniform on [0, chi_max]^2 × [-1,1]^2
    p_prior = 1.0 / (CHI_MAX ** 2) * 0.25

    return p_pop / p_prior


def _w_truncgauss(df, theta):
    """Truncated Gaussian on χ_eff / Callister prior."""
    mu, sig = theta
    chieff = df['chieff'].values
    q = df['q'].values
    a = (-1.0 - mu) / sig
    b = (1.0 - mu) / sig
    numerator = truncnorm.pdf(chieff, a, b, loc=mu, scale=sig)
    denominator = df['_callister_cache'].values
    mask = denominator > 0
    out = np.zeros_like(numerator)
    out[mask] = numerator[mask] / denominator[mask]
    return out


def _w_twogauss(df, theta):
    """Two-Gaussian mixture on χ_eff / Callister prior."""
    f, sig0, mu_su, sig_su = theta
def _w_roulet(df, theta):
    """Roulet+ 2021 three-component χ_eff model (Eq. 6 of 2105.10580).

    Three half-Gaussians centered at zero:
      ζ₀ N(χ_eff; σ₀=0.04, [-1,1])   — nonspinning (dynamical)
      ζ_neg N(χ_eff; σ, [-1,0])       — anti-aligned
      ζ_pos N(χ_eff; σ, [0,1])        — aligned (field)
    """
    zeta_pos, zeta_neg, sigma = theta
    zeta_0 = 1.0 - zeta_pos - zeta_neg
    chieff = df['chieff'].values
    q = df['q'].values

    sigma_0 = 0.04  # fixed narrow peak width

    # N(x; σ₀) truncated to [-1, 1]
    p_zero = _truncnorm_pdf(chieff, 0.0, sigma_0, -1.0, 1.0)

    # N<0(x; σ) truncated to [-1, 0]
    p_neg = _truncnorm_pdf(chieff, 0.0, sigma, -1.0, 0.0)

    # N>0(x; σ) truncated to [0, 1]
    p_pos = _truncnorm_pdf(chieff, 0.0, sigma, 0.0, 1.0)

    numerator = zeta_0 * p_zero + zeta_neg * p_neg + zeta_pos * p_pos
    denominator = df['_callister_cache'].values
    mask = denominator > 0
    out = np.zeros_like(numerator)
    out[mask] = numerator[mask] / denominator[mask]
    return out

# Model registry
# ─────────────────────────────────────────────────────────────────────────────

MODELS = [
    {
        'name': 'Isotropic',
        'weight_fn': _w_isotropic,
        'bounds': [],
        'theta0': (),
        'has_free_params': False,
    },
    {
        'name': 'Skew-normal',
        'weight_fn': _w_skewnorm,
        'bounds': [(0, 30), (-0.5, 0.5), (0.01, 0.5)],
        'theta0': (13.438, -0.069, 0.161),  # max-lnL from LVK hierarchical fit
        'has_free_params': True,
    },
    {
        'name': 'LVK Default',
        'weight_fn': _w_lvk_default,
        'bounds': [(0.0, 1.0), (0.01, 1.0), (-1.0, 1.0), (0.01, 2.0), (0.0, 1.0)],
        'theta0': (0.0913, 0.3176, 0.7522, 0.8835, 0.3821),  # MAP from GWTC-4.0
        'has_free_params': True,
    },
    {
        'name': 'Truncated Gaussian',
        'weight_fn': _w_truncgauss,
        'bounds': [(-0.5, 0.5), (0.01, 0.5)],
        'theta0': (0.0424, 0.1034),  # median from GWTC-4.0
        'has_free_params': True,
    },
    {
        'name': 'Roulet+ 2021',
        'weight_fn': _w_roulet,
        'bounds': [(0.0, 1.0), (0.0, 0.5), (0.01, 1.0)],
        'theta0': (0.45, 0.00, 0.23),  # MLE from 2105.10580 Fig.3
        'has_free_params': True,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Histogram and χ² computation
# ─────────────────────────────────────────────────────────────────────────────

def make_histogram(quantiles_05, quantiles_95, bin_edges):
    """Concatenated histogram vector from 5th and 95th χ_eff quantiles.

    Parameters
    ----------
    quantiles_05 : array-like
        5th-percentile χ_eff values.
    quantiles_95 : array-like
        95th-percentile χ_eff values.
    bin_edges : np.ndarray
        Bin edges (length n_bins + 1).

    Returns
    -------
    vec : np.ndarray, shape (2 * n_bins,)
        Concatenation of two histograms, each normalized to sum = 1.
    """
    h05 = np.histogram(quantiles_05, bins=bin_edges)[0].astype(float)
    h95 = np.histogram(quantiles_95, bins=bin_edges)[0].astype(float)
    s05 = h05.sum()
    s95 = h95.sum()
    if s05 > 0:
        h05 /= s05
    if s95 > 0:
        h95 /= s95
    return np.concatenate([h05, h95])


def compute_chi2(data_vec, inj_df, weight_fn, theta, bin_edges,
                 n_events=71, n_bootstrap=1000, seed=42):
    """Bootstrap χ² statistic comparing data to weighted injection model.

    Parameters
    ----------
    data_vec : np.ndarray, shape (2 * n_bins,)
        Observed histogram vector (from ``make_histogram``).
    inj_df : pd.DataFrame
        Injection dataframe with chieff_0.05, chieff_0.95 columns.
    weight_fn : callable
        ``weight_fn(inj_df, theta) -> array of weights``.
    theta : tuple
        Model parameters.
    bin_edges : np.ndarray
        Bin edges.
    n_events : int
        Number of events to draw per bootstrap (= real catalogue size).
    n_bootstrap : int
        Number of bootstrap resamples (fixed seed → deterministic).
    seed : int
        RNG seed for bootstrap draws.

    Returns
    -------
    result : dict
        Keys: chi2 (float), mu (ndarray), cov (ndarray),
        cond_number (float), n_dof (int).
    """
    weights = np.asarray(weight_fn(inj_df, theta), dtype=float)
    weights = np.maximum(weights, 0.0)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    w_sum = weights.sum()
    if w_sum == 0:
        return {
            'chi2': np.inf, 'mu': np.zeros(2 * (len(bin_edges) - 1)),
            'cov': np.eye(2 * (len(bin_edges) - 1)),
            'cond_number': np.inf, 'n_dof': 0,
        }
    prob = weights / w_sum

    q05 = inj_df['chieff_0.05'].values
    q95 = inj_df['chieff_0.95'].values
    n_inj = len(inj_df)

    # Pre-compute bootstrap draw indices (fixed seed → deterministic)
    rng = np.random.default_rng(seed)
    boot_indices = rng.choice(n_inj, size=(n_bootstrap, n_events),
                              replace=True, p=prob)

    # Build bootstrap histogram vectors
    n_bins = len(bin_edges) - 1
    boot_vecs = np.empty((n_bootstrap, 2 * n_bins))
    for b in range(n_bootstrap):
        idx = boot_indices[b]
        boot_vecs[b] = make_histogram(q05[idx], q95[idx], bin_edges)

    mu = boot_vecs.mean(axis=0)
    cov = np.cov(boot_vecs, rowvar=False, ddof=1)

    # Drop bins with zero variance (empty bins that never get filled)
    diag = np.diag(cov)
    good = diag > 1e-30
    n_dof = int(good.sum())

    if n_dof == 0:
        return {
            'chi2': 0.0, 'mu': mu, 'cov': cov,
            'cond_number': np.inf, 'n_dof': 0,
        }

    # Restrict to non-degenerate subspace
    diff_full = data_vec - mu
    diff = diff_full[good]
    cov_sub = cov[np.ix_(good, good)]
    cond = np.linalg.cond(cov_sub)

    # Use pseudoinverse for robustness against residual linear dependence
    cov_pinv = np.linalg.pinv(cov_sub)
    chi2_val = float(diff @ cov_pinv @ diff)

    return {
        'chi2': chi2_val,
        'mu': mu,
        'cov': cov,
        'cond_number': cond,
        'n_dof': n_dof,
    }
# ─────────────────────────────────────────────────────────────────────────────
# Mock catalogue generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_mock_catalog(inj_df, weight_fn, theta, n_events=71, rng=None):
    """Draw a mock event catalogue by resampling injections.

    Parameters
    ----------
    inj_df : pd.DataFrame
        Injection dataframe with chieff_0.05, chieff_0.95.
    weight_fn : callable
        ``weight_fn(inj_df, theta) -> weights``.
    theta : tuple
        Model parameters.
    n_events : int
        Number of events to draw.
    rng : np.random.Generator or None
        Random generator; if None, uses default_rng().

    Returns
    -------
    mock : dict
        Keys: chieff_0.05, chieff_0.95 — arrays of length n_events.
    """
    if rng is None:
        rng = np.random.default_rng()

    weights = np.asarray(weight_fn(inj_df, theta), dtype=float)
    weights = np.maximum(weights, 0.0)
    w_sum = weights.sum()
    if w_sum == 0:
        raise ValueError("All weights are zero — cannot draw mock catalogue.")
    prob = weights / w_sum

    idx = rng.choice(len(inj_df), size=n_events, replace=True, p=prob)
    return {
        'chieff_0.05': inj_df['chieff_0.05'].values[idx],
        'chieff_0.95': inj_df['chieff_0.95'].values[idx],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Calibrated goodness-of-fit test
# ─────────────────────────────────────────────────────────────────────────────

def calibrated_gof(real_vec, inj_df, model, bin_edges,
                   n_events=71, n_mocks=500, n_bootstrap=1000,
                   bootstrap_seed=42):
    """Calibrated goodness-of-fit test.

    For models with free parameters, minimises χ²(θ) for the real data
    and each mock catalogue, then builds a null distribution of
    χ²_min values to compute a calibrated p-value.

    For fixed-parameter models (isotropic), computes the
    raw χ² for the real data and each mock — no optimisation.

    Parameters
    ----------
    real_vec : np.ndarray, shape (2 * n_bins,)
        Observed histogram vector.
    inj_df : pd.DataFrame
        Injection dataframe.
    model : dict
        Entry from ``MODELS`` list.
    bin_edges : np.ndarray
        Bin edges.
    n_mocks : int
        Number of mock catalogues.
    n_bootstrap : int
        Bootstrap resamples per χ² evaluation.
    bootstrap_seed : int
        Fixed seed for bootstrap (determinism).

    Returns
    -------
    result : dict
        Keys: chi2_real (float), chi2_mocks (ndarray of length n_mocks),
        p_value (float), theta_real (tuple), theta_mocks (list of tuples).
        For fixed-param models, theta_real == theta_mocks[i] == model theta0.
    """
    weight_fn = model['weight_fn']
    has_free = model['has_free_params']
    theta0 = model['theta0']
    bounds = model['bounds']

    def _chi2_objective(theta_arr):
        """Objective for scipy.optimize.minimize."""
        # Enforce bounds (Nelder-Mead ignores them)
        for val, (lo, hi) in zip(theta_arr, bounds):
            if val < lo or val > hi:
                return 1e10
        theta_tup = tuple(theta_arr)
        res = compute_chi2(data_vec_local, inj_df, weight_fn, theta_tup,
                           bin_edges, n_events=n_events,
                           n_bootstrap=n_bootstrap,
                           seed=bootstrap_seed)
        return res['chi2']

    def _fit_chi2(data_vec_local_arg):
        """Minimize χ² over theta with 3 random restarts."""
        nonlocal data_vec_local
        data_vec_local = data_vec_local_arg

        if not has_free:
            res = compute_chi2(data_vec_local, inj_df, weight_fn, theta0,
                               bin_edges, n_events=n_events,
                               n_bootstrap=n_bootstrap,
                               seed=bootstrap_seed)
            return res['chi2'], theta0

        best_chi2 = np.inf
        best_theta = theta0

        # 3 restarts: first from theta0, then 2 random points in bounds
        starts = [np.array(theta0, dtype=float)]
        restart_rng = np.random.default_rng(12345)
        for _ in range(2):
            pt = np.array([
                restart_rng.uniform(lo, hi) for lo, hi in bounds
            ])
            starts.append(pt)

        for x0 in starts:
            opt = minimize(_chi2_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if opt.fun < best_chi2:
                best_chi2 = opt.fun
                best_theta = tuple(opt.x)

        return best_chi2, best_theta

    # Closure variable for objective
    data_vec_local = None

    # ── Real data ──
    chi2_real, theta_real = _fit_chi2(real_vec)

    # ── Mock catalogues ──
    # Use best-fit theta (or theta0 for fixed models) as the truth
    theta_truth = theta_real
    chi2_mocks = np.empty(n_mocks)
    theta_mocks = []

    for i in range(n_mocks):
        mock_rng = np.random.default_rng(1000 + i)
        mock = generate_mock_catalog(inj_df, weight_fn, theta_truth,
                                     n_events=n_events, rng=mock_rng)
        mock_vec = make_histogram(mock['chieff_0.05'], mock['chieff_0.95'],
                                  bin_edges)
        chi2_i, theta_i = _fit_chi2(mock_vec)
        chi2_mocks[i] = chi2_i
        theta_mocks.append(theta_i)

    p_value = float(np.sum(chi2_mocks >= chi2_real)) / n_mocks

    return {
        'chi2_real': chi2_real,
        'chi2_mocks': chi2_mocks,
        'p_value': p_value,
        'theta_real': theta_real,
        'theta_mocks': theta_mocks,
    }
