#!/home/teja/anaconda3/envs/cogwheel/bin/python
"""CLI runner for the χ_eff goodness-of-fit pipeline.

Usage
-----
Phase A (quick smoke-test):
    python run_gof.py --models isotropic truncgauss --n-mocks 50

Phase B (full run):
    python run_gof.py --n-mocks 500
"""

import argparse
import json
import logging
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import gof_core

# ─────────────────────────────────────────────────────────────────────────────
# CLI ↔ model name mapping
# ─────────────────────────────────────────────────────────────────────────────

CLI_TO_MODEL = {
    'isotropic': 'Isotropic',
    'skewnorm': 'Skew-normal',
    'lvk': 'LVK Default',
    'truncgauss': 'Truncated Gaussian',
    'roulet': 'Roulet+ 2021',
}

ALL_CLI_NAMES = list(CLI_TO_MODEL.keys())

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _model_by_name(name):
    """Look up model dict from MODELS list by display name."""
    for m in gof_core.MODELS:
        if m['name'] == name:
            return m
    raise KeyError(f"Unknown model name: {name}")


def plot_histogram_overlay(real_vec, inj_df, model, theta, bin_edges,
                           out_path, n_events=71):
    """Per-model overlay: real data histogram vs synthetic at best-fit θ."""
    import matplotlib
    matplotlib.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.labelsize': 12, 'legend.fontsize': 9,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
    })

    n_bins = len(bin_edges) - 1

    # Synthetic expectation + covariance from bootstrap
    res = gof_core.compute_chi2(real_vec, inj_df, model['weight_fn'], theta,
                                bin_edges, n_events=n_events,
                                n_bootstrap=1000, seed=42)
    mu = res['mu']
    sigma = np.sqrt(np.diag(res['cov']))
    mu_05, mu_95 = mu[:n_bins], mu[n_bins:]
    sig_05, sig_95 = sigma[:n_bins], sigma[n_bins:]
    real_05, real_95 = real_vec[:n_bins], real_vec[n_bins:]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, real_h, model_h, model_sig, label in [
        (axes[0], real_05, mu_05, sig_05, r'5th percentile of $\chi_\mathrm{eff}$'),
        (axes[1], real_95, mu_95, sig_95, r'95th percentile of $\chi_\mathrm{eff}$'),
    ]:
        # Data as step histogram
        ax.step(bin_edges, np.append(real_h, real_h[-1]), where='post',
                color='black', linewidth=1.5, label='GWTC events')

        # Model as step + shaded 1σ band
        ax.step(bin_edges, np.append(model_h, model_h[-1]), where='post',
                color='C0', linewidth=1.2, label='Model prediction')
        # 1σ band
        lo = np.maximum(model_h - model_sig, 0)
        hi = model_h + model_sig
        for i in range(n_bins):
            ax.fill_between([bin_edges[i], bin_edges[i+1]], lo[i], hi[i],
                            color='C0', alpha=0.2)

        ax.set_xlabel(label)
        ax.set_ylabel('Fraction per bin')
        ax.legend(frameon=False)

    fig.suptitle(model['name'], fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    log.info("Saved %s", out_path)


def plot_null_distribution(chi2_real, chi2_mocks, model_name, out_path):
    """Histogram of null χ² distribution with observed value marked."""
    import matplotlib
    matplotlib.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.labelsize': 12, 'legend.fontsize': 9,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(chi2_mocks, bins=30, density=True, histtype='stepfilled',
            alpha=0.4, color='grey', edgecolor='black', linewidth=0.8,
            label='Null distribution')
    ax.axvline(chi2_real, color='C3', linewidth=2, linestyle='--',
               label=r'Observed $\chi^2_\mathrm{min}$' + f' = {chi2_real:.1f}')
    p_val = float(np.sum(chi2_mocks >= chi2_real)) / len(chi2_mocks)
    ax.set_title(f'{model_name}  ($p$ = {p_val:.3f})', fontsize=12)
    ax.set_xlabel(r'$\chi^2_\mathrm{min}$')
    ax.set_ylabel('Density')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    log.info("Saved %s", out_path)


def plot_pvalues_summary(results, out_path):
    """Bar chart of p-values across all tested models."""
    names = list(results.keys())
    pvals = [results[n]['p_value'] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(names))
    bars = ax.bar(x, pvals, color='steelblue', edgecolor='k', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('p-value')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.05, color='red', linewidth=1, linestyle=':', label='α = 0.05')
    ax.legend(fontsize=9)
    ax.set_title('Calibrated GoF p-values')

    # Annotate bars
    for bar, pv in zip(bars, pvals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{pv:.3f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    log.info("Saved %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(results):
    """Print a readable summary table to stdout."""
    header = f"{'Model':<24s} {'χ²_real':>10s} {'p-value':>10s} {'N_eff':>10s}"
    sep = '-' * len(header)
    print('\n' + sep)
    print(header)
    print(sep)
    for name, r in results.items():
        chi2 = r['chi2_min_real']
        pval = r['p_value']
        neff = r.get('n_eff', 'N/A')
        neff_str = f'{neff:.0f}' if isinstance(neff, (int, float)) else str(neff)
        print(f'{name:<24s} {chi2:10.2f} {pval:10.3f} {neff_str:>10s}')
    print(sep + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Calibrated χ_eff goodness-of-fit test pipeline.')
    parser.add_argument(
        '--models', nargs='+', choices=ALL_CLI_NAMES, default=ALL_CLI_NAMES,
        help='Models to test (default: all).')
    parser.add_argument(
        '--n-mocks', type=int, default=500,
        help='Number of mock catalogues per model (default: 500).')
    parser.add_argument(
        '--output-dir', type=str, default='.',
        help='Directory for output files (default: current directory).')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%H:%M:%S',
    )

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load data ──
    log.info("Loading injections …")
    inj_df = gof_core.load_injections()
    log.info("Loaded %d injections.", len(inj_df))

    log.info("Loading real events …")
    real_events = gof_core.load_real_events()
    log.info("Loaded %d real events.", len(real_events))

    # ── 2. Bin edges ──
    bin_edges = gof_core.compute_bin_edges(inj_df, real_events)
    log.info("Bin edges (%d bins): %s", len(bin_edges) - 1,
             np.array2string(bin_edges, precision=3))

    # ── 3. Real data histogram ──
    q05 = np.array([e['chieff_0.05'] for e in real_events])
    q95 = np.array([e['chieff_0.95'] for e in real_events])
    real_vec = gof_core.make_histogram(q05, q95, bin_edges)

    # ── 4. Calibrated GoF for each selected model ──
    selected_models = [CLI_TO_MODEL[c] for c in args.models]
    results = {}

    for model_name in selected_models:
        model = _model_by_name(model_name)
        log.info("Running calibrated GoF for %s (n_mocks=%d) …",
                 model_name, args.n_mocks)
        t0 = time.time()
        gof_res = gof_core.calibrated_gof(
            real_vec, inj_df, model, bin_edges,
            n_events=len(real_events), n_mocks=args.n_mocks)
        elapsed = time.time() - t0
        log.info("  %s: χ²_real=%.2f  p=%.3f  (%.1f s)",
                 model_name, gof_res['chi2_real'],
                 gof_res['p_value'], elapsed)

        # Compute N_eff for this model (sum(w)^2 / sum(w^2))
        w = model['weight_fn'](inj_df, gof_res['theta_real'])
        w = np.asarray(w, dtype=float)
        w_sum = w.sum()
        n_eff = (w_sum ** 2) / (w ** 2).sum() if w_sum > 0 else 0.0

        # Condition number from a compute_chi2 call at best-fit
        chi2_info = gof_core.compute_chi2(
            real_vec, inj_df, model['weight_fn'], gof_res['theta_real'],
            bin_edges, n_events=len(real_events),
            n_bootstrap=1000, seed=42)

        results[model_name] = {
            'p_value': gof_res['p_value'],
            'chi2_min_real': gof_res['chi2_real'],
            'best_fit_params': list(gof_res['theta_real']),
            'null_distribution': gof_res['chi2_mocks'].tolist(),
            'condition_number': chi2_info['cond_number'],
            'n_eff': n_eff,
        }

    # ── 5. Save JSON ──
    json_path = os.path.join(out_dir, 'gof_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info("Saved results to %s", json_path)

    # ── 6. Figures ──
    for model_name, r in results.items():
        model = _model_by_name(model_name)
        cli_key = [k for k, v in CLI_TO_MODEL.items() if v == model_name][0]

        theta_best = tuple(r['best_fit_params'])
        chi2_mocks = np.array(r['null_distribution'])

        plot_histogram_overlay(
            real_vec, inj_df, model, theta_best, bin_edges,
            os.path.join(out_dir, f'fig_hist_{cli_key}.pdf'),
            n_events=len(real_events))
        plot_null_distribution(
            r['chi2_min_real'], chi2_mocks, model_name,
            os.path.join(out_dir, f'fig_null_{cli_key}.pdf'))

    plot_pvalues_summary(results,
                         os.path.join(out_dir, 'fig_pvalues_summary.pdf'))

    # ── 7. Summary table ──
    print_summary_table(results)


if __name__ == '__main__':
    main()
