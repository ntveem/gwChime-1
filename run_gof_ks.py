"""KS-based calibrated GoF test (binning-free)."""
import json, sys, time, logging, numpy as np
from scipy.optimize import minimize

sys.path.insert(0, '/home/teja/Work/gwChime')
import gof_core

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def ks_statistic(q05, q95, inj_df, weight_fn, theta):
    """Sum of KS distances for 5th and 95th percentile CDFs."""
    weights = np.asarray(weight_fn(inj_df, theta), dtype=float)
    weights = np.maximum(weights, 0.0)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    w_sum = weights.sum()
    if w_sum == 0:
        return np.inf
    prob = weights / w_sum

    ks_total = 0.0
    for real_q, col in [(q05, 'chieff_0.05'), (q95, 'chieff_0.95')]:
        inj_q = inj_df[col].values
        order = np.argsort(inj_q)
        inj_sorted = inj_q[order]
        w_sorted = prob[order]
        inj_cdf = np.cumsum(w_sorted)

        idx = np.searchsorted(inj_sorted, real_q)
        model_cdf = np.where(idx > 0, inj_cdf[np.minimum(idx - 1, len(inj_cdf) - 1)], 0.0)
        real_cdf = np.arange(1, len(real_q) + 1) / len(real_q)
        ks_total += np.max(np.abs(model_cdf - real_cdf))

    return ks_total


def fit_ks(q05, q95, inj_df, model):
    """Minimize KS over theta with 3 restarts."""
    weight_fn = model['weight_fn']
    theta0 = model['theta0']
    bounds = model['bounds']

    if not model['has_free_params']:
        return ks_statistic(q05, q95, inj_df, weight_fn, theta0), theta0

    best_ks, best_theta = np.inf, theta0
    starts = [np.array(theta0, dtype=float)]
    rng = np.random.default_rng(12345)
    for _ in range(2):
        starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds]))

    def _ks_obj(t):
        for val, (lo, hi) in zip(t, bounds):
            if val < lo or val > hi:
                return 1e10
        return ks_statistic(q05, q95, inj_df, weight_fn, tuple(t))

    for x0 in starts:
        opt = minimize(_ks_obj, x0, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
        if opt.fun < best_ks:
            best_ks, best_theta = opt.fun, tuple(opt.x)

    return best_ks, best_theta


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['isotropic', 'skewnorm', 'lvk', 'truncgauss', 'roulet'])
    parser.add_argument('--n-mocks', type=int, default=500)
    parser.add_argument('--output', default='gof_ks_results.json')
    args = parser.parse_args()

    CLI_TO_MODEL = {
        'isotropic': 'Isotropic', 'skewnorm': 'Skew-normal',
        'lvk': 'LVK Default', 'truncgauss': 'Truncated Gaussian',
        'roulet': 'Roulet+ 2021',
    }

    inj_df = gof_core.load_injections()
    real_events = gof_core.load_real_events()
    q05_real = np.sort(np.array([e['chieff_0.05'] for e in real_events]))
    q95_real = np.sort(np.array([e['chieff_0.95'] for e in real_events]))
    n_ev = len(real_events)

    results = {}
    for cli_key in args.models:
        model_name = CLI_TO_MODEL[cli_key]
        model = [m for m in gof_core.MODELS if m['name'] == model_name][0]
        log.info('Running KS GoF for %s (n_mocks=%d) ...', model_name, args.n_mocks)

        t0 = time.time()
        ks_real, theta_real = fit_ks(q05_real, q95_real, inj_df, model)

        # Generate mocks from theta_BF
        theta_bf = model['theta0'] if model['theta0'] else ()
        weights_bf = np.asarray(model['weight_fn'](inj_df, theta_bf), dtype=float)
        weights_bf = np.maximum(weights_bf, 0.0)
        weights_bf = np.where(np.isfinite(weights_bf), weights_bf, 0.0)
        prob_bf = weights_bf / weights_bf.sum()

        q05_inj = inj_df['chieff_0.05'].values
        q95_inj = inj_df['chieff_0.95'].values

        ks_mocks = np.empty(args.n_mocks)
        for j in range(args.n_mocks):
            rng = np.random.default_rng(1000 + j)
            drawn = rng.choice(len(inj_df), size=n_ev, replace=True, p=prob_bf)
            mock_q05 = np.sort(q05_inj[drawn])
            mock_q95 = np.sort(q95_inj[drawn])
            ks_mocks[j], _ = fit_ks(mock_q05, mock_q95, inj_df, model)

        elapsed = time.time() - t0
        p = float(np.sum(ks_mocks >= ks_real)) / args.n_mocks
        log.info('  %s: KS_real=%.4f  p=%.3f  (%.1f s)', model_name, ks_real, p, elapsed)

        results[model_name] = {
            'ks_real': ks_real,
            'p_value': p,
            'best_fit_params': list(theta_real),
            'null_distribution': ks_mocks.tolist(),
        }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    log.info('Saved %s', args.output)

    print()
    print('-' * 50)
    for name, r in results.items():
        print(f'{name:25s}  KS={r["ks_real"]:.4f}  p={r["p_value"]:.3f}')
    print('-' * 50)


if __name__ == '__main__':
    main()
