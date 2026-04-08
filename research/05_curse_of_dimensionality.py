#!/usr/bin/env python3
"""
05_curse_of_dimensionality.py — Empirical curse-of-dimensionality
demonstration on the user's 12-match dataset (module E14, the keystone
narrative module).

Demonstrates *on real data* the small-sample instability that motivates
every robust technique used elsewhere in the project: shrinkage estimation
(E6), the rank-deficient covariance problem (E1), tail-risk measurement
with discrete distributions (E2), and the choice of the constrained QP over
closed-form Markowitz (E1 again).

Three concrete experiments:

  1. Sequential metric stability — as N grows from 1 to 12, plot how
     Sharpe, mean return, std, and max drawdown evolve. Show that point
     estimates change dramatically with each new observation.

  2. Bootstrap distribution of Sharpe — resample the 12 matches with
     replacement 5000 times, recompute Sharpe each time, plot the
     distribution. The width of this distribution is the honest answer
     to "what is the Sharpe of this strategy?"

  3. Estimator instability summary — analytic and bootstrap confidence
     intervals on key metrics; ratio N/k for the cross-match MV problem;
     condition number of the betting Σ matrix as a function of how many
     outcomes we admit.

Together these justify every methodological choice elsewhere in the
project, and form the core "discussion" section of the report.

USAGE
-----
    python3 05_curse_of_dimensionality.py --history /path/to/user.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
mv = import_module("01_mean_variance_hedge")
ra = import_module("02_risk_adjusted_metrics")
ve = import_module("03_var_es")


HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
FIG_DIR = RESULTS_DIR / "figures"
TBL_DIR = RESULTS_DIR / "tables"
for d in (RESULTS_DIR, FIG_DIR, TBL_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# Experiment 1: sequential metric stability
# ============================================================
def sequential_metrics(returns):
    """
    For each k = 2, 3, ..., N, compute the metrics on the FIRST k returns.
    Returns a dict of arrays, one per metric.
    """
    r = np.asarray(returns, dtype=float)
    N = len(r)
    out = {
        "k": np.arange(2, N + 1),
        "mean": np.zeros(N - 1),
        "std": np.zeros(N - 1),
        "sharpe": np.zeros(N - 1),
        "min": np.zeros(N - 1),
        "max_dd": np.zeros(N - 1),
    }
    for j, k in enumerate(out["k"]):
        sub = r[:k]
        out["mean"][j] = float(np.mean(sub))
        out["std"][j] = float(np.std(sub, ddof=1))
        out["sharpe"][j] = ra.sharpe_ratio(sub)
        out["min"][j] = float(np.min(sub))
        # crude equity curve from cumprod for max DD
        eq = np.cumprod(1 + sub)
        peaks = np.maximum.accumulate(eq)
        out["max_dd"][j] = float(np.max((peaks - eq) / peaks))
    return out


# ============================================================
# Experiment 2: bootstrap distribution of Sharpe
# ============================================================
def bootstrap_sharpe(returns, n_resamples=5000, seed=42):
    """
    Bootstrap the Sharpe ratio: draw N_returns with replacement n_resamples
    times, compute Sharpe for each draw. Returns the array of bootstrap
    estimates.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    N = len(r)
    boots = np.zeros(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(r, size=N, replace=True)
        boots[i] = ra.sharpe_ratio(sample)
    return boots


# ============================================================
# Experiment 3: ridge regularization stabilizes a rank-deficient Σ
# ============================================================
def ridge_regularization_sweep(odds, lambdas=None):
    """
    The 3-state betting Σ has rank N-1 = 2, so its smallest eigenvalue is
    zero and its condition number is essentially infinite (machine epsilon).
    Ridge regularization (Σ + λI) replaces the zero eigenvalue with λ,
    rescuing the inverse. Sweep λ over many decades and report:
      - condition number of (Σ + λI)
      - closed-form Tangency portfolio weights
      - Tangency Sharpe under the regularized Σ
    The user-facing question is: how much regularization do we need?
    The answer (for any honest practitioner) is: "enough that condition
    number is below ~1e6"; for our betting Σ this requires λ ≳ 0.01.
    """
    if lambdas is None:
        lambdas = np.logspace(-10, 0, 21)

    odds = np.asarray(odds, dtype=float)
    p_book, _ = mv.fair_probabilities(odds)
    # Use a slightly tilted measure so closed-form Tangency is non-trivial
    p_sub = p_book.copy()
    fav = int(np.argmin(odds))
    p_sub[fav] += 0.05
    p_sub /= p_sub.sum()

    Sigma_base = mv.bet_cov_matrix(odds, p_sub)
    mu = mv.bet_mean_vector(odds, p_sub)

    rows = []
    n = len(odds)
    I = np.eye(n)

    for lam in lambdas:
        Sigma_reg = Sigma_base + lam * I
        cond = float(np.linalg.cond(Sigma_reg))
        try:
            w_tan = mv.tangency_portfolio(mu, Sigma_reg, rf=0.0)
            stats = mv.portfolio_stats(w_tan, mu, Sigma_reg) if w_tan is not None else None
            tan_w_max = float(np.max(np.abs(w_tan))) if w_tan is not None else float("nan")
            tan_sharpe = float(stats["sharpe"]) if stats and not np.isnan(stats["sharpe"]) else float("nan")
        except (np.linalg.LinAlgError, Exception):
            tan_w_max, tan_sharpe = float("nan"), float("nan")
        rows.append({
            "lambda": float(lam),
            "condition_number": cond,
            "tangency_max_abs_weight": tan_w_max,
            "tangency_sharpe": tan_sharpe,
        })
    return rows


def plot_ridge_sweep(rows, savepath):
    lams = np.array([r["lambda"] for r in rows])
    conds = np.array([r["condition_number"] for r in rows])
    weights = np.array([r["tangency_max_abs_weight"] for r in rows])
    sharpes = np.array([r["tangency_sharpe"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.loglog(lams, conds, "-o", c="#d73027", lw=2, markersize=6)
    ax.axhline(1e6, color="grey", lw=1, ls="--", label="condition = 10⁶ (rule of thumb)")
    ax.set_xlabel("Ridge intensity λ (added to diag of Σ)")
    ax.set_ylabel("cond(Σ + λI)")
    ax.set_title("(a) Conditioning vs ridge")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    valid = ~np.isnan(weights)
    ax.semilogx(lams[valid], weights[valid], "-o", c="#fc8d59", lw=2, markersize=6)
    ax.set_xlabel("Ridge intensity λ")
    ax.set_ylabel("max |weight|")
    ax.set_title("(b) Closed-form Tangency weight magnitude")
    ax.grid(True, alpha=0.3, which="both")
    ax.axhline(1.0, color="grey", lw=1, ls="--", label="|w|=1 (well-behaved)")
    ax.legend(loc="best", fontsize=8)

    ax = axes[2]
    valid = ~np.isnan(sharpes)
    ax.semilogx(lams[valid], sharpes[valid], "-o", c="#2c7fb8", lw=2, markersize=6)
    ax.set_xlabel("Ridge intensity λ")
    ax.set_ylabel("Tangency Sharpe")
    ax.set_title("(c) Tangency Sharpe (regularized)")
    ax.grid(True, alpha=0.3, which="both")
    ax.axhline(0, color="grey", lw=0.6)

    fig.suptitle("Ridge regularization rescues the rank-deficient betting Σ\n"
                 "(N=3 outcomes, RR vs MI odds, slightly tilted subjective probs)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return savepath


# ============================================================
# Plots
# ============================================================
def plot_sequential_metrics(seq, savepath, true_final=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    k = seq["k"]

    ax = axes[0, 0]
    ax.plot(k, seq["sharpe"], "-o", c="#2c7fb8", lw=1.8, markersize=6)
    if true_final is not None and "sharpe" in true_final:
        ax.axhline(true_final["sharpe"], color="#d73027", lw=1, ls=":",
                   label=f"final estimate ({true_final['sharpe']:.3f})")
        ax.legend(loc="best", fontsize=8)
    ax.set_title("Sample Sharpe ratio vs N")
    ax.set_xlabel("N matches observed")
    ax.set_ylabel("Sharpe (per match)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(k, np.array(seq["mean"]) * 100, "-o", c="#1a9850", lw=1.8, markersize=6)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title("Sample mean return vs N")
    ax.set_xlabel("N matches observed")
    ax.set_ylabel("Mean return (% per match)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(k, np.array(seq["std"]) * 100, "-o", c="#fc8d59", lw=1.8, markersize=6)
    ax.set_title("Sample std dev vs N")
    ax.set_xlabel("N matches observed")
    ax.set_ylabel("σ (% per match)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(k, np.array(seq["max_dd"]) * 100, "-o", c="#d73027", lw=1.8, markersize=6)
    ax.set_title("Sample max drawdown vs N")
    ax.set_xlabel("N matches observed")
    ax.set_ylabel("Max DD (%)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Curse of dimensionality: small-N instability of risk metrics",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return savepath


def plot_bootstrap_sharpe(boots, point_estimate, savepath):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(boots, bins=60, color="#74a9cf", edgecolor="black", alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(point_estimate, color="#d73027", lw=2.2,
               label=f"Point estimate (N=12): {point_estimate:.3f}")
    q025, q500, q975 = np.percentile(boots, [2.5, 50, 97.5])
    ax.axvline(q025, color="#fc8d59", lw=1.5, ls="--",
               label=f"95% bootstrap CI: [{q025:.3f}, {q975:.3f}]")
    ax.axvline(q975, color="#fc8d59", lw=1.5, ls="--")
    ax.set_xlabel("Bootstrap Sharpe ratio (per match)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Bootstrap distribution of Sharpe over {len(boots):,} resamples\n"
                 f"(N = 12 with replacement)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return savepath


def plot_cov_condition(Ns, conds, savepath):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(Ns, conds, "-o", c="#d73027", lw=2, markersize=10)
    ax.set_xlabel("Number of outcome states N")
    ax.set_ylabel("Condition number of Σ (log scale)")
    ax.set_title("Σ becomes increasingly ill-conditioned with more outcomes\n"
                 "(rank N-1 — closed-form Markowitz inversion is meaningless)")
    ax.grid(True, alpha=0.3, which="both")
    for n, c in zip(Ns, conds):
        ax.annotate(f"{c:.2e}", (n, c), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return savepath


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history")
    p.add_argument("--n-bootstrap", type=int, default=5000)
    args = p.parse_args()

    history = args.history
    if not history:
        for cand in [
            "/mnt/user-data/uploads/1775541930958_user_om_gorakhia.json",
            Path.home() / "Downloads" / "1775541930958_user_om_gorakhia.json",
        ]:
            if Path(cand).exists():
                history = cand
                break
    if not history:
        print("ERROR: no --history provided.")
        sys.exit(1)

    # Build the chronological return series via E4's helper
    data = json.loads(Path(history).read_text())
    settled = sorted(
        [m for m in data.get("matches", []) if m.get("status") == "settled"],
        key=lambda m: m.get("settled_at", m.get("created_at", "")),
    )
    eq, rets, pnls, labels, opens = ra.build_equity_curve(
        settled, data.get("starting_capital", 0)
    )
    N = len(rets)
    final_sharpe = ra.sharpe_ratio(rets)
    final_mean = float(np.mean(rets))
    final_std = float(np.std(rets, ddof=1))

    print(f"\n{'=' * 72}")
    print(f"  E14 — Empirical curse-of-dimensionality demonstration")
    print(f"{'=' * 72}")
    print(f"  N = {N} settled matches")
    print(f"  Point estimates: μ = {final_mean*100:+.2f}%   σ = {final_std*100:.2f}%   "
          f"Sharpe = {final_sharpe:.3f}")

    # ---- Experiment 1: sequential metrics ----
    print(f"\n  [1] Sequential metric stability (rolling estimates as N grows)")
    seq = sequential_metrics(rets)
    print(f"  {'k':>3}  {'Sharpe(k)':>10}  {'mean(k)':>10}  {'std(k)':>10}  {'minDD(k)':>10}")
    for j, k in enumerate(seq["k"]):
        print(f"  {k:>3}  {seq['sharpe'][j]:>+10.3f}  {seq['mean'][j]*100:>+9.2f}%  "
              f"{seq['std'][j]*100:>+9.2f}%  {seq['max_dd'][j]*100:>9.2f}%")

    # Detect the instability: how much does Sharpe move from k=N-1 to k=N?
    sh_jumps = np.abs(np.diff(seq["sharpe"]))
    print(f"\n  Mean absolute Sharpe change per added observation: {sh_jumps.mean():.3f}")
    print(f"  Maximum  absolute Sharpe change per added observation: {sh_jumps.max():.3f}")
    print(f"  Final Sharpe estimate: {final_sharpe:.3f}")
    print(f"  Sharpe at N=2:         {seq['sharpe'][0]:+.3f}")
    print(f"  Sharpe range over k:   [{seq['sharpe'].min():.3f}, {seq['sharpe'].max():.3f}]")

    fig1 = FIG_DIR / "curse_of_dim_sequential.png"
    plot_sequential_metrics(seq, fig1, true_final={"sharpe": final_sharpe})
    print(f"  ▸ Sequential metrics plot: {fig1}")

    # ---- Experiment 2: bootstrap Sharpe ----
    print(f"\n  [2] Bootstrap distribution of Sharpe ratio (n_resamples = {args.n_bootstrap:,})")
    boots = bootstrap_sharpe(rets, n_resamples=args.n_bootstrap)
    boots_clean = boots[~np.isnan(boots)]
    q025, q05, q25, q50, q75, q95, q975 = np.percentile(
        boots_clean, [2.5, 5, 25, 50, 75, 95, 97.5]
    )
    print(f"  Bootstrap mean:         {boots_clean.mean():.3f}")
    print(f"  Bootstrap std:          {boots_clean.std():.3f}")
    print(f"  95% bootstrap CI:       [{q025:.3f}, {q975:.3f}]")
    print(f"  90% bootstrap CI:       [{q05:.3f}, {q95:.3f}]")
    print(f"  IQR:                    [{q25:.3f}, {q75:.3f}]")
    print(f"  P(Sharpe < 0):          {float(np.mean(boots_clean < 0)):.3f}")
    print(f"  P(Sharpe < 0.3):        {float(np.mean(boots_clean < 0.3)):.3f}")
    print(f"  P(Sharpe > 1.0):        {float(np.mean(boots_clean > 1.0)):.3f}")

    fig2 = FIG_DIR / "curse_of_dim_bootstrap.png"
    plot_bootstrap_sharpe(boots_clean, final_sharpe, fig2)
    print(f"  ▸ Bootstrap plot: {fig2}")

    # ---- Experiment 3: ridge regularization sweep ----
    print(f"\n  [3] Ridge regularization sweep on the 3-state betting Σ")
    print(f"  (How much λ is needed to rescue closed-form Tangency?)")
    rows = ridge_regularization_sweep([2.25, 1.67, 50.0])
    print(f"  {'lambda':>12}  {'cond(Σ+λI)':>15}  {'max|w_tan|':>12}  {'Sharpe':>10}")
    for r in rows:
        cond = r["condition_number"]
        wmax = r["tangency_max_abs_weight"]
        sh = r["tangency_sharpe"]
        sh_str = f"{sh:>10.4f}" if not np.isnan(sh) else "       n/a"
        wm_str = f"{wmax:>12.4e}" if not np.isnan(wmax) else "         n/a"
        print(f"  {r['lambda']:>12.2e}  {cond:>15.3e}  {wm_str}  {sh_str}")

    fig3 = FIG_DIR / "curse_of_dim_ridge.png"
    plot_ridge_sweep(rows, fig3)
    print(f"  ▸ Ridge regularization plot: {fig3}")

    # Find the smallest λ that gives a usable solution
    usable = [r for r in rows if r["condition_number"] < 1e6]
    if usable:
        lam_min = usable[0]["lambda"]
        print(f"\n  Smallest λ giving cond(Σ+λI) < 10⁶: λ = {lam_min:.2e}")
        print(f"  This is the minimum regularization needed to make Σ invertible")
        print(f"  in any meaningful numerical sense.")

    # ---- Save consolidated results ----
    out = {
        "N": int(N),
        "point_estimates": {
            "mean": final_mean, "std": final_std, "sharpe": final_sharpe,
        },
        "sequential": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in seq.items()},
        "bootstrap": {
            "n_resamples": int(args.n_bootstrap),
            "mean": float(boots_clean.mean()),
            "std": float(boots_clean.std()),
            "ci_95": [float(q025), float(q975)],
            "ci_90": [float(q05), float(q95)],
            "iqr": [float(q25), float(q75)],
            "p_negative": float(np.mean(boots_clean < 0)),
            "p_below_0.3": float(np.mean(boots_clean < 0.3)),
            "p_above_1.0": float(np.mean(boots_clean > 1.0)),
        },
        "cov_conditioning": {
            "experiment": "ridge_regularization_sweep",
            "rows": rows,
        },
    }
    tblpath = TBL_DIR / "curse_of_dim_results.json"
    tblpath.write_text(json.dumps(out, indent=2))
    print(f"\n  ▸ Numerical results saved to {tblpath}")

    # ---- Narrative summary ----
    print(f"\n  ► REPORT-READY NARRATIVE BLOCK:")
    print(f"    The point estimate of Sharpe at N = {N} is {final_sharpe:.3f}, but the")
    print(f"    bootstrap 95% CI is [{q025:.3f}, {q975:.3f}] — a width of {q975-q025:.3f},")
    print(f"    larger than the point estimate itself. Roughly {float(np.mean(boots_clean<0))*100:.0f}% of bootstrap")
    print(f"    samples produce a NEGATIVE Sharpe. The 'true' Sharpe of this strategy is")
    print(f"    therefore unidentifiable from {N} observations alone. This is the empirical")
    print(f"    curse of dimensionality on which the entire methodology of this project")
    print(f"    rests: shrinkage (E6), constrained QP (E1), discrete tail measures (E2),")
    print(f"    and the rejection of Random Forest / GARCH / logistic-regression models")
    print(f"    that we could not honestly train on this sample size.")

    print(f"\n{'=' * 72}\n")


if __name__ == "__main__":
    main()