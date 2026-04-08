#!/usr/bin/env python3
"""
01_mean_variance_hedge.py — Mean-Variance Portfolio Theory applied to
3-outcome in-play betting (IPL).

This module is the centrepiece of the QRM project's "Option A" framing.
It re-casts the existing LP-based hedge solver as a mean-variance portfolio
optimisation problem, mapping directly onto Wk 9–10 lecture material:

  - Each unit bet on an outcome is treated as a risky asset
  - Closed-form expected return, variance and covariance under bookmaker
    or subjective probabilities
  - Closed-form GMV and Tangency portfolio weights via the Wk 10 formulas
  - Efficient frontier in (sigma, E[r]) space, plotted as a Markowitz bullet
  - Constrained hedge problem (existing position + budget cap) solved as a
    quadratic program for the realistic decision-making case

USAGE
-----
    # Run with the baked-in tonight's RR vs MI snapshot:
    python3 01_mean_variance_hedge.py

    # Use a captured snapshot from data_feed.py:
    python3 01_mean_variance_hedge.py \
        --snapshot ../captures/2026-04-07__rajasthan_royals__vs__mumbai_indians.json

    # Use historical match log to do the cross-match analysis:
    python3 01_mean_variance_hedge.py \
        --history ~/Downloads/1775541930958_user_om_gorakhia.json

OUTPUTS
-------
    results/figures/efficient_frontier_<match>.png
    results/figures/historical_mv_weights.png
    results/tables/mv_results_<match>.json
    Console: report-ready tables

DERIVATIONS (for the report — copy-paste-able)
-----------------------------------------------
A unit bet on outcome i at decimal odds o_i has random per-unit return r_i:
    r_i = (o_i - 1)  with probability p_i      (the bet wins)
    r_i = -1         with probability 1 - p_i  (the bet loses)

Mean:
    E[r_i] = p_i (o_i - 1) - (1 - p_i) = p_i o_i - 1

Variance:
    Var(r_i) = p_i (o_i - 1 - E[r_i])^2 + (1 - p_i)(-1 - E[r_i])^2
             = p_i (1 - p_i) o_i^2

Covariance for i != j (in a 3-state world where exactly one outcome occurs):
    Cov(r_i, r_j) = - p_i p_j o_i o_j

Note: under "fair" probabilities (p_i = 1/o_i / sum_k 1/o_k after overround
removal) we have p_i o_i = 1/(1 + overround). All bets share the same
expected return = -overround/(1+overround), so the Tangency direction is
degenerate. To get a meaningful Tangency portfolio we use *subjective*
probabilities that differ from the bookmaker's implied measure — this is
exactly the role of the shrinkage estimator in module E6.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")  # no display required
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================
HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
FIG_DIR = RESULTS_DIR / "figures"
TBL_DIR = RESULTS_DIR / "tables"
for d in (RESULTS_DIR, FIG_DIR, TBL_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# Core math: closed-form mean, covariance, GMV, Tangency, EF
# ============================================================
def fair_probabilities(odds):
    """
    Convert bookmaker decimal odds to fair (overround-adjusted) probabilities.
    Returns (probs, overround_pct).

    odds : array-like of length N, decimal odds
    """
    odds = np.asarray(odds, dtype=float)
    raw = 1.0 / odds                              # implied probs (sum > 1)
    overround = raw.sum() - 1.0
    probs = raw / raw.sum()                       # normalized to sum = 1
    return probs, overround * 100.0               # overround as percent


def bet_mean_vector(odds, probs):
    """
    E[r_i] = p_i * o_i - 1 for each outcome i.
    """
    o = np.asarray(odds, dtype=float)
    p = np.asarray(probs, dtype=float)
    return p * o - 1.0


def bet_cov_matrix(odds, probs):
    """
    Closed-form covariance matrix of unit bets on each outcome.
    Sigma[i,i] = p_i (1 - p_i) o_i^2
    Sigma[i,j] = - p_i p_j o_i o_j   for i != j
    """
    o = np.asarray(odds, dtype=float)
    p = np.asarray(probs, dtype=float)
    n = len(o)
    Sigma = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                Sigma[i, i] = p[i] * (1 - p[i]) * o[i] ** 2
            else:
                Sigma[i, j] = -p[i] * p[j] * o[i] * o[j]
    return Sigma


def gmv_portfolio(Sigma):
    """
    Global Minimum Variance portfolio (Wk 10 closed form):
        w = Sigma^{-1} 1 / (1' Sigma^{-1} 1)
    Constraint: weights sum to 1 (allows short positions).
    Returns the weight vector.
    """
    n = Sigma.shape[0]
    ones = np.ones(n)
    inv = np.linalg.inv(Sigma)
    w = inv @ ones
    return w / (ones @ w)


def tangency_portfolio(mu, Sigma, rf=0.0):
    """
    Tangency portfolio with risk-free rate rf (Wk 10 closed form):
        w = Sigma^{-1} (mu - rf*1) / (1' Sigma^{-1} (mu - rf*1))
    Returns the weight vector. Degenerates if mu - rf*1 = 0 or its sum
    through Sigma^{-1} is zero.
    """
    n = len(mu)
    ones = np.ones(n)
    excess = mu - rf * ones
    inv = np.linalg.inv(Sigma)
    numer = inv @ excess
    denom = ones @ numer
    if abs(denom) < 1e-12:
        return None
    return numer / denom


def portfolio_stats(w, mu, Sigma, rf=0.0, sd_tol=1e-6):
    """
    Mean, std, Sharpe of a portfolio with weights w.
    Sharpe is reported as NaN if std < sd_tol (degenerate / risk-free).
    """
    er = float(w @ mu)
    var = float(w @ Sigma @ w)
    sd = float(np.sqrt(max(var, 0.0)))
    sharpe = (er - rf) / sd if sd > sd_tol else float("nan")
    return {"mean": er, "std": sd, "sharpe": sharpe, "degenerate": sd < sd_tol}


def efficient_frontier(mu, Sigma, n_points=80):
    """
    Compute the unconstrained efficient frontier (no short-sale constraints,
    only sum-to-one). Returns arrays (sigmas, mus, weights_list).

    Uses the two-fund separation theorem: every frontier portfolio is a
    linear combination of two known frontier portfolios. We parameterize by
    target return and solve closed-form for each.
    """
    n = len(mu)
    ones = np.ones(n)
    inv = np.linalg.inv(Sigma)
    A = ones @ inv @ ones
    B = ones @ inv @ mu
    C = mu @ inv @ mu
    D = A * C - B * B  # discriminant; > 0 for non-degenerate frontier

    if D <= 1e-12:
        # All assets share the same return; frontier collapses to a point
        return None, None, None

    mu_min = float(mu.min()) - abs(float(mu.min())) * 0.5 - 0.05
    mu_max = float(mu.max()) + abs(float(mu.max())) * 0.5 + 0.05
    targets = np.linspace(mu_min, mu_max, n_points)

    sigmas = []
    mus = []
    ws = []
    for t in targets:
        # Closed-form min-var portfolio for target return t (no short constraint)
        # w* = (g + h*t)
        g = (C * (inv @ ones) - B * (inv @ mu)) / D
        h = (A * (inv @ mu) - B * (inv @ ones)) / D
        w = g + h * t
        var = float(w @ Sigma @ w)
        if var < 0:
            continue
        sigmas.append(np.sqrt(var))
        mus.append(t)
        ws.append(w)
    return np.array(sigmas), np.array(mus), ws


# ============================================================
# Constrained hedge problem (the realistic case)
# ============================================================
def hedge_payoff_matrix(odds):
    """
    For a stake vector s = (s_1, ..., s_N), the additional P&L in state k is
    sum_i M[k,i] s_i where:
        M[k,i] = (o_i - 1)  if k == i  (the bet won)
        M[k,i] = -1         otherwise   (the bet lost)
    Existing position e in R^N + new payoff M @ s gives total P&L per state.
    """
    o = np.asarray(odds, dtype=float)
    n = len(o)
    M = -np.ones((n, n))
    for i in range(n):
        M[i, i] = o[i] - 1
    return M


def constrained_hedge_solver(existing_pnl, odds, probs, capital,
                              objective="gmv", subjective_probs=None):
    """
    Solve the realistic hedging problem:
        Given existing P&L vector e in R^N across the N outcome states,
        choose new stakes s >= 0 with sum(s) <= capital that minimise the
        variance (objective='gmv') or maximise the Sharpe ratio
        (objective='tangency') of the resulting total P&L distribution.

    Variance is computed under `probs` (the measure used for risk).
    If `objective='tangency'`, expected return is computed under
    `subjective_probs` (defaults to `probs`, in which case Tangency
    degenerates if odds are fair).

    Returns dict with stakes, total P&L per state, mean, std, Sharpe.
    """
    e = np.asarray(existing_pnl, dtype=float)
    p = np.asarray(probs, dtype=float)
    M = hedge_payoff_matrix(odds)
    n = len(odds)

    if subjective_probs is None:
        subjective_probs = probs
    p_sub = np.asarray(subjective_probs, dtype=float)

    def total_pnl(s):
        return e + M @ s

    def variance(s, prob_vec):
        v = total_pnl(s)
        mean = float(prob_vec @ v)
        return float(prob_vec @ ((v - mean) ** 2))

    def neg_sharpe(s):
        v = total_pnl(s)
        mean_sub = float(p_sub @ v)
        var = variance(s, p)  # variance under risk measure p
        if var <= 1e-12:
            return -1e9 if mean_sub > 0 else 0.0
        return -mean_sub / np.sqrt(var)

    s0 = np.full(n, capital / (4 * n))   # mild positive seed
    bounds = [(0.0, capital)] * n
    constraints = [{"type": "ineq", "fun": lambda s: capital - s.sum()}]

    if objective == "gmv":
        obj = lambda s: variance(s, p)
    elif objective == "tangency":
        obj = neg_sharpe
    else:
        raise ValueError(f"unknown objective: {objective}")

    res = minimize(obj, s0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"ftol": 1e-9, "maxiter": 300})

    s_opt = np.maximum(res.x, 0.0)
    v = total_pnl(s_opt)
    mean_p = float(p @ v)
    mean_sub = float(p_sub @ v)
    var = variance(s_opt, p)
    std = float(np.sqrt(max(var, 0.0)))
    sharpe = mean_sub / std if std > 1e-3 else float("nan")
    degenerate = std < 1e-3

    return {
        "stakes": s_opt,
        "total_stake": float(s_opt.sum()),
        "total_pnl_per_state": v,
        "mean_under_p": mean_p,
        "mean_under_subjective": mean_sub,
        "std": std,
        "sharpe": sharpe,
        "degenerate": degenerate,
        "success": bool(res.success),
        "objective": objective,
    }


# ============================================================
# Plotting
# ============================================================
def plot_efficient_frontier(odds, probs, subjective_probs, label, savepath,
                             existing_pnl=None, capital=10000.0):
    """
    Plot the Markowitz bullet for the 3 betting "assets" with GMV and
    Tangency marked, against the individual asset positions in (sigma, E[r])
    space.

    For 3-state betting markets the closed-form efficient frontier is
    DEGENERATE — it collapses to a single point because the GMV portfolio
    achieves zero variance (the probability-matching portfolio). We detect
    this and plot a feasibility cloud from the constrained QP instead, which
    is the realistic case any practitioner faces.
    """
    odds_arr = np.asarray(odds, dtype=float)
    odds_t = tuple(round(float(x), 2) for x in odds_arr)

    mu_book = bet_mean_vector(odds_arr, probs)
    Sigma_book = bet_cov_matrix(odds_arr, probs)
    mu_sub = bet_mean_vector(odds_arr, subjective_probs)
    Sigma_sub = bet_cov_matrix(odds_arr, subjective_probs)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # ---- 1. Individual assets (under subjective) ----
    asset_sds = np.sqrt(np.diag(Sigma_sub))
    asset_labels = ["T1 (home)", "T2 (away)", "Tie"]
    ax.scatter(asset_sds, mu_sub, s=160, c="#2c7fb8", zorder=5,
               edgecolors="black", linewidths=1.2, label="Individual bets")
    for i, lab in enumerate(asset_labels):
        ax.annotate(f"  {lab}\n  o={odds_arr[i]:.2f}",
                    (asset_sds[i], mu_sub[i]),
                    fontsize=8.5, va="center")

    # ---- 2. Feasibility cloud from constrained QP ----
    # Sample many random feasible stake vectors s ≥ 0, sum(s) ≤ capital,
    # and plot their (σ, E[r]) under subjective probs. Lower frontier = EF.
    e = np.zeros(3) if existing_pnl is None else np.asarray(existing_pnl, dtype=float)
    M = hedge_payoff_matrix(odds_arr)
    n_samples = 4000
    rng = np.random.default_rng(42)
    # Dirichlet ensures sum(weights) = 1, then scale by random budget [0, capital]
    dirich = rng.dirichlet([0.5, 0.5, 0.5], size=n_samples)
    budgets = rng.uniform(0, capital, size=n_samples)
    stakes = dirich * budgets[:, None]
    # P&L per state for each sample (rescaled to per-unit-of-budget for shape clarity)
    means = []
    sds = []
    for s in stakes:
        v = e + M @ s
        m = float(subjective_probs @ v)
        var = float(subjective_probs @ ((v - m) ** 2))
        means.append(m)
        sds.append(np.sqrt(max(var, 0.0)))
    sds = np.array(sds); means = np.array(means)
    # Normalize to per-unit-stake scale so they're comparable to the asset dots
    # Only do this when existing_pnl is zero (otherwise the magnitudes differ)
    if existing_pnl is None or np.allclose(e, 0):
        denom = np.maximum(stakes.sum(axis=1), 1e-9)
        sds_norm = sds / denom
        means_norm = means / denom
        ax.scatter(sds_norm, means_norm, s=2, c="#cccccc", alpha=0.5,
                   label=f"Feasible portfolios (n={n_samples})", zorder=2)

    # ---- 3. Closed-form GMV / TAN under subjective ----
    w_gmv = gmv_portfolio(Sigma_sub)
    s_gmv = portfolio_stats(w_gmv, mu_sub, Sigma_sub)
    ax.scatter([s_gmv["std"]], [s_gmv["mean"]], s=240, marker="s",
               c="#fc8d59", edgecolors="black", linewidths=1.4,
               label=f"GMV  σ={s_gmv['std']:.3f}  E[r]={s_gmv['mean']:.4f}",
               zorder=6)

    w_tan = tangency_portfolio(mu_sub, Sigma_sub, rf=0.0)
    if w_tan is not None:
        s_tan = portfolio_stats(w_tan, mu_sub, Sigma_sub)
        sharpe_str = f"Sharpe={s_tan['sharpe']:.3f}" if not np.isnan(s_tan['sharpe']) else "Sharpe=n/a"
        ax.scatter([s_tan["std"]], [s_tan["mean"]], s=240, marker="*",
                   c="#d73027", edgecolors="black", linewidths=1.4,
                   label=f"Tangency  {sharpe_str}", zorder=6)

    # ---- 4. Theoretical efficient frontier curve ----
    sd_f, mu_f, _ = efficient_frontier(mu_sub, Sigma_sub, n_points=120)
    if sd_f is not None and len(sd_f) > 1:
        ax.plot(sd_f, mu_f, "-", c="#1a9850", lw=2,
                label="Closed-form EF (subjective)")

    # ---- 5. Annotation if GMV is degenerate ----
    if s_gmv["degenerate"]:
        ax.annotate(
            "GMV is degenerate (σ ≈ 0):\n"
            "the closed-form solution\n"
            "matches bookmaker probs and\n"
            "achieves a riskless P&L of\n"
            f"{s_gmv['mean']:.4f} per unit stake.\n"
            "This is the 'pay the\n"
            "overround for certainty' bet.",
            xy=(s_gmv["std"], s_gmv["mean"]),
            xytext=(0.42, 0.55), textcoords="axes fraction",
            fontsize=8.5, ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff5e6", ec="#fc8d59", lw=1),
            arrowprops=dict(arrowstyle="->", color="#fc8d59", lw=1.2),
        )

    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("Standard deviation σ (per unit stake)")
    ax.set_ylabel("Expected return E[r] (per unit stake)")
    ax.set_title(f"Mean-variance bullet — {label}\n"
                 f"Decimal odds  T1={odds_t[0]}   T2={odds_t[1]}   Tie={odds_t[2]}")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return savepath


# ============================================================
# Demos
# ============================================================
def fmt_pct(x): return f"{x*100:+7.2f}%"
def fmt_num(x): return f"{x:+8.4f}"


def demo_single_match(odds, label, capital=10000.0, existing_pnl=None,
                      subjective_tilt=0.05):
    """
    Run the full mean-variance analysis on a single match snapshot.
    `odds` = (o_t1, o_t2, o_tie) decimal odds.
    `subjective_tilt` = how much to perturb fair probs toward the favourite
    (so Tangency is non-degenerate). 0.05 = +5pp on the favourite.
    """
    odds = np.asarray(odds, dtype=float)
    probs, overround = fair_probabilities(odds)

    # Build a tilted subjective prior — small overweight on the implied favourite.
    # This is a placeholder for what the E6 shrinkage estimator will provide.
    fav = int(np.argmin(odds))  # smallest odds = favourite
    sub = probs.copy()
    sub[fav] += subjective_tilt
    sub /= sub.sum()

    mu_book = bet_mean_vector(odds, probs)
    Sigma_book = bet_cov_matrix(odds, probs)
    mu_sub = bet_mean_vector(odds, sub)
    Sigma_sub = bet_cov_matrix(odds, sub)

    print(f"\n{'=' * 72}")
    print(f"  E1 — Mean-variance hedge analysis: {label}")
    print(f"{'=' * 72}")
    print(f"  Decimal odds:           T1={odds[0]:.3f}  T2={odds[1]:.3f}  Tie={odds[2]:.3f}")
    print(f"  Bookmaker overround:    {overround:+.2f}%")
    print(f"  Fair probabilities:     T1={probs[0]:.4f}  T2={probs[1]:.4f}  Tie={probs[2]:.4f}")
    print(f"  Subjective probs (+{subjective_tilt:.0%} on favourite):")
    print(f"                          T1={sub[0]:.4f}  T2={sub[1]:.4f}  Tie={sub[2]:.4f}")

    # ---- Per-asset stats ----
    print(f"\n  Per-asset (per ₹1 staked):")
    print(f"  {'Asset':10}  {'E[r] (book)':>12}  {'E[r] (sub.)':>12}  {'σ (sub.)':>12}")
    asset_sds = np.sqrt(np.diag(Sigma_sub))
    for i, name in enumerate(["T1", "T2", "Tie"]):
        print(f"  {name:10}  {fmt_num(mu_book[i]):>12}  {fmt_num(mu_sub[i]):>12}  {fmt_num(asset_sds[i]):>12}")

    # ---- Closed-form GMV / TAN ----
    w_gmv = gmv_portfolio(Sigma_sub)
    stats_gmv = portfolio_stats(w_gmv, mu_sub, Sigma_sub)

    w_tan = tangency_portfolio(mu_sub, Sigma_sub)
    if w_tan is not None:
        stats_tan = portfolio_stats(w_tan, mu_sub, Sigma_sub)
    else:
        stats_tan = None

    def _fmt_sharpe(s):
        return "  n/a (deg.)" if (s is None or np.isnan(s)) else f"{s:>10.4f}"

    print(f"\n  Closed-form portfolios (subjective probs, sum-to-1, shorts allowed):")
    print(f"  {'Portfolio':10}  {'w_T1':>10}  {'w_T2':>10}  {'w_Tie':>10}  {'E[r]':>10}  {'σ':>10}  {'Sharpe':>10}")
    print(f"  {'GMV':10}  {w_gmv[0]:>10.4f}  {w_gmv[1]:>10.4f}  {w_gmv[2]:>10.4f}  "
          f"{stats_gmv['mean']:>10.4f}  {stats_gmv['std']:>10.4f}  {_fmt_sharpe(stats_gmv['sharpe'])}")
    if stats_tan:
        print(f"  {'TAN':10}  {w_tan[0]:>10.4f}  {w_tan[1]:>10.4f}  {w_tan[2]:>10.4f}  "
              f"{stats_tan['mean']:>10.4f}  {stats_tan['std']:>10.4f}  {_fmt_sharpe(stats_tan['sharpe'])}")

    # ---- Highlight the probability-matching theorem ----
    # Check whether closed-form GMV equals the bookmaker fair probabilities.
    # This is a theoretically interesting degenerate case.
    if np.allclose(w_gmv, probs, atol=1e-3):
        print(f"\n  ► THEORETICAL FINDING: closed-form GMV = bookmaker fair probabilities.")
        print(f"    Under a 3-state betting market with p_i · o_i = constant (the no-arbitrage")
        print(f"    condition), the unconstrained GMV portfolio is the probability-matching one.")
        print(f"    It achieves σ = 0 and constant P&L = -overround/(1+overround) ≈ {-overround/100/(1+overround/100):.4f}")
        print(f"    per unit invested across ALL states. In other words, the textbook GMV solution")
        print(f"    is the trivial 'pay the bookmaker the margin in exchange for certainty' bet.")
        print(f"    This motivates the constrained problem below (s ≥ 0, sum(s) ≤ B).")

    # ---- Constrained hedge (realistic case) ----
    if existing_pnl is None:
        existing_pnl = np.zeros(3)
    e = np.asarray(existing_pnl, dtype=float)

    print(f"\n  Constrained hedge problem:")
    print(f"    Existing P&L per state: T1={e[0]:.0f}  T2={e[1]:.0f}  Tie={e[2]:.0f}")
    print(f"    Capital budget:         ₹{capital:,.0f}")

    res_gmv = constrained_hedge_solver(e, odds, probs, capital, objective="gmv",
                                        subjective_probs=sub)
    res_tan = constrained_hedge_solver(e, odds, probs, capital, objective="tangency",
                                        subjective_probs=sub)

    print(f"\n  {'Solution':18}  {'s_T1':>10}  {'s_T2':>10}  {'s_Tie':>10}  {'Σs':>10}  {'E[P&L]':>10}  {'σ':>10}  {'Sharpe':>12}")
    for label_, r in [("MV-GMV hedge", res_gmv), ("MV-TAN hedge", res_tan)]:
        s = r["stakes"]
        sharpe_str = "  n/a (deg.)" if r.get("degenerate") or np.isnan(r["sharpe"]) else f"{r['sharpe']:>12.4f}"
        print(f"  {label_:18}  {s[0]:>10.0f}  {s[1]:>10.0f}  {s[2]:>10.0f}  "
              f"{r['total_stake']:>10.0f}  {r['mean_under_subjective']:>10.0f}  "
              f"{r['std']:>10.0f}  {sharpe_str}")
        if r.get("degenerate"):
            v = r["total_pnl_per_state"]
            print(f"  {'':18}  → degenerate: P&L is constant ≈ {v.mean():.0f} across all states.")
            print(f"  {'':18}    This is the probability-matching ('full-cash-out') portfolio.")

    # Worst-case P&L for comparison with the existing LP solver's objective
    print(f"\n  Worst-case P&L per state under each solution:")
    print(f"  {'Solution':18}  {'min P&L':>10}  {'P&L if T1':>12}  {'P&L if T2':>12}  {'P&L if Tie':>12}")
    for label_, r in [("MV-GMV hedge", res_gmv), ("MV-TAN hedge", res_tan)]:
        v = r["total_pnl_per_state"]
        print(f"  {label_:18}  {v.min():>10.0f}  {v[0]:>12.0f}  {v[1]:>12.0f}  {v[2]:>12.0f}")

    # ---- Save plot ----
    safe_label = "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")
    figpath = FIG_DIR / f"efficient_frontier_{safe_label}.png"
    plot_efficient_frontier(odds, probs, sub, label, figpath)
    print(f"\n  ▸ Efficient frontier saved to {figpath}")

    # ---- Save numerical results JSON ----
    out = {
        "label": label,
        "odds": odds.tolist(),
        "overround_pct": overround,
        "fair_probs": probs.tolist(),
        "subjective_probs": sub.tolist(),
        "subjective_tilt_on_favourite": subjective_tilt,
        "asset_mean_under_book": mu_book.tolist(),
        "asset_mean_under_subjective": mu_sub.tolist(),
        "asset_std_under_subjective": asset_sds.tolist(),
        "gmv_weights": w_gmv.tolist(),
        "gmv_stats": stats_gmv,
        "tan_weights": (w_tan.tolist() if w_tan is not None else None),
        "tan_stats": stats_tan,
        "existing_pnl_per_state": e.tolist(),
        "capital": capital,
        "constrained_gmv_hedge": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                   for k, v in res_gmv.items()},
        "constrained_tan_hedge": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                   for k, v in res_tan.items()},
    }
    tblpath = TBL_DIR / f"mv_results_{safe_label}.json"
    tblpath.write_text(json.dumps(out, indent=2, default=str))
    print(f"  ▸ Numerical results saved to {tblpath}")
    return out


def demo_historical_matches(history_path):
    """
    Loop over the user's settled-match history and compute the MV portfolio
    that would have been recommended at the OPENING ODDS of each match.
    Compare against the realised P&L of the actual bets the user placed.
    """
    history_path = Path(history_path)
    if not history_path.exists():
        print(f"\n[!] History file not found: {history_path} — skipping cross-match analysis")
        return None

    data = json.loads(history_path.read_text())
    matches = [m for m in data.get("matches", []) if m.get("status") == "settled"]
    if not matches:
        print("[!] No settled matches in history.")
        return None

    print(f"\n{'=' * 72}")
    print(f"  E1 — Cross-match MV analysis on {len(matches)} settled matches")
    print(f"{'=' * 72}")
    print(f"  {'#':>2}  {'Match':25}  {'odds (T1/T2/Tie)':22}  {'GMV weights':28}  {'Real PnL':>10}")

    rows = []
    for i, m in enumerate(matches):
        oo = m.get("opening_odds")
        if not oo or not all(oo.get(k, 0) > 1 for k in ("t1", "t2", "tie")):
            continue
        odds = np.array([oo["t1"], oo["t2"], oo["tie"]], dtype=float)
        probs, overround = fair_probabilities(odds)
        Sigma = bet_cov_matrix(odds, probs)
        try:
            w_gmv = gmv_portfolio(Sigma)
        except np.linalg.LinAlgError:
            continue
        real_pnl = m.get("realized_pnl", 0) or 0
        rows.append({
            "match_id": m.get("id"),
            "label": m.get("label", ""),
            "opening_odds": odds.tolist(),
            "overround_pct": overround,
            "gmv_weights": w_gmv.tolist(),
            "realized_pnl": real_pnl,
        })
        print(f"  {i:>2}  {m.get('label','')[:25]:25}  "
              f"({odds[0]:5.2f}/{odds[1]:5.2f}/{odds[2]:5.1f})  "
              f"({w_gmv[0]:+.3f}, {w_gmv[1]:+.3f}, {w_gmv[2]:+.3f})    "
              f"{real_pnl:>10.0f}")

    # Plot GMV weights across matches
    if rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        idx = np.arange(len(rows))
        w1 = [r["gmv_weights"][0] for r in rows]
        w2 = [r["gmv_weights"][1] for r in rows]
        wt = [r["gmv_weights"][2] for r in rows]
        ax.bar(idx - 0.27, w1, width=0.27, label="w_T1", color="#2c7fb8")
        ax.bar(idx,        w2, width=0.27, label="w_T2", color="#fc8d59")
        ax.bar(idx + 0.27, wt, width=0.27, label="w_Tie", color="#999999")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(idx)
        ax.set_xticklabels([r["label"][:18] for r in rows], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("GMV weight")
        ax.set_title(f"GMV portfolio weights across {len(rows)} settled matches\n"
                     f"(at opening odds, fair probabilities)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        figpath = FIG_DIR / "historical_mv_weights.png"
        fig.savefig(figpath, dpi=150)
        plt.close(fig)
        print(f"\n  ▸ Cross-match GMV weights saved to {figpath}")

    tblpath = TBL_DIR / "historical_mv_weights.json"
    tblpath.write_text(json.dumps(rows, indent=2))
    print(f"  ▸ Cross-match results saved to {tblpath}")
    return rows


# ============================================================
# Snapshot loader
# ============================================================
def load_snapshot_odds(snapshot_path):
    """
    Read a capture file produced by data_feed.py and return the most recent
    median odds as (o_t1, o_t2, o_tie) and a label string.
    """
    d = json.loads(Path(snapshot_path).read_text())
    snaps = d.get("snapshots", [])
    if not snaps:
        return None, None
    last = snaps[-1]
    odds = (last["median_odds"]["t1"],
            last["median_odds"]["t2"],
            last["median_odds"]["tie"])
    label = f"{d.get('home_team')} vs {d.get('away_team')} ({last.get('phase','?')})"
    return odds, label


# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", help="Path to capture file from data_feed.py")
    p.add_argument("--history", help="Path to user JSON history file")
    p.add_argument("--capital", type=float, default=10000.0,
                   help="Capital budget for constrained hedge demo")
    p.add_argument("--tilt", type=float, default=0.05,
                   help="Subjective probability tilt on favourite (default 0.05)")
    args = p.parse_args()

    # Single-match demo
    if args.snapshot:
        odds, label = load_snapshot_odds(args.snapshot)
        if odds is None:
            print(f"[!] No snapshots in {args.snapshot}")
            return
    else:
        # Default: tonight's RR vs MI (median across 10 books, captured 14:44 SGT)
        odds = (2.250, 1.670, 50.0)
        label = "RR vs MI (Apr 7, pre-match snapshot)"

    demo_single_match(odds, label, capital=args.capital, subjective_tilt=args.tilt)

    # Existing-position scenario for the same match (demonstrates the constrained QP)
    print(f"\n{'=' * 72}")
    print(f"  Scenario: existing position — already long T1 (the underdog)")
    print(f"  (re-using the same odds and probs; new stakes only)")
    print(f"{'=' * 72}")
    # Simulate user backed T1 with ₹2000 at odds 2.40 earlier in the day
    stake, entry_odds = 2000.0, 2.40
    e = np.array([stake * (entry_odds - 1), -stake, -stake])

    odds_arr = np.asarray(odds, dtype=float)
    probs2, _ = fair_probabilities(odds_arr)
    fav = int(np.argmin(odds_arr))
    sub2 = probs2.copy(); sub2[fav] += args.tilt; sub2 /= sub2.sum()

    print(f"\n  Existing P&L per state: T1={e[0]:.0f}  T2={e[1]:.0f}  Tie={e[2]:.0f}")
    print(f"  Capital budget:         ₹{args.capital:,.0f}")

    res_gmv = constrained_hedge_solver(e, odds_arr, probs2, args.capital,
                                        objective="gmv", subjective_probs=sub2)
    res_tan = constrained_hedge_solver(e, odds_arr, probs2, args.capital,
                                        objective="tangency", subjective_probs=sub2)

    def _fmt_sharpe2(s, deg):
        return "  n/a (deg.)" if deg or np.isnan(s) else f"{s:>12.4f}"

    print(f"\n  {'Solution':18}  {'s_T1':>8}  {'s_T2':>8}  {'s_Tie':>8}  {'Σs':>8}  {'E[P&L]':>10}  {'σ':>10}  {'Sharpe':>12}")
    for label_, r in [("MV-GMV hedge", res_gmv), ("MV-TAN hedge", res_tan)]:
        s = r["stakes"]
        print(f"  {label_:18}  {s[0]:>8.0f}  {s[1]:>8.0f}  {s[2]:>8.0f}  "
              f"{r['total_stake']:>8.0f}  {r['mean_under_subjective']:>10.0f}  "
              f"{r['std']:>10.0f}  {_fmt_sharpe2(r['sharpe'], r.get('degenerate', False))}")

    print(f"\n  Total P&L per state under each solution:")
    print(f"  {'Solution':18}  {'min P&L':>10}  {'P&L if T1':>12}  {'P&L if T2':>12}  {'P&L if Tie':>12}")
    for label_, r in [("MV-GMV hedge", res_gmv), ("MV-TAN hedge", res_tan)]:
        v = r["total_pnl_per_state"]
        print(f"  {label_:18}  {v.min():>10.0f}  {v[0]:>12.0f}  {v[1]:>12.0f}  {v[2]:>12.0f}")

    # Save these existing-position results to a separate JSON
    out2 = {
        "scenario": "with_existing_T1_bet",
        "existing_pnl_per_state": e.tolist(),
        "capital": args.capital,
        "constrained_gmv_hedge": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                   for k, v in res_gmv.items()},
        "constrained_tan_hedge": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                   for k, v in res_tan.items()},
    }
    (TBL_DIR / "mv_results_existing_position.json").write_text(json.dumps(out2, indent=2, default=str))

    # Cross-match analysis
    if args.history:
        demo_historical_matches(args.history)
    else:
        # Try the bundled upload path automatically
        for cand in [
            "/mnt/user-data/uploads/1775541930958_user_om_gorakhia.json",
            Path.home() / "Downloads" / "1775541930958_user_om_gorakhia.json",
        ]:
            if Path(cand).exists():
                demo_historical_matches(cand)
                break

    print(f"\n{'=' * 72}")
    print(f"  E1 complete. All figures in {FIG_DIR}, tables in {TBL_DIR}.")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()