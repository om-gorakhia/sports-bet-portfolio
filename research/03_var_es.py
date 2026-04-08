#!/usr/bin/env python3
"""
03_var_es.py — Value-at-Risk and Expected Shortfall analysis (module E2).

Two complementary computations of VaR_α and ES_α:

  1. Per-match (analytic): the 3-state {T1, T2, Tie} P&L distribution under
     overround-adjusted bookmaker probabilities. VaR and ES are computed
     exactly from the discrete distribution — no estimation noise.

  2. Across-match (empirical / historical): the realised per-match return
     distribution from the 12 settled matches. VaR is the historical
     quantile, ES is the average of returns below that quantile.

Connects to Wk 11 directly: shows the well-known result that VaR is non-
coherent (sub-additivity can fail) and that ES is preferred for tail-risk
measurement, with a worked numerical example on the user's own data.

USAGE
-----
    python3 03_var_es.py --history /path/to/user_om_gorakhia.json

DEFINITIONS (Wk 11 — verbatim, with rf = 0)
--------------------------------------------
For a loss random variable L (= - P&L) and confidence level α ∈ (0,1):

    VaR_α(L)  =  inf { x  :  P(L ≤ x) ≥ α }
              =  the α-quantile of the loss distribution

    ES_α(L)   =  E[ L | L ≥ VaR_α(L) ]
              =  average loss in the worst (1 − α) fraction of outcomes

We typically report at α = 0.95 (worst 5%) and α = 0.99 (worst 1%).
ES dominates VaR because it (a) is a coherent risk measure (sub-additive)
and (b) describes how bad things actually get in the tail, not just where
the tail starts.
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
# Discrete (analytic) VaR / ES
# ============================================================
def discrete_var_es(losses, probs, alpha=0.95):
    """
    Compute VaR_α and ES_α exactly for a finite discrete loss distribution
    {(p_i, L_i)}. Returns dict with var, es, and the worst-case states used.

    For a discrete distribution, VaR_α is the smallest L value such that
    P(L ≤ L) ≥ α. ES_α is the conditional mean over the tail above VaR_α,
    properly weighted (with the standard handling of the atom at VaR_α
    when probabilities don't divide cleanly into 1 - α).
    """
    losses = np.asarray(losses, dtype=float)
    probs = np.asarray(probs, dtype=float)

    order = np.argsort(losses)         # ascending in loss
    L_sorted = losses[order]
    p_sorted = probs[order]

    # CDF
    cdf = np.cumsum(p_sorted)
    # Smallest L_sorted index where cdf >= alpha
    idx = int(np.searchsorted(cdf, alpha, side="left"))
    idx = min(idx, len(L_sorted) - 1)
    var = float(L_sorted[idx])

    # ES: weighted mean of losses in the (1 - alpha) tail.
    # Standard formula handling the atom at VaR (Acerbi & Tasche 2002):
    #   ES_α = (1/(1-α)) * [ E[L · 1{L > VaR}] + VaR · (P(L ≤ VaR) - α) ]
    # This degenerates correctly for finite-support distributions.
    tail_mass_above = float(np.sum(p_sorted[idx + 1:]))
    expected_above = float(np.sum(L_sorted[idx + 1:] * p_sorted[idx + 1:]))
    p_var_excess = float(cdf[idx] - alpha)  # how much of VaR's atom is in tail
    es_numer = expected_above + var * p_var_excess
    es = es_numer / (1 - alpha) if (1 - alpha) > 1e-12 else var

    return {"var": var, "es": es, "alpha": alpha,
            "states_above_var": [(float(L_sorted[i]), float(p_sorted[i]))
                                  for i in range(idx, len(L_sorted))]}


# ============================================================
# Empirical (historical) VaR / ES
# ============================================================
def historical_var_es(returns_or_losses, alpha=0.95, are_losses=False):
    """
    Empirical VaR_α and ES_α from a sample. By convention, the input is
    interpreted as RETURNS (positive = good); set are_losses=True to invert.
    Returns are converted to losses internally.
    """
    x = np.asarray(returns_or_losses, dtype=float)
    if not are_losses:
        L = -x
    else:
        L = x

    if len(L) == 0:
        return {"var": float("nan"), "es": float("nan"), "alpha": alpha}

    # VaR is the alpha-quantile of losses.
    # numpy.quantile uses linear interpolation by default, which is fine for
    # a smooth empirical distribution but slightly biased for tiny samples.
    # We use the "lower" interpolation to be conservative (matches historical
    # quantile convention).
    var = float(np.quantile(L, alpha, method="higher"))
    tail = L[L >= var]
    es = float(np.mean(tail)) if len(tail) > 0 else var
    return {"var": var, "es": es, "alpha": alpha,
            "n_in_tail": int(len(tail)), "n_total": int(len(L))}


# ============================================================
# Per-match analytic VaR/ES from a single odds snapshot
# ============================================================
def fair_probabilities(odds):
    o = np.asarray(odds, dtype=float)
    raw = 1.0 / o
    return raw / raw.sum()


def match_pnl_per_state(bets, odds):
    """
    Given a list of bets [{outcome, odds, stake}, ...] and the 3 outcome
    keys, return the P&L vector across the 3 states (T1, T2, Tie).
    """
    pnl = np.zeros(3)
    keys = ["t1", "t2", "tie"]
    for b in bets:
        outcome = b.get("outcome")
        if outcome not in keys:
            continue
        i = keys.index(outcome)
        stake = float(b.get("stake", 0))
        bet_odds = float(b.get("odds", 0))
        # Bet wins iff its outcome is the state
        for k in range(3):
            if k == i:
                pnl[k] += stake * (bet_odds - 1)
            else:
                pnl[k] -= stake
    return pnl


# ============================================================
# Plots
# ============================================================
def plot_discrete_loss_distribution(losses, probs, var, es, label, savepath, alpha=0.95):
    """Bar chart of the 3-state loss distribution with VaR and ES marked."""
    losses = np.asarray(losses, dtype=float)
    probs = np.asarray(probs, dtype=float)
    state_names = ["T1 wins", "T2 wins", "Tie"]
    order = np.argsort(losses)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(range(3), losses[order],
                  width=0.6, color=["#2c7fb8", "#74a9cf", "#bdc9e1"][:3],
                  edgecolor="black", linewidth=1)
    for i, idx in enumerate(order):
        ax.text(i, losses[idx] + (max(abs(losses)) * 0.02 if losses[idx] >= 0 else -max(abs(losses)) * 0.04),
                f"L = {losses[idx]:+.0f}\np = {probs[idx]:.3f}",
                ha="center", fontsize=9,
                va="bottom" if losses[idx] >= 0 else "top")
    ax.set_xticks(range(3))
    ax.set_xticklabels([state_names[idx] for idx in order])
    ax.axhline(0, color="grey", lw=0.6)
    ax.axhline(var, color="#fc8d59", lw=2, ls="--", label=f"VaR_{int(alpha*100)} = {var:+.0f}")
    ax.axhline(es, color="#d73027", lw=2, ls="--", label=f"ES_{int(alpha*100)} = {es:+.0f}")
    ax.set_ylabel("Loss (- P&L)")
    ax.set_title(f"Discrete loss distribution: {label}")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return savepath


def plot_empirical_loss_distribution(returns, var95, es95, var99, es99, savepath):
    """Histogram of empirical losses with VaR/ES at two confidence levels."""
    losses = -np.asarray(returns, dtype=float) * 100  # convert to % loss
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(losses, bins=12, color="#74a9cf", edgecolor="black", alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(var95 * 100, color="#fc8d59", lw=2.2, ls="--",
               label=f"VaR_95 = {var95*100:+.1f}%")
    ax.axvline(es95 * 100, color="#d73027", lw=2.2, ls="--",
               label=f"ES_95 = {es95*100:+.1f}%")
    if not np.isnan(var99):
        ax.axvline(var99 * 100, color="#990000", lw=2.2, ls=":",
                   label=f"VaR_99 = {var99*100:+.1f}%")
    if not np.isnan(es99):
        ax.axvline(es99 * 100, color="#660000", lw=2.2, ls=":",
                   label=f"ES_99 = {es99*100:+.1f}%")
    ax.set_xlabel("Loss as % of opening capital  (positive = bad)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Empirical per-match loss distribution  (N = {len(losses)})\n"
                 f"VaR and ES from historical method")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return savepath


# ============================================================
# Analyses
# ============================================================
def analyse_per_match(matches, alpha=0.95):
    """
    For each settled match: reconstruct the bet ledger's 3-state P&L
    distribution under fair odds, compute exact VaR and ES.
    """
    rows = []
    print(f"\n  Per-match analytic VaR_{int(alpha*100)} / ES_{int(alpha*100)} from final bet ledger:")
    print(f"  {'#':>2}  {'match':18}  {'opening odds':22}  "
          f"{'PnL T1':>9}  {'PnL T2':>9}  {'PnL Tie':>9}  "
          f"{'VaR':>9}  {'ES':>9}")

    for i, m in enumerate(matches):
        if m.get("status") != "settled":
            continue
        opening = m.get("opening_odds")
        bets = (m.get("bets") or []) + (m.get("pre_bets") or [])
        if not opening or not bets:
            continue
        odds = np.array([opening.get("t1", 0), opening.get("t2", 0), opening.get("tie", 0)],
                        dtype=float)
        if not all(odds > 1):
            continue
        probs = fair_probabilities(odds)
        pnl_states = match_pnl_per_state(bets, odds)
        losses = -pnl_states
        result = discrete_var_es(losses, probs, alpha=alpha)
        rows.append({
            "match_id": m.get("id"),
            "label": m.get("label", ""),
            "opening_odds": odds.tolist(),
            "fair_probs": probs.tolist(),
            "pnl_per_state": pnl_states.tolist(),
            "var": result["var"],
            "es": result["es"],
            "alpha": alpha,
        })
        print(f"  {i:>2}  {m.get('label','')[:18]:18}  "
              f"({odds[0]:5.2f}/{odds[1]:5.2f}/{odds[2]:5.1f})   "
              f"{pnl_states[0]:>+9.0f}  {pnl_states[1]:>+9.0f}  {pnl_states[2]:>+9.0f}  "
              f"{result['var']:>+9.0f}  {result['es']:>+9.0f}")

    return rows


def analyse_empirical(matches, alphas=(0.95, 0.99)):
    """Empirical VaR / ES on the realised per-match return distribution."""
    settled = [m for m in matches if m.get("status") == "settled"]
    settled.sort(key=lambda m: m.get("settled_at", m.get("created_at", "")))

    rets = []
    for m in settled:
        opening = float(m.get("opening_capital", 0) or 0)
        pnl = (m.get("realized_pnl", 0) or 0) + (m.get("misc_realized_pnl", 0) or 0)
        if opening > 0:
            rets.append(pnl / opening)
    rets = np.array(rets)

    out = {"n": int(len(rets)), "mean_return": float(np.mean(rets)),
           "std_return": float(np.std(rets, ddof=1)) if len(rets) > 1 else float("nan"),
           "results": {}}
    for a in alphas:
        r = historical_var_es(rets, alpha=a)
        out["results"][f"alpha_{int(a*100)}"] = r

    print(f"\n{'=' * 72}")
    print(f"  E2 — Empirical (historical) VaR / ES on realised returns")
    print(f"{'=' * 72}")
    print(f"  N = {len(rets)} settled matches")
    print(f"  Mean return per match: {np.mean(rets)*100:+.2f}%")
    print(f"  Std  return per match: {np.std(rets, ddof=1)*100:.2f}%")
    print()
    print(f"  {'α':>5}  {'VaR_α':>12}  {'ES_α':>12}  {'tail size':>12}")
    for a in alphas:
        r = out["results"][f"alpha_{int(a*100)}"]
        print(f"  {a*100:>4.0f}%  {r['var']*100:>+11.2f}%  {r['es']*100:>+11.2f}%  "
              f"{r['n_in_tail']}/{r['n_total']}")

    print(f"\n  Interpretation:")
    var95 = out["results"]["alpha_95"]["var"]
    es95 = out["results"]["alpha_95"]["es"]
    print(f"    With 95% confidence, the worst per-match loss is no more than {var95*100:.1f}% of capital.")
    print(f"    Conditional on falling into the worst 5% of matches, the expected loss is {es95*100:.1f}%.")
    print(f"    Note: with N = {len(rets)}, the 95% empirical quantile is determined by the worst")
    print(f"    1 observation. With N = 12 we cannot estimate VaR_99 reliably (it requires ≥100 obs).")

    return out, rets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", help="Path to user history JSON")
    p.add_argument("--alpha", type=float, default=0.95)
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
        print("ERROR: no --history provided and no default found")
        sys.exit(1)

    data = json.loads(Path(history).read_text())
    matches = data.get("matches", [])

    print(f"\n{'=' * 72}")
    print(f"  E2 — Per-match analytic VaR / ES (Wk 11)")
    print(f"{'=' * 72}")
    per_match_rows = analyse_per_match(matches, alpha=args.alpha)

    # Empirical analysis
    empirical, rets = analyse_empirical(matches, alphas=(0.95, 0.99))

    # Save full results
    out = {"per_match_analytic": per_match_rows, "empirical": empirical}
    tblpath = TBL_DIR / "var_es_results.json"
    tblpath.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  ▸ Numerical results saved to {tblpath}")

    # Plot empirical loss distribution
    var95 = empirical["results"]["alpha_95"]["var"]
    es95 = empirical["results"]["alpha_95"]["es"]
    var99 = empirical["results"]["alpha_99"]["var"]
    es99 = empirical["results"]["alpha_99"]["es"]
    figpath = FIG_DIR / "empirical_loss_distribution.png"
    plot_empirical_loss_distribution(rets, var95, es95, var99, es99, figpath)
    print(f"  ▸ Empirical loss distribution plot: {figpath}")

    # Plot the worst per-match analytic case (largest VaR)
    if per_match_rows:
        worst = max(per_match_rows, key=lambda r: r["var"])
        figpath2 = FIG_DIR / f"loss_distribution_{worst['label'].replace(' ','_')}.png"
        plot_discrete_loss_distribution(
            -np.array(worst["pnl_per_state"]),
            np.array(worst["fair_probs"]),
            worst["var"], worst["es"], worst["label"],
            figpath2, alpha=args.alpha,
        )
        print(f"  ▸ Worst per-match loss distribution ({worst['label']}): {figpath2}")

    print(f"\n{'=' * 72}\n")


if __name__ == "__main__":
    main()