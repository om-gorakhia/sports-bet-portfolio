#!/usr/bin/env python3
"""
02_risk_adjusted_metrics.py — Sharpe, Sortino, Calmar and equity-curve risk
analytics for the QRM project (module E4).

Computes the standard Wk 9 risk-adjusted return metrics on the user's
realised match-level P&L history, plus drawdown and distributional summaries.
Also produces a clean equity curve plot with peak/drawdown shading and a
per-match return histogram.

The output table is the headline number set for the report's empirical
performance section.

USAGE
-----
    python3 02_risk_adjusted_metrics.py \
        --history /path/to/user_om_gorakhia.json

DEFINITIONS (for the report — verbatim from Wk 9 lecture, rf = 0)
------------------------------------------------------------------
    r_i        = realised return on match i = realized_pnl_i / opening_capital_i
    μ_r        = sample mean of {r_i}
    σ_r        = sample standard deviation of {r_i}
    σ_r⁻       = downside deviation = std of min(r_i, 0)

    Sharpe     = μ_r / σ_r
    Sortino    = μ_r / σ_r⁻
    Calmar     = μ_r / max_drawdown        (using returns, not P&L)

    Annualised: multiply by sqrt(N_per_year). For IPL we use N_per_year = 70
    (a regular IPL season is 70 league matches across ~2 months). The square-
    root scaling assumes returns are i.i.d.; we flag this as a limitation.

    max_drawdown    = max over t of (peak_equity_t - equity_t) / peak_equity_t
    profit_factor   = sum of positive P&L / |sum of negative P&L|
    win_rate        = #(r_i > 0) / N
    payoff_ratio    = mean win / |mean loss|
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

# A regular IPL season has 70 league matches (10 teams × 14 each / 2)
# across roughly 2 months. Used for "annualised" Sharpe scaling.
N_MATCHES_PER_SEASON = 70


# ============================================================
# Core metrics
# ============================================================
def sharpe_ratio(returns, rf=0.0):
    """Sharpe ratio with rf = 0. Returns NaN if variance is zero."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return float("nan")
    excess = r - rf
    sd = float(np.std(excess, ddof=1))
    if sd < 1e-12:
        return float("nan")
    return float(np.mean(excess) / sd)


def sortino_ratio(returns, rf=0.0):
    """
    Sortino ratio = mean excess return / downside deviation.
    Downside deviation only counts returns below the target (rf here).
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return float("nan")
    excess = r - rf
    downside = np.minimum(excess, 0.0)
    # Use the conventional definition: sqrt of mean of squared downside deviations
    dd = float(np.sqrt(np.mean(downside ** 2)))
    if dd < 1e-12:
        return float("nan")
    return float(np.mean(excess) / dd)


def max_drawdown(equity):
    """
    Max drawdown of an equity curve, expressed as a fraction (0–1).
    Returns (max_dd, peak_idx, trough_idx).
    """
    eq = np.asarray(equity, dtype=float)
    if len(eq) == 0:
        return 0.0, 0, 0
    peaks = np.maximum.accumulate(eq)
    dd = (peaks - eq) / peaks
    trough = int(np.argmax(dd))
    # Find the peak that preceded this trough
    peak = int(np.argmax(eq[:trough + 1])) if trough > 0 else 0
    return float(dd[trough]), peak, trough


def calmar_ratio(returns, equity):
    """Calmar = mean return / max drawdown."""
    r = np.asarray(returns, dtype=float)
    mdd, _, _ = max_drawdown(equity)
    if mdd < 1e-12:
        return float("nan")
    return float(np.mean(r) / mdd)


def profit_factor(pnls):
    """Sum of gains / |sum of losses|."""
    p = np.asarray(pnls, dtype=float)
    gains = p[p > 0].sum()
    losses = -p[p < 0].sum()
    if losses < 1e-12:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def payoff_ratio(returns):
    """Mean of winning returns / |mean of losing returns|."""
    r = np.asarray(returns, dtype=float)
    wins = r[r > 0]
    losses = r[r < 0]
    if len(wins) == 0 or len(losses) == 0:
        return float("nan")
    return float(np.mean(wins) / abs(np.mean(losses)))


def win_rate(returns):
    r = np.asarray(returns, dtype=float)
    return float(np.mean(r > 0))


# ============================================================
# Equity curve construction
# ============================================================
def build_equity_curve(matches, starting_capital):
    """
    Build (equity, returns, pnls, labels, opening_capitals) arrays from a
    chronologically-ordered list of settled matches.

    Equity uses each match's closing_capital field directly (ground truth
    from the data). This handles capital deposits/withdrawals between
    matches that wouldn't be visible from cumulative P&L alone.
    Per-match returns are pnl / opening_capital — the realised return on
    the actual capital deployed for that match.
    """
    eq = []
    rets = []
    pnls = []
    labels = []
    opens = []
    for m in matches:
        opening = float(m.get("opening_capital", 0) or 0)
        closing = float(m.get("closing_capital", 0) or 0)
        pnl = (m.get("realized_pnl", 0) or 0) + (m.get("misc_realized_pnl", 0) or 0)
        ret = pnl / opening if opening > 0 else 0.0
        eq.append(closing)
        rets.append(ret)
        pnls.append(pnl)
        labels.append(m.get("label", "")[:18])
        opens.append(opening)
    return (np.array(eq), np.array(rets),
            np.array(pnls), labels, np.array(opens))


# ============================================================
# Plotting
# ============================================================
def plot_equity_and_drawdown(equity, starting_capital, labels, savepath):
    """Equity curve with drawdown panel below."""
    eq = np.asarray(equity, dtype=float)
    full = np.concatenate([[starting_capital], eq])
    peaks = np.maximum.accumulate(full)
    dd = (peaks - full) / peaks * 100
    mdd, peak_idx, trough_idx = max_drawdown(full)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)
    x = np.arange(len(full))

    # Equity curve
    ax1.plot(x, full, "-o", c="#2c7fb8", lw=2, markersize=6,
             label="Equity")
    ax1.plot(x, peaks, "--", c="#999999", lw=1.0, label="Running peak")
    ax1.fill_between(x, full, peaks, where=(full < peaks),
                     color="#fee5d9", alpha=0.6, label="In drawdown")
    ax1.scatter([peak_idx], [full[peak_idx]], s=140, marker="^",
                c="#1a9850", edgecolors="black", linewidths=1.2,
                zorder=10, label=f"Peak ₹{full[peak_idx]:,.0f}")
    ax1.scatter([trough_idx], [full[trough_idx]], s=140, marker="v",
                c="#d73027", edgecolors="black", linewidths=1.2,
                zorder=10, label=f"Trough ₹{full[trough_idx]:,.0f}  (DD {mdd:.1%})")
    ax1.axhline(starting_capital, color="grey", lw=0.6, ls=":")
    ax1.set_ylabel("Equity (₹)")
    ax1.set_title(f"Equity curve over {len(equity)} settled matches  "
                  f"(start ₹{starting_capital:,.0f} → end ₹{full[-1]:,.0f}, "
                  f"net {full[-1]/starting_capital - 1:+.1%})")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.92)
    ax1.grid(True, alpha=0.3)

    # Drawdown sub-plot
    ax2.fill_between(x, 0, -dd, color="#d73027", alpha=0.6)
    ax2.plot(x, -dd, c="#d73027", lw=1.5)
    ax2.axhline(0, color="grey", lw=0.6)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Match index")
    ax2.grid(True, alpha=0.3)
    xticks = list(range(0, len(full), max(1, len(full) // 12)))
    ax2.set_xticks(xticks)

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return savepath


def plot_return_distribution(returns, savepath):
    """Per-match return histogram with stats overlay."""
    r = np.asarray(returns, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(r * 100, bins=12, color="#74a9cf", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(np.mean(r) * 100, color="#1a9850", lw=2, ls="--",
               label=f"Mean = {np.mean(r)*100:+.1f}%")
    ax.axvline(np.median(r) * 100, color="#fc8d59", lw=2, ls="--",
               label=f"Median = {np.median(r)*100:+.1f}%")
    ax.set_xlabel("Per-match return (% of opening capital)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of per-match returns  (N = {len(r)})")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return savepath


# ============================================================
# Main analysis
# ============================================================
def analyse(history_path, n_per_year=N_MATCHES_PER_SEASON):
    history_path = Path(history_path)
    if not history_path.exists():
        print(f"[X] History file not found: {history_path}")
        return None

    data = json.loads(history_path.read_text())
    starting_capital = float(data.get("starting_capital", 0))
    all_matches = data.get("matches", [])
    settled = [m for m in all_matches if m.get("status") == "settled"]
    # Sort chronologically
    settled.sort(key=lambda m: m.get("settled_at", m.get("created_at", "")))

    if len(settled) == 0:
        print("[X] No settled matches found.")
        return None

    eq, rets, pnls, labels, opens = build_equity_curve(settled, starting_capital)

    # Core metrics
    n = len(rets)
    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets, ddof=1)) if n > 1 else float("nan")
    sr = sharpe_ratio(rets)
    so = sortino_ratio(rets)
    sr_ann = sr * np.sqrt(n_per_year) if not np.isnan(sr) else float("nan")
    so_ann = so * np.sqrt(n_per_year) if not np.isnan(so) else float("nan")
    full_eq = np.concatenate([[starting_capital], eq])
    mdd, peak_idx, trough_idx = max_drawdown(full_eq)
    cal = calmar_ratio(rets, full_eq)
    pf = profit_factor(pnls)
    pr = payoff_ratio(rets)
    wr = win_rate(rets)
    total_pnl = float(np.sum(pnls))
    total_return = total_pnl / starting_capital
    cagr_proxy = (1 + total_return) ** (n_per_year / n) - 1  # if these returns repeated for a full season

    # Print report-ready table
    print(f"\n{'=' * 72}")
    print(f"  E4 — Risk-adjusted performance metrics")
    print(f"{'=' * 72}")
    print(f"  Source:           {history_path.name}")
    print(f"  Settled matches:  {n}")
    print(f"  Starting capital: ₹{starting_capital:,.2f}")
    print(f"  Ending capital:   ₹{full_eq[-1]:,.2f}")
    print(f"  Total P&L:        ₹{total_pnl:+,.2f}")
    print(f"  Total return:     {total_return:+.2%}")
    print()
    print(f"  Per-match return statistics:")
    print(f"    Mean (μ_r):     {mean_r:+.4f}  ({mean_r*100:+.2f}%)")
    print(f"    Std  (σ_r):     {std_r:.4f}  ({std_r*100:.2f}%)")
    print(f"    Min:            {np.min(rets):+.4f}  ({np.min(rets)*100:+.2f}%)")
    print(f"    Max:            {np.max(rets):+.4f}  ({np.max(rets)*100:+.2f}%)")
    print(f"    Skew:           {float(((rets-mean_r)**3).mean() / std_r**3 if std_r > 0 else 0):+.4f}")
    print(f"    Kurtosis:       {float(((rets-mean_r)**4).mean() / std_r**4 - 3 if std_r > 0 else 0):+.4f}  (excess)")
    print()
    print(f"  Risk-adjusted ratios (rf = 0):")
    print(f"    Sharpe (per match):     {sr:+.4f}")
    print(f"    Sortino (per match):    {so:+.4f}")
    print(f"    Calmar:                 {cal:+.4f}")
    print(f"    Sharpe (annualised):    {sr_ann:+.4f}    [× √{n_per_year}]")
    print(f"    Sortino (annualised):   {so_ann:+.4f}    [× √{n_per_year}]")
    print()
    print(f"  Drawdown:")
    print(f"    Max DD:                 {mdd:.2%}")
    print(f"    Peak match #{peak_idx}:        ₹{full_eq[peak_idx]:,.0f}")
    print(f"    Trough match #{trough_idx}:    ₹{full_eq[trough_idx]:,.0f}")
    print()
    print(f"  Other metrics:")
    print(f"    Win rate:               {wr:.2%}    ({int(wr*n)}/{n})")
    print(f"    Profit factor:          {pf:.3f}")
    print(f"    Payoff ratio:           {pr:.3f}")
    print()

    # Per-match table
    print(f"  Per-match returns (chronological):")
    print(f"  {'#':>2}  {'match':18}  {'open cap':>12}  {'pnl':>12}  {'return':>10}  {'equity':>12}")
    for i, (lab, op, p, r, e) in enumerate(zip(labels, opens, pnls, rets, eq)):
        print(f"  {i+1:>2}  {lab:18}  {op:>12,.0f}  {p:>+12,.0f}  {r*100:>+9.2f}%  {e:>12,.0f}")

    # Save plots
    fig1 = FIG_DIR / "equity_curve.png"
    fig2 = FIG_DIR / "return_distribution.png"
    plot_equity_and_drawdown(eq, starting_capital, labels, fig1)
    plot_return_distribution(rets, fig2)
    print(f"\n  ▸ Equity curve saved to {fig1}")
    print(f"  ▸ Return distribution saved to {fig2}")

    # Save numerical results
    out = {
        "source": str(history_path),
        "n_matches": n,
        "starting_capital": starting_capital,
        "ending_capital": float(full_eq[-1]),
        "total_pnl": total_pnl,
        "total_return": total_return,
        "mean_return_per_match": mean_r,
        "std_return_per_match": std_r,
        "min_return": float(np.min(rets)),
        "max_return": float(np.max(rets)),
        "sharpe_per_match": sr,
        "sortino_per_match": so,
        "calmar": cal,
        "sharpe_annualised": sr_ann,
        "sortino_annualised": so_ann,
        "n_per_year_assumed": n_per_year,
        "max_drawdown_pct": mdd,
        "peak_match_index": peak_idx,
        "trough_match_index": trough_idx,
        "win_rate": wr,
        "profit_factor": pf,
        "payoff_ratio": pr,
        "per_match": [
            {"index": i + 1, "label": lab, "opening_capital": float(op),
             "pnl": float(p), "return": float(r), "equity_after": float(e)}
            for i, (lab, op, p, r, e)
            in enumerate(zip(labels, opens, pnls, rets, eq))
        ],
    }
    tblpath = TBL_DIR / "risk_adjusted_metrics.json"
    tblpath.write_text(json.dumps(out, indent=2))
    print(f"  ▸ Metrics JSON saved to {tblpath}")
    print(f"\n{'=' * 72}\n")
    return out


# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", help="Path to user history JSON")
    p.add_argument("--n-per-year", type=int, default=N_MATCHES_PER_SEASON,
                   help=f"Matches per season for annualisation (default {N_MATCHES_PER_SEASON})")
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

    analyse(history, n_per_year=args.n_per_year)


if __name__ == "__main__":
    main()