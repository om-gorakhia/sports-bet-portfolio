#!/usr/bin/env python3
"""
06_delta_hedge_analogy.py — Bets-as-binary-options framing and position
delta computation (module E8).

Connects in-play sports betting to Wk 4 derivatives lecture: a unit bet on
outcome i at decimal odds o_i is structurally equivalent to buying o_i
units of a binary "asset-or-nothing" claim that pays $1 if outcome i occurs
and $0 otherwise. The bookmaker sells this claim at price 1/o_i, which
exceeds the true probability p_i by the overround margin.

This framing yields a clean Wk 4–style "delta" for any portfolio of bets:
the sensitivity of expected position value to a change in the market's
implied probability for each outcome.

DEFINITIONS
-----------
A unit bet on outcome i at decimal odds o_i:
    payoff = (o_i - 1)  if outcome i occurs   (with prob p_i)
    payoff = -1         otherwise              (with prob 1 - p_i)
    E[payoff | p] = p_i · o_i - 1

For a stake s on outcome i, total expected payoff:
    V_i(p) = s · (p_i · o_i - 1)

For a portfolio of bets B with stakes s_b on outcomes outcome(b) at odds o_b:
    V(p) = Σ_{b in B}  s_b · (p_{outcome(b)} · o_b - 1)

The "implied-probability delta" of the portfolio for outcome i:
    Δ_i = ∂V / ∂p_i = Σ_{b in B, outcome(b) = i}  s_b · o_b

This is the total notional exposure to outcome i, in dollars per unit
change in the market's implied probability of that outcome.

DELTA-NEUTRAL HEDGING
---------------------
A position is delta-neutral in the betting sense if all Δ_i are equal
(equal exposure across all outcomes — gain/loss is invariant to which
side the market moves toward). For a pre-existing position with deltas
Δ = (Δ_1, ..., Δ_N), the new stakes (s'_1, ..., s'_N) needed to achieve
delta-neutrality are:

    s'_i = (Δ_max - Δ_i) / o_i

where Δ_max = max_j Δ_j. This produces equal Δ_i across all outcomes,
making the portfolio insensitive to "where the market moves next."

USAGE
-----
    python3 06_delta_hedge_analogy.py --history /path/to/user.json
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


HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
FIG_DIR = RESULTS_DIR / "figures"
TBL_DIR = RESULTS_DIR / "tables"
for d in (RESULTS_DIR, FIG_DIR, TBL_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# Position value and delta
# ============================================================
def position_value(bets, probs):
    """
    Expected total P&L of a portfolio of bets under the given probability
    measure.
    bets: list of {outcome, odds, stake}
    probs: dict {t1: p1, t2: p2, tie: pt}
    """
    total = 0.0
    for b in bets:
        out = b.get("outcome")
        if out not in ("t1", "t2", "tie"):
            continue
        s = float(b.get("stake", 0))
        o = float(b.get("odds", 1))
        p = float(probs.get(out, 0))
        total += s * (p * o - 1.0)
    return total


def position_delta(bets):
    """
    The delta vector for a portfolio of bets:
        Δ_i = Σ over bets on outcome i of (stake · odds)
    Returns dict {t1: Δ1, t2: Δ2, tie: Δ_tie}.
    """
    delta = {"t1": 0.0, "t2": 0.0, "tie": 0.0}
    for b in bets:
        out = b.get("outcome")
        if out not in delta:
            continue
        s = float(b.get("stake", 0))
        o = float(b.get("odds", 1))
        delta[out] += s * o
    return delta


def delta_neutralizing_stakes(delta, current_odds):
    """
    Compute the additional stakes needed (at current odds) to make the
    portfolio's delta vector uniform across outcomes — i.e., delta-neutral.

    For each outcome i, we want new Δ'_i = Δ_target (some constant).
    Current Δ_i + s'_i · o_i^current = Δ_target
    => s'_i = (Δ_target - Δ_i) / o_i^current

    Δ_target is chosen as max(Δ_i) so that all stakes are non-negative
    (we cannot place negative bets — that would require betting against
    ourselves on the same exchange).
    """
    delta_target = max(delta.values())
    stakes_needed = {}
    for outcome in ("t1", "t2", "tie"):
        d = delta.get(outcome, 0.0)
        o = current_odds.get(outcome, 1.0)
        if o > 1.01:
            stakes_needed[outcome] = (delta_target - d) / o
        else:
            stakes_needed[outcome] = 0.0
    return stakes_needed, delta_target


# ============================================================
# Plot: position value vs implied probability
# ============================================================
def plot_position_value_vs_prob(bets, current_odds, label, savepath):
    """
    For a portfolio of bets at fixed odds and stakes, sweep one outcome's
    implied probability p_T1 from 0 to (1 - p_tie) and plot the expected
    position value. The slope at any point is the delta w.r.t. p_T1
    (with p_T2 absorbing the residual).
    """
    p_tie = 1.0 / current_odds["tie"] if current_odds["tie"] > 1 else 0.02

    p_t1_grid = np.linspace(0.05, 1 - p_tie - 0.05, 80)
    values = []
    for p1 in p_t1_grid:
        p2 = 1 - p_tie - p1
        probs = {"t1": p1, "t2": p2, "tie": p_tie}
        values.append(position_value(bets, probs))
    values = np.array(values)

    # Fair point (under bookmaker probs)
    raw = np.array([1/current_odds["t1"], 1/current_odds["t2"], 1/current_odds["tie"]])
    fair = raw / raw.sum()
    p1_fair = fair[0]
    v_fair = position_value(bets, {"t1": fair[0], "t2": fair[1], "tie": fair[2]})

    # Delta at the fair point (slope of the line)
    delta = position_delta(bets)
    delta_t1 = delta["t1"] - delta["t2"]  # slope w.r.t. p_t1 holding p_tie fixed

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(p_t1_grid, values, "-", c="#2c7fb8", lw=2.2,
            label="Position value V(p_T1)")
    ax.scatter([p1_fair], [v_fair], s=200, marker="*", c="#d73027",
               edgecolors="black", linewidths=1.4, zorder=10,
               label=f"Fair point (p_T1 = {p1_fair:.3f})")

    # Tangent line at the fair point
    p1_grid_tan = np.linspace(p1_fair - 0.15, p1_fair + 0.15, 30)
    tan_line = v_fair + delta_t1 * (p1_grid_tan - p1_fair)
    ax.plot(p1_grid_tan, tan_line, "--", c="#fc8d59", lw=1.8,
            label=f"Tangent (Δ = {delta_t1:+.0f} per unit Δp_T1)")

    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("Market implied probability p(T1)")
    ax.set_ylabel("Expected position value V (₹)")
    ax.set_title(f"Portfolio value vs implied prob — {label}\n"
                 f"Slope at fair point = position delta w.r.t. p(T1)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return savepath


# ============================================================
# Demo
# ============================================================
def demo_match(match, label_override=None):
    """Run delta analysis on a single match's bet ledger."""
    bets = (match.get("bets") or []) + (match.get("pre_bets") or [])
    if not bets:
        return None
    opening = match.get("opening_odds")
    if not opening or not all(opening.get(k, 0) > 1 for k in ("t1", "t2", "tie")):
        return None

    label = label_override or match.get("label", "?")
    odds = {k: float(opening[k]) for k in ("t1", "t2", "tie")}

    # Position fingerprint
    print(f"\n  ── Match: {label} ──")
    print(f"  Opening odds:  T1={odds['t1']:.2f}  T2={odds['t2']:.2f}  Tie={odds['tie']:.1f}")
    print(f"  Bets in ledger:")
    for b in bets:
        print(f"    [{b.get('outcome'):4}]  stake = ₹{float(b.get('stake', 0)):>7,.0f}  "
              f"@ odds {float(b.get('odds', 1)):.2f}  ({b.get('time_label', b.get('source','?'))})")

    # Compute fair probs (overround-adjusted)
    raw = np.array([1/odds[k] for k in ("t1", "t2", "tie")])
    fair = raw / raw.sum()
    fair_dict = {"t1": fair[0], "t2": fair[1], "tie": fair[2]}

    # P&L per state (deterministic, the discrete distribution)
    states = {"t1": 0.0, "t2": 0.0, "tie": 0.0}
    for b in bets:
        out = b.get("outcome")
        if out not in states:
            continue
        s = float(b.get("stake", 0))
        o = float(b.get("odds", 1))
        for k in states:
            if k == out:
                states[k] += s * (o - 1)
            else:
                states[k] -= s
    print(f"  P&L per state: T1=₹{states['t1']:+,.0f}  T2=₹{states['t2']:+,.0f}  Tie=₹{states['tie']:+,.0f}")

    # Position delta vector
    delta = position_delta(bets)
    print(f"  Position deltas (₹ exposure to a unit ↑ in implied prob of each outcome):")
    print(f"    Δ(T1) = {delta['t1']:>10,.1f}")
    print(f"    Δ(T2) = {delta['t2']:>10,.1f}")
    print(f"    Δ(Tie)= {delta['tie']:>10,.1f}")
    print(f"    max - min = {max(delta.values()) - min(delta.values()):>10,.1f}")

    # Expected position value under fair probs
    ev = position_value(bets, fair_dict)
    print(f"  E[V] under fair probs: ₹{ev:+,.0f}  (per unit stake = {ev/sum(float(b.get('stake',0)) for b in bets)*100:+.2f}% of total stake)")

    # Delta-neutral hedge (using opening odds as proxy for current odds)
    extra_stakes, delta_target = delta_neutralizing_stakes(delta, odds)
    print(f"  Delta-neutralizing additional stakes (at opening odds, to bring all Δ_i to {delta_target:,.0f}):")
    for k, s in extra_stakes.items():
        print(f"    s'({k}) = ₹{s:,.0f}")
    total_hedge = sum(extra_stakes.values())
    print(f"  Total hedge cost: ₹{total_hedge:,.0f}")

    return {
        "label": label,
        "opening_odds": odds,
        "bets": bets,
        "pnl_per_state": states,
        "delta": delta,
        "fair_probs": fair_dict,
        "ev_under_fair": ev,
        "delta_neutral_extra_stakes": extra_stakes,
        "delta_target": delta_target,
        "total_hedge_cost": total_hedge,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history")
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
        print("ERROR: no --history provided")
        sys.exit(1)

    data = json.loads(Path(history).read_text())
    matches = [m for m in data.get("matches", []) if m.get("status") == "settled"]

    print(f"\n{'=' * 72}")
    print(f"  E8 — Delta-hedge analogy: bets as binary options (Wk 4)")
    print(f"{'=' * 72}")

    print(f"""
  THEORETICAL FRAMING (for the report):

  A unit bet on outcome i at decimal odds o_i is structurally equivalent
  to BUYING o_i units of a binary asset-or-nothing claim:

      claim_i  pays  $1   if outcome i occurs
                     $0   otherwise

  The "fair price" of claim_i is p_i (the true probability). The bookmaker
  sells claim_i at price 1/o_i, which exceeds p_i by the overround margin.

  For a portfolio of bets, the implied-probability delta vector is

      Δ_i  =  ∂V/∂p_i  =  Σ over bets on outcome i of  (stake × odds)

  This is the dollar exposure to a unit change in the market's implied
  probability for outcome i — exactly analogous to a Black-Scholes option
  delta with the "underlying" being the implied probability rather than
  a stock price. Hedging in the betting setting becomes the search for
  a stake vector that brings the delta vector to a target (e.g., uniform
  → delta-neutral, indifferent to which way the market moves).
""")

    # Run on a few illustrative matches
    rows = []
    for m in matches:
        result = demo_match(m)
        if result is not None:
            rows.append(result)

    # Save numerical results
    out = {
        "n_matches_analyzed": len(rows),
        "matches": rows,
    }
    tblpath = TBL_DIR / "delta_hedge_results.json"
    tblpath.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  ▸ Numerical results saved to {tblpath}")

    # Plot position value for the most exposed match (largest |delta spread|)
    if rows:
        most_exposed = max(rows, key=lambda r: max(r["delta"].values()) - min(r["delta"].values()))
        figpath = FIG_DIR / f"delta_position_value_{most_exposed['label'].replace(' ','_')}.png"
        plot_position_value_vs_prob(
            most_exposed["bets"],
            most_exposed["opening_odds"],
            most_exposed["label"],
            figpath,
        )
        print(f"  ▸ Position value plot ({most_exposed['label']}): {figpath}")

    print(f"\n{'=' * 72}\n")


if __name__ == "__main__":
    main()