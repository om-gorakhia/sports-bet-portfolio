#!/usr/bin/env python3
"""
data_feed.py — Live IPL odds + match-state capture for QRM project.

Captures odds snapshots from The Odds API (10 EU bookmakers) plus optional
live match state from CricketData.org, and writes them to a per-match JSON
file in ./captures/. Adaptive polling cadence by match phase.

USAGE
-----
    # Set credentials (preferred — keep keys out of source):
    export ODDS_API_KEY="your_odds_api_key"
    export CRICKET_DATA_API_KEY="your_cricketdata_org_key"   # optional but recommended

    # Discovery dry-run (1 API call, prints what's available):
    python data_feed.py --match "Mumbai" --dry-run

    # Live capture for a 4-hour window:
    python data_feed.py --match "Mumbai" --duration 14400

    # Capture any IPL match (picks the first live or upcoming one):
    python data_feed.py --auto --duration 14400

NOTES
-----
- Each /odds API call costs 1 credit. Free plan = 500/month.
- The script auto-stops when remaining credits drop below QUOTA_HARD_FLOOR (30).
- Adaptive cadence: polls more often during high-volatility phases (death overs)
  and less often during pre-match / innings break.
- Ctrl-C is handled cleanly — current state is saved before exit.
- Output schema is designed to plug into the existing dashboard's odds_snapshots
  format with minimal transformation.
"""

import os
import sys
import json
import time
import signal
import argparse
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
import urllib.request
import urllib.parse
import urllib.error

# ============================================================
# Config
# ============================================================
SGT = timezone(timedelta(hours=8))
IST = timezone(timedelta(hours=5, minutes=30))

# Credentials — env vars take priority, hardcoded fallbacks for convenience.
# WARNING: Both keys below are exposed in this source file. ROTATE BOTH KEYS
# AFTER THE PROJECT IS SUBMITTED. Do not commit this file to a public repo.
ODDS_API_KEY = os.environ.get("ODDS_API_KEY") or ""
CRICKET_API_KEY = os.environ.get("CRICKET_DATA_API_KEY") or ""

ODDS_BASE = "https://api.the-odds-api.com/v4"
CRICKET_BASE = "https://api.cricapi.com/v1"

CAPTURES_DIR = Path(__file__).parent / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)

# Stop polling when fewer than this many credits remain — preserves a buffer
# so you don't burn the entire monthly quota on one match.
QUOTA_HARD_FLOOR = 30

# Adaptive cadence (seconds between polls) by match phase.
# More frequent polling = more snapshots = better data for GARCH/EWMA later.
CADENCE_SECONDS = {
    "pre_match":       30 * 60,   # 30 min — odds drift slowly before toss
    "innings_1_pp":    10 * 60,   # 10 min — powerplay (overs 1-6)
    "innings_1_mid":   10 * 60,   # 10 min — middle overs
    "innings_1_death":  5 * 60,   #  5 min — death overs (15-20)
    "innings_break":   15 * 60,   # 15 min — long quiet period
    "innings_2_pp":    10 * 60,
    "innings_2_mid":    8 * 60,
    "innings_2_death":  2 * 60,   #  2 min — peak volatility, the data we want most
    "post_match":           0,    # 0 = stop polling
    "unknown":          5 * 60,   # fallback
}


# ============================================================
# HTTP helpers
# ============================================================
def http_get_json(url, params=None, timeout=15):
    """GET a JSON endpoint. Returns (data, response_headers_dict)."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "qrm-research/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return json.loads(body), headers
    except urllib.error.HTTPError as e:
        return None, {"error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return None, {"error": str(e)}


# ============================================================
# The Odds API
# ============================================================
def fetch_ipl_odds():
    """Fetch all live + upcoming IPL matches with EU bookmaker h2h odds."""
    data, headers = http_get_json(
        f"{ODDS_BASE}/sports/cricket_ipl/odds/",
        params={
            "apiKey": ODDS_API_KEY,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
        },
    )
    quota_remaining = -1
    if data is not None and "x-requests-remaining" in headers:
        try:
            quota_remaining = int(headers["x-requests-remaining"])
        except (ValueError, TypeError):
            pass
    return data, quota_remaining, headers


def find_match(matches, search_team):
    """Return the first match where home or away contains search_team (case-insensitive)."""
    if not matches:
        return None
    s = search_team.lower()
    for m in matches:
        if s in m.get("home_team", "").lower() or s in m.get("away_team", "").lower():
            return m
    return None


def pick_first_active(matches):
    """Pick the chronologically nearest live or upcoming match."""
    if not matches:
        return None
    return sorted(matches, key=lambda m: m.get("commence_time", ""))[0]


def extract_odds_summary(match):
    """
    Reduce a match's bookmaker list into:
      - median odds across books for each outcome (robust point estimate)
      - dispersion (stdev) — proxy for market disagreement
      - implied probabilities and overround
      - per-book raw odds (preserved for later cross-book research)
    Tie outcome is synthesized at constant 50.0 to keep the dashboard's
    3-state {t1, t2, tie} schema intact (T20 ties are <0.5% historically).
    """
    home = match["home_team"]
    away = match["away_team"]
    home_prices = []
    away_prices = []
    per_book = {}

    for bm in match.get("bookmakers", []):
        for mk in bm.get("markets", []):
            if mk.get("key") != "h2h":
                continue
            outcomes_dict = {}
            for o in mk.get("outcomes", []):
                price = o.get("price")
                name = o.get("name")
                if price and price > 1:
                    if name == home:
                        home_prices.append(price)
                    elif name == away:
                        away_prices.append(price)
                    outcomes_dict[name] = price
            per_book[bm["key"]] = {
                "title": bm.get("title"),
                "last_update": bm.get("last_update"),
                "outcomes": outcomes_dict,
            }

    if not home_prices or not away_prices:
        return None

    home_med = statistics.median(home_prices)
    away_med = statistics.median(away_prices)
    home_std = statistics.stdev(home_prices) if len(home_prices) > 1 else 0.0
    away_std = statistics.stdev(away_prices) if len(away_prices) > 1 else 0.0

    home_imp = 100.0 / home_med
    away_imp = 100.0 / away_med
    overround = home_imp + away_imp - 100.0

    tie_synthetic = 50.0

    return {
        "median_odds": {
            "t1": round(home_med, 3),    # home_team is mapped to t1
            "t2": round(away_med, 3),    # away_team is mapped to t2
            "tie": tie_synthetic,
        },
        "implied_prob_pct": {
            "t1": round(home_imp, 2),
            "t2": round(away_imp, 2),
            "tie": round(100.0 / tie_synthetic, 2),
        },
        "overround_pct": round(overround, 3),
        "dispersion_std": {
            "t1": round(home_std, 4),
            "t2": round(away_std, 4),
        },
        "n_books": len(home_prices),
        "per_book": per_book,
    }


# ============================================================
# CricketData.org (live match state)
# ============================================================
def fetch_match_state(home, away):
    """
    Look up live state for a match by team names.
    Returns None if no key configured or match not found in currentMatches.
    """
    if not CRICKET_API_KEY:
        return None
    data, _ = http_get_json(
        f"{CRICKET_BASE}/currentMatches",
        params={"apikey": CRICKET_API_KEY, "offset": 0},
    )
    if not data or not isinstance(data, dict) or "data" not in data:
        return None

    h_low = home.lower()
    a_low = away.lower()

    def team_match(team_str):
        t = team_str.lower()
        # Loose match: any team substring overlap
        return (h_low in t or t in h_low or any(w in t for w in h_low.split())) or \
               (a_low in t or t in a_low or any(w in t for w in a_low.split()))

    for m in data["data"]:
        teams = m.get("teams", []) or []
        # Need BOTH teams to appear
        h_seen = any((h_low in t.lower() or t.lower() in h_low) for t in teams)
        a_seen = any((a_low in t.lower() or t.lower() in a_low) for t in teams)
        if h_seen and a_seen:
            return {
                "id": m.get("id"),
                "name": m.get("name"),
                "status": m.get("status"),
                "match_type": m.get("matchType"),
                "venue": m.get("venue"),
                "score": m.get("score", []),
                "match_started": m.get("matchStarted"),
                "match_ended": m.get("matchEnded"),
            }
    return None


# ============================================================
# Phase classification
# ============================================================
def classify_phase(commence_iso, match_state):
    """
    Decide which phase we're in. Uses live score (over count, innings number)
    if available, otherwise falls back to elapsed-time-since-commence heuristic.
    """
    now = datetime.now(timezone.utc)
    commence = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))

    if now < commence:
        return "pre_match"

    if match_state and match_state.get("match_ended"):
        return "post_match"

    # Try to use live score first
    if match_state and match_state.get("score"):
        scores = match_state["score"]
        if scores:
            last = scores[-1]
            try:
                overs = float(last.get("o", 0) or 0)
            except (ValueError, TypeError):
                overs = 0
            inning_num = len(scores)

            if inning_num == 1:
                if overs <= 6:
                    return "innings_1_pp"
                if overs >= 15:
                    return "innings_1_death"
                return "innings_1_mid"
            if inning_num >= 2:
                if overs <= 6:
                    return "innings_2_pp"
                if overs >= 15:
                    return "innings_2_death"
                return "innings_2_mid"

    # Clock-only fallback
    elapsed_min = (now - commence).total_seconds() / 60
    if elapsed_min < 90:
        return "innings_1_mid"
    if elapsed_min < 110:
        return "innings_break"
    if elapsed_min < 200:
        return "innings_2_mid"
    return "post_match"


# ============================================================
# Capture file IO
# ============================================================
def slug(s):
    return "".join(c if c.isalnum() else "_" for c in s.lower()).strip("_")


def get_capture_path(match):
    home_s = slug(match["home_team"])
    away_s = slug(match["away_team"])
    date_s = match["commence_time"][:10]
    return CAPTURES_DIR / f"{date_s}__{home_s}__vs__{away_s}.json"


def load_or_init_capture(path, match):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"[!] Existing capture file at {path} is corrupt — starting fresh.")
    return {
        "match_id": match["id"],
        "sport_key": match["sport_key"],
        "home_team": match["home_team"],
        "away_team": match["away_team"],
        "commence_time": match["commence_time"],
        "first_capture_at": datetime.now(timezone.utc).isoformat(),
        "snapshots": [],
        "quota_used_in_session": 0,
        "last_quota_remaining": None,
    }


def append_snapshot(capture, odds_summary, phase, match_state):
    now = datetime.now(timezone.utc)
    snap = {
        "ts_utc": now.isoformat(),
        "ts_sgt": now.astimezone(SGT).strftime("%Y-%m-%d %H:%M:%S"),
        "ts_ist": now.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "phase": phase,
        "median_odds": odds_summary["median_odds"],
        "implied_prob_pct": odds_summary["implied_prob_pct"],
        "overround_pct": odds_summary["overround_pct"],
        "dispersion_std": odds_summary["dispersion_std"],
        "n_books": odds_summary["n_books"],
        "per_book": odds_summary["per_book"],
        "match_state": match_state,
    }
    capture["snapshots"].append(snap)


def save_capture(path, capture):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(capture, indent=2))
    tmp.replace(path)


# ============================================================
# Main loop
# ============================================================
RUNNING = True


def handle_sigint(sig, frame):
    global RUNNING
    print("\n[!] Caught Ctrl-C — saving and exiting cleanly...")
    RUNNING = False


def fmt_odds(o):
    return f"T1={o['t1']:.3f}  T2={o['t2']:.3f}"


def run(target_team, max_duration_sec, dry_run, auto):
    signal.signal(signal.SIGINT, handle_sigint)

    print(f"\n{'='*60}")
    print(f"  IPL DATA FEED — QRM RESEARCH")
    print(f"{'='*60}")
    print(f"  Started:  {datetime.now(SGT).strftime('%Y-%m-%d %H:%M:%S SGT')}")
    print(f"  Mode:     {'AUTO (first match)' if auto else f'TARGET: {target_team!r}'}")
    print(f"  Duration: max {max_duration_sec/60:.0f} minutes")
    print(f"  Captures: {CAPTURES_DIR}")
    print(f"  Cricket state: {'ON' if CRICKET_API_KEY else 'OFF (no CRICKET_DATA_API_KEY)'}")
    if dry_run:
        print(f"  *** DRY RUN — 1 discovery call only ***")
    print(f"{'='*60}\n")

    # Step 1: discover the target match
    print("[1] Fetching IPL match list from The Odds API...")
    matches, quota, headers = fetch_ipl_odds()
    if matches is None:
        print(f"[X] FAILED. Headers: {headers}")
        return

    print(f"    {len(matches)} match(es) returned. Quota remaining: {quota}")
    if not matches:
        print("[X] No IPL matches in the feed right now (off-day or schedule gap).")
        return

    print(f"\n    Available matches:")
    for m in matches:
        ct = m["commence_time"]
        ct_sgt = datetime.fromisoformat(ct.replace("Z", "+00:00")).astimezone(SGT)
        print(f"      - {m['home_team']:25} vs {m['away_team']:25}  {ct_sgt.strftime('%Y-%m-%d %H:%M SGT')}  ({len(m.get('bookmakers',[]))} books)")

    target = pick_first_active(matches) if auto else find_match(matches, target_team)
    if not target:
        print(f"\n[X] No match found containing '{target_team}'.")
        return

    commence_sgt = datetime.fromisoformat(target['commence_time'].replace('Z', '+00:00')).astimezone(SGT)
    print(f"\n[+] SELECTED: {target['home_team']} (T1) vs {target['away_team']} (T2)")
    print(f"    Commence:   {commence_sgt.strftime('%Y-%m-%d %H:%M:%S SGT')}")
    print(f"    Bookmakers: {len(target.get('bookmakers', []))}")

    capture_path = get_capture_path(target)
    capture = load_or_init_capture(capture_path, target)
    print(f"    Capture:    {capture_path}")
    print(f"    Existing snapshots in file: {len(capture['snapshots'])}")

    # Step 2: take the discovery snapshot immediately (it's already paid for)
    print(f"\n[2] Taking initial snapshot from discovery call...")
    summary = extract_odds_summary(target)
    if not summary:
        print("[X] No usable odds in initial fetch. Exiting.")
        return

    state = fetch_match_state(target['home_team'], target['away_team'])
    phase = classify_phase(target['commence_time'], state)
    append_snapshot(capture, summary, phase, state)
    capture["quota_used_in_session"] += 1
    capture["last_quota_remaining"] = quota
    save_capture(capture_path, capture)

    print(f"    Phase:       {phase}")
    print(f"    Median odds: {fmt_odds(summary['median_odds'])}")
    print(f"    Overround:   {summary['overround_pct']:+.2f}%")
    print(f"    Books used:  {summary['n_books']}")
    print(f"    Snapshots in file now: {len(capture['snapshots'])}")

    if dry_run:
        print(f"\n[*] Dry run complete. Capture saved.")
        return

    # Step 3: main polling loop
    started_at = time.time()
    cadence = CADENCE_SECONDS.get(phase, 600)
    next_poll_at = time.time() + cadence

    print(f"\n[3] Entering live polling loop. Press Ctrl-C to stop & save.")
    print(f"    Next poll in {cadence}s (cadence for phase '{phase}').\n")

    while RUNNING:
        now_t = time.time()

        if now_t - started_at > max_duration_sec:
            print(f"[!] Max duration reached ({max_duration_sec/60:.0f} min). Exiting.")
            break

        if now_t < next_poll_at:
            time.sleep(min(5, next_poll_at - now_t))
            continue

        # POLL
        ts = datetime.now(SGT).strftime("%H:%M:%S SGT")
        print(f"[{ts}] Polling...")
        matches, quota, headers = fetch_ipl_odds()

        if quota >= 0 and quota < QUOTA_HARD_FLOOR:
            print(f"[!] QUOTA LOW ({quota} credits remain, floor={QUOTA_HARD_FLOOR}). Stopping.")
            break

        if matches is None:
            print(f"[X] Fetch failed: {headers}. Retrying in 60s.")
            next_poll_at = time.time() + 60
            continue

        target = find_match(matches, target_team) if not auto else \
                 next((m for m in matches if m["id"] == capture["match_id"]), None)
        if not target:
            print(f"[!] Target match no longer in feed — likely ended. Exiting cleanly.")
            break

        summary = extract_odds_summary(target)
        if not summary:
            print(f"[!] No usable odds in this poll. Backing off.")
            next_poll_at = time.time() + CADENCE_SECONDS["unknown"]
            continue

        state = fetch_match_state(target['home_team'], target['away_team'])
        phase = classify_phase(target['commence_time'], state)
        append_snapshot(capture, summary, phase, state)
        capture["quota_used_in_session"] += 1
        capture["last_quota_remaining"] = quota
        save_capture(capture_path, capture)

        print(f"    Snap #{len(capture['snapshots'])}  phase={phase}  "
              f"{fmt_odds(summary['median_odds'])}  "
              f"OR={summary['overround_pct']:+.2f}%  "
              f"books={summary['n_books']}  quota={quota}")
        if state and state.get("score"):
            try:
                last_score = state['score'][-1]
                print(f"    Score: {last_score.get('inning','?')}: "
                      f"{last_score.get('r','?')}/{last_score.get('w','?')} "
                      f"({last_score.get('o','?')} ov)")
            except (IndexError, KeyError, TypeError):
                pass

        cadence = CADENCE_SECONDS.get(phase, 600)
        if cadence == 0:
            print(f"[!] Phase '{phase}' → cadence 0 → done.")
            break
        next_poll_at = time.time() + cadence
        print(f"    Next poll in {cadence}s\n")

    # Final save
    save_capture(capture_path, capture)
    print(f"\n{'='*60}")
    print(f"  SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total snapshots in file: {len(capture['snapshots'])}")
    print(f"  This session added:      {capture['quota_used_in_session']}")
    print(f"  Quota remaining:         {capture.get('last_quota_remaining', '?')}")
    print(f"  File:                    {capture_path}")
    print(f"{'='*60}\n")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="IPL odds + state capture for QRM project.")
    p.add_argument("--match", default="Mumbai",
                   help="Team-name substring to find target match (default: 'Mumbai')")
    p.add_argument("--auto", action="store_true",
                   help="Auto-pick the first live or upcoming match (ignores --match)")
    p.add_argument("--duration", type=int, default=14400,
                   help="Max run duration in seconds (default 14400 = 4hr)")
    p.add_argument("--dry-run", action="store_true",
                   help="Make 1 discovery call, save the snapshot, exit")
    args = p.parse_args()

    if not ODDS_API_KEY:
        print("ERROR: ODDS_API_KEY not set (env var or fallback).", file=sys.stderr)
        sys.exit(1)

    if ODDS_API_KEY == "ae69ee92e01205d6df9221986d9ca548":
        print("WARNING: Using the API key that was pasted into chat. ROTATE IT.")
        print("         Then: export ODDS_API_KEY=\"<new_key>\"\n")

    run(
        target_team=args.match,
        max_duration_sec=args.duration,
        dry_run=args.dry_run,
        auto=args.auto,
    )