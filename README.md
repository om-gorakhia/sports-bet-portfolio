# IPL Bet-It | In-Play Cricket Betting Dashboard

A personal portfolio management and hedge calculation tool for IPL in-play betting. Built with Streamlit, it applies quantitative finance concepts like mean-variance optimisation, Kelly Criterion, linear programming, and Gemini AI to cricket match betting.

**Live app:** [sports-bet-portfolio.streamlit.app](https://sports-bet-portfolio.streamlit.app/)

<br>

## What this is

This is a real tool used by the author across 13 IPL matches during the 2026 season, growing a starting bank of ₹10,400 to ₹30,685, which is a net P&L of ₹20,285 (+195%) as of April 2026.

It is not a tipster app. It does not tell you who will win. What it does is help you manage the bets you have already placed by sizing hedges mathematically, tracking your full portfolio, and showing your real-time exposure across all three possible match outcomes (Team 1 Win, Team 2 Win, and Tie).

<br>

## Screenshots

> Taken from a live session. All figures are real.

### User Select | Landing Screen
![User Select](screenshots/01_user_select.png)

### Match Lobby | Capital Overview
![Match Lobby](screenshots/02_match_lobby.png)

### Active Match Session | Live Odds and P&L Matrix
![Match Dashboard](screenshots/match_dashboard.png)

### Portfolio Stats | 13 Matches, 85% Win Rate
![Portfolio](screenshots/03_portfolio.png)

### Match History | Full P&L Table
![Match History](screenshots/04_match_history.png)

<br>

## Features

### Core Betting Tools

**Live Bet Entry** lets you log bets with a time label (E1, M1, D1, etc.), odds, and stake as the match progresses.

**Pre-Match Bets** are tracked separately from in-play bets so your entry position is always clear.

**Side Bets** let you track miscellaneous bets like top scorer or fall of wicket independently from the main match position.

**The P&L Matrix** shows your real-time profit or loss for all three match outcomes at all times, updating live as you enter bets.

<br>

### Hedge Solvers

There are three hedge solving modes depending on what you are trying to do.

| Solver | What it does |
|---|---|
| Optimal Hedge | Minimises worst-case loss across all outcomes using linear programming |
| Conviction Hedge | Minimises hedge cost when you are confident in one result |
| Point of No Return | Eliminates a mathematically impossible outcome from your hedge |

<br>

### Portfolio Analytics

The portfolio tab shows an equity curve with drawdown chart across all your matches, Sharpe/Sortino/Calmar ratio calculations, VaR (Value at Risk) and Expected Shortfall at 95% confidence, match-by-match P&L breakdown, and a capital progression chart showing opening vs closing capital per match.

<br>

### Live Odds Integration

The app fetches real-time H2H odds from 10 EU bookmakers via The Odds API, computes median odds, overround, and implied probabilities per bookmaker, and logs timestamped snapshots so you can see how odds moved during the match. The polling cadence adapts to the match phase and runs more frequently during death overs.

<br>

### AI Strategist (Gemini 2.5 Flash)

The AI strategist gives specific, actionable advice that references your actual bets and capital. It identifies which bets are LOOKING GOOD, BAD, or NEUTRAL and will not give generic cricket commentary. Every output is focused on helping you make a decision.

<br>

### Kelly Criterion Sizing

Calculates the mathematically optimal stake size given your edge and bank, with half-Kelly and quarter-Kelly options for more conservative sizing.

<br>

### Arbitrage Detection

Flags when the combined implied probabilities across bookmakers fall below 100% and shows the guaranteed profit amount if arbitrage is available.

<br>

### Excel Export

Exports your full portfolio to a formatted .xlsx workbook including a bet-by-bet breakdown, settlement history, and aggregate stats.

<br>

## My data (Om Gorakhia)

The repository includes my real betting history in `user_om_gorakhia.json`. When you open the app, select **om_gorakhia** from the user list to see the full picture.

| Stat | Value |
|---|---|
| Starting bank | ₹10,400 |
| Current capital | ₹30,685.98 |
| Net P&L | +₹20,285.98 |
| Return | +195% |
| Matches tracked | 13 |
| Season target | 300% (3x) |

The match history includes bets across RR vs MI, MI vs KKR, CSK vs RCB, and more.

<br>

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Charts | Plotly |
| Optimisation | SciPy (linprog) |
| Numerical | NumPy |
| AI | Google Gemini 2.5 Flash |
| Odds data | The Odds API v4 |
| Cricket data | CricAPI |
| Export | OpenPyXL |
| Runtime | Python 3.13 |

<br>

## Quantitative research modules

The `research/` folder contains standalone scripts that explore the mathematics behind the hedging algorithms.

| Module | Topic |
|---|---|
| `01_mean_variance_hedge.py` | Markowitz portfolio optimisation and efficient frontier |
| `02_risk_adjusted_metrics.py` | Sharpe, Sortino, Calmar ratios on betting equity curve |
| `03_var_es.py` | Value-at-Risk and Expected Shortfall |
| `04_shrinkage.py` | Covariance matrix shrinkage for small sample sizes |
| `05_curse_of_dimensionality.py` | Why the 3-outcome bet problem is well-behaved |
| `06_delta_hedge_analogy.py` | Options Greeks analogy applied to betting positions |
| `07_loocv_backtest.py` | Leave-One-Out Cross-Validation backtesting framework |

<br>

## Setup

### 1. Clone and install

```bash
git clone https://github.com/om-gorakhia/sports-bet-portfolio.git
cd sports-bet-portfolio
pip install -r requirements.txt
```

### 2. API keys (optional, the app works without them)

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Key | Where to get it | Required for |
|---|---|---|
| `ODDS_API_KEY` | [the-odds-api.com](https://the-odds-api.com) | Live odds fetching |
| `CRICKET_DATA_API_KEY` | [cricapi.com](https://cricapi.com) | Live match state |

The Gemini API key is entered inside the app on the Settings page and is stored only in your local user file. You can also set it as an environment variable if you prefer.

### 3. Run

```bash
streamlit run ipl_betting_dashboard.py
```

The app opens at `http://localhost:8501`.

<br>

## Live odds capture (background service)

`data_feed.py` is a separate background service that polls The Odds API and saves snapshots to the `captures/` folder. Run it alongside the dashboard for real-time odds updates.

```bash
# Capture odds for a specific match
python data_feed.py --match "Mumbai" --duration 14400

# Auto-pick the next live IPL match
python data_feed.py --auto --duration 14400

# Test your API connection without writing any files
python data_feed.py --match "Mumbai" --dry-run
```

Captured files are stored as `captures/YYYY-MM-DD__team1__vs__team2.json` and the dashboard reads them automatically.

<br>

## Data feed configuration

Set these environment variables before running `data_feed.py`:

```bash
export ODDS_API_KEY="your_key"
export CRICKET_DATA_API_KEY="your_key"
```

The Odds API free plan gives 500 requests per month. The poller is adaptive in that it slows down pre-match and speeds up during death overs, and stops automatically when fewer than 30 credits remain.

<br>

## Project structure

```
sports-bet-portfolio/
├── ipl_betting_dashboard.py   # Main Streamlit app (6,200+ lines)
├── data_feed.py               # Background odds capture service
├── requirements.txt
├── .env.example               # Template for API keys
├── .streamlit/
│   └── config.toml            # Dark theme config
├── user_om_gorakhia.json      # Real betting history (Gemini key removed)
├── users_index.json           # User registry
├── captures/                  # Odds snapshots (JSON, timestamped)
│   └── 2026-04-07__rajasthan_royals__vs__mumbai_indians.json
└── research/                  # Quant research scripts
    ├── 01_mean_variance_hedge.py
    ├── 02_risk_adjusted_metrics.py
    ├── 03_var_es.py
    ├── 04_shrinkage.py
    ├── 05_curse_of_dimensionality.py
    ├── 06_delta_hedge_analogy.py
    └── 07_loocv_backtest.py
```

<br>

## How hedging works (plain English)

When you back Team 1 to win and the odds then shift in your favour, you are sitting on an unrealised profit. A hedge is a bet on the other outcome or outcomes that locks in some of that profit regardless of the final result.

The three solvers in this app each answer a different question.

The **Optimal hedge** asks what stakes on Team 2 and Tie give you the highest guaranteed floor, and uses SciPy linprog to solve that as a linear programme.

The **Conviction hedge** asks what the cheapest hedge is that eliminates ruin if you are wrong, for when you still think Team 1 wins.

The **Point of No Return** asks whether you should hedge the Tie at all when Team 2 is already mathematically eliminated at the current score.

The P&L matrix updates live as you enter bets so you always see exactly what you gain or lose in each scenario.

<br>

## Disclaimer

This is a personal project for educational and analytical purposes. Betting involves real financial risk. Nothing in this repository constitutes financial or betting advice. Use at your own discretion.
