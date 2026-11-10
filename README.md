# ATP Edge

Men's singles betting model for every **ATP 250, ATP 500, Masters 1000, Grand Slam and the ATP Finals**:
- win probabilities and fair odds for each match;
- value bets found by comparing every bookmaker's price with the fair price;
- **Grand Slam over/under 3.5 sets**;
- fully automated with GitHub Actions: data refresh, retraining, predictions, bet logging and settlement.

```
pip install -r requirements.txt
streamlit run app/streamlit_app.py      # the app (sidebar → "Demo data" works without keys)
python scripts/live_loop.py --once      # one refresh: prices, predictions, bet log  (needs keys)
python scripts/update_data.py           # refresh results, rankings, court speed
python scripts/train.py                 # rebuild ratings + retrain all models (~2 min)
pytest -q                               # 47 tests
```

## Set it up (about 15 minutes)

1. **Create a new GitHub repository** and upload everything in this folder. That includes the hidden folders
   `.github/` and `.streamlit/`; on a Mac, press Cmd+Shift+. in Finder to see them.
2. **Add your API keys as secrets.** Never put keys in the code or in a chat. In the repo, go to **Settings → Secrets and variables → Actions → New repository secret**:
   - `RAPIDAPI_KEY`: your RapidAPI key, subscribed to *Tennis API – ATP WTA ITF* (Matchstat)
   - `ODDS_API_KEY`: your key from the-odds-api.com
3. **Let the workflows write to the repo:** **Settings → Actions → General → Workflow permissions →
   Read and write**.
4. **Run each workflow once by hand:** **Actions → "Daily data refresh…" → Run workflow**, then **Actions → "Live prices…" → Run workflow**.
   After that they run by themselves.
5. **Deploy the app** on share.streamlit.io: choose the repo, set the main file to `app/streamlit_app.py`, and paste
   both keys into the app's **Secrets** box (format in `.streamlit/secrets.toml.example`).

### What runs automatically

| Workflow | When | What it does |
|---|---|---|
| `slate.yml` (Live prices) | Always on: a refresh every 10 minutes | Fixtures (Matchstat) and prices (The Odds API, Matchstat as fallback); predicts every match in the next 48 hours; logs new bets the moment a price passes the rule; tracks each logged bet's price until the start; settles finished bets; optional phone alert per new bet |
| `daily.yml` | Once a day (08:20 UTC, often later) | New results (TennisMyLife, Matchstat), rankings, court speed (Tennis Abstract) and closing odds; rebuilds ratings and retrains every model; runs the tests. About 5–10 minutes |

**How prices stay current.** GitHub's timer is unreliable (an "hourly" job really ran 3–4 times a day), so the live
job doesn't depend on it: each run refreshes every 10 minutes for about 50 minutes and then starts the next run itself.
The hourly schedule is only a backup that restarts it if it ever stops. When nothing starts within 24 hours (off-season),
it stops after one check, and the backup schedule looks again an hour later. To pause it: **Actions → Live prices → ⋯ → Disable workflow**.

Not every price is pulled every 10 minutes. Each match is refreshed by how close it is to the start:

| Time to start | Prices refreshed every |
|---|---|
| under 3 hours | 10 minutes (`PRICE_INTERVAL_MINUTES`) |
| 3–12 hours | 30 minutes |
| more than 12 hours | 2 hours |

**Quotas look after themselves.** The Odds API costs 1 credit per tournament per pull (2 during a Slam, for the sets
market); at this schedule that's typically 5,000–10,000 credits a month. If fewer than 1,500 credits remain, pulls
slow to a third; under 300 they stop. Matchstat (RapidAPI) calls are budgeted from the plan usage RapidAPI reports with
every response: what's left is spread evenly until the plan resets, keeping a reserve for the daily job.

A logged bet always keeps the price at which it was first flagged (that's what results and CLV use). The **Results**
page also shows each open bet's best price now, the move since it was flagged, and whether it's still a bet.

**Phone alerts (optional).** Install the free **ntfy** app, subscribe to a long random topic name (e.g.
`atp-edge-7f3k29xq`), and add the same name as a repo secret `NTFY_TOPIC`. Each new bet then arrives as a
notification with the price, book, EV and stake. A Discord channel webhook works too (`DISCORD_WEBHOOK_URL`).

**GitHub Actions minutes:** the live job runs most of the day. That's free and unlimited on a public repository; on a
private one it would use far more than the free 2,000 minutes a month.

The app checks for new prices every 2 minutes and reloads itself when the live job publishes them.

## How a prediction is made

1. **One rating engine for training and live use** (`src/atp_edge/engine.py`). It replays every tour-level **and
   Challenger** match since 2000 in date order (about 200,000 matches). Challenger results only feed the ratings and form
   (with a smaller rating step); the model is trained and bets only on tour-level matches. Training features and tomorrow's
   features come from the same code, so the two can't drift apart.
2. **Features:**
   - **Ratings:** dynamic-K Elo, overall, per surface, and indoor vs outdoor hard court, weighted by margin of victory.
   - **Serve:** surface-specific service points won, opponent-adjusted service points won, ace and double-fault rates.
   - **Return:** surface-specific return points won and opponent-adjusted return points won.
   - **Form:** last-10 results, Elo trend, opponent strength.
   - **Fatigue:** matches, sets and minutes played in the last 7 to 14 days, and rest days.
   - **Injuries:** see the injuries section below.
   - **History at this event:** the player's results at this tournament compared with what his rating predicted, shrunk for small samples.
   - **Event type:** the same comparison per level (Slam, Masters, 500, 250, Finals) and for best-of-five.
   - **Court speed:** the court speed index for this event, the player's measured preference for fast or slow courts, and speed × serve, return and aces.
   - **Head to head:** shrunk toward neutral for small samples.
   - **Rank, age and height.**
3. **Model:** a logistic regression, chosen over gradient boosting on the last three seasons. It's trained symmetrically,
   so P(A beats B) = 1 − P(B beats A) exactly.
4. **Fair price = the model's probability** (after any manual injury flag).
5. **Bet:** a bet is signalled when, at the best price among your books, **EV ≥ 5% and the model is at least 5 points
   above the market's no-vig probability** (Pinnacle, or the books' consensus when Pinnacle is missing). Stake is
   ¼-Kelly, capped at 2% of bankroll (1.5% for set totals). Change any of these without touching code by adding repo
   **variables**: `MIN_EV`, `MIN_EDGE`, `MIN_EV_SETS`, `KELLY_FRACTION`, `KELLY_CAP`, and `BET_BOOKS` (comma-separated
   book keys you can actually use, e.g. `pinnacle,bet365,draftkings,fanduel`).

### Court speed index
The primary source is the **Tennis Abstract surface-speed ratings**: ace rate adjusted for the servers and returners,
covering every tour event since 2000, refreshed every day. 1.00 is tour average and 1.25 means 25% more aces.
A match is only ever given ratings from earlier seasons. During an event, the rating is also updated from the aces in matches already played that week.
If Tennis Abstract has no rating for an event, the model estimates one from its own match statistics.

### Injuries
- **Learned automatically from match history:** retirements and walkovers (recent ones count more), number of retirements in the last 12 months, layoff length, and the first few matches of a comeback.
- **Your own flags:** no free API reports tennis injuries reliably, so news goes in `data/reference/injuries.csv`.
  Use a `severity` from 0 to 1 and an optional `until` date. Severity 0.5 moves a 50/50 match to roughly 35/65.
  Edit the file on GitHub; the next run picks it up.

### Grand Slam over/under 3.5 sets
1. The match probability comes from the Pinnacle moneyline when available, otherwise from the model.
2. It's converted to the chance of winning a single set.
3. That gives P(4+ sets) assuming sets are independent.
4. A **learned correction** adjusts it for serve strength (tiebreaks), serve-vs-return mismatch, court speed,
   both players' history of long Slam matches, best-of-five experience, injury signals and round.

Walk-forward over 4,760 Slam matches (2017–2026), log loss (lower is better):

| Method | Log loss |
|---|---|
| Model | **0.668** |
| Always predicting the base rate | 0.693 |
| Naive independent-sets calculation | 0.702 |

Prices for the sets market come from The Odds API's `totals` market: in a best-of-five match, a 3.5 line can only
be sets. Not every book offers it. When none do, use the price checker on the Grand Slam page.

## Honest performance

Walk-forward, 2018–2026: each season is predicted by a model trained only on earlier seasons and compared
with Pinnacle closing odds on 18,637 matches.

| | Log loss (lower is better) | Accuracy |
|---|---|---|
| Model (v1.1: Challenger history, indoor ratings, opponent-adjusted serve/return) | 0.598 | 66.8% |
| Model (v1.0) | 0.604 | 66.3% |
| Pinnacle closing price | **0.586** | **68.0%** |

| Bet rule, replayed at Pinnacle *closing* prices | Bets | Actual ROI |
|---|---|---|
| EV ≥ 2%, edge ≥ 2% | 12,548 (67% of matches) | −2.5% ± 1.1% |
| **EV ≥ 5%, edge ≥ 5% (default)** | 7,833 (42%) | **−0.3% ± 1.4%** |
| EV ≥ 8%, edge ≥ 8% | 4,384 (24%) | −2.7% ± 1.7% |

At closing prices the default rule is now roughly break-even (it was −2.8% in v1.0). Closing prices are the hardest
test: earlier prices and the best price across several books are better than the Pinnacle close, so live results can
be better, or worse. The live bet log and its CLV are the real test.

## Results & closing-line value (CLV)

Only the bets the model signals are tracked, each with the ¼-Kelly stake it was given:
- **Logged before the match**, at the price and book that triggered the signal (never overwritten later).
- **Settled** the next morning from official results: won / lost, profit in % of bankroll and CA$.
- **CLV:** price taken × closing no-vig probability − 1. The close is the last Pinnacle snapshot (taken every 10 minutes near the start) before the
  start (or the books' consensus if Pinnacle has none). Positive CLV over a few hundred bets is the earliest reliable sign
  of an edge; the page gives an "edge / no edge / too early" verdict using a two-standard-error rule.

## Match previews (optional AI)

Every match page has a written preview built from the model's numbers. Add a repo secret `ANTHROPIC_API_KEY`
(from console.anthropic.com) and the live job also asks Claude Haiku to write a 3–4 sentence preview from the same
facts. Previews are cached and only rewritten when the numbers move; cost is well under a cent per match.

## Things to check after your first live run

These parts couldn't be tested without your keys:
- **Matchstat results endpoint:** the repo tries fixtures-by-date first and falls back to recent player histories.
  `data/generated/data_status.json` shows which one worked.
- **Which tournaments The Odds API carries,** and whether it offers sets totals for Slams. The board's *Data sources* panel lists them.

## Layout

```
app/                      Streamlit app (board, Grand Slam sets, match detail, results & CLV, model & data)
src/atp_edge/
  tournaments.py          registry of every 250/500/1000/Slam/Finals event: level, surface, indoor, aliases
  data.py                 match database + cleaning (duplicates, ITF/Challenger, qualifying, level codes)
  court_speed.py          Tennis Abstract court speed index + in-event estimate
  engine.py               ratings and all features (one code path for training and live)
  modeling.py             match model, walk-forward benchmark, market blend, Grand Slam sets model
  betting.py              no-vig, EV, Kelly, best-of-5 set maths, backtests
  rapidapi.py             Matchstat (RapidAPI) client
  odds_api.py             The Odds API client
  predict.py              predictions and recommendations
  live.py                 price scheduler (refresh by time to start, quota budgets)
  tracking.py             prediction & bet logs, line moves, settlement, CLV
  notify.py               optional phone alerts (ntfy / Discord)
  injuries.py             manual injury flags
scripts/                  live_loop.py (live prices), update_data.py + train.py (daily), run_slate.py (one-off / demo)
data/seed                 match archive 2000–2026 (bootstrap; refreshed daily)
data/reference            court speed table, injury flags
data/odds                 historical closing odds (tennis-data.co.uk, evaluation only)
data/predictions          latest slate, prediction log, bet log (written by Actions)
```

Data sources: TennisMyLife match archive, Matchstat via RapidAPI, The Odds API, Tennis Abstract surface-speed
ratings (Jeff Sackmann), and tennis-data.co.uk closing odds. Check each provider's terms before commercial use.
This tool estimates probabilities and doesn't guarantee profit.
