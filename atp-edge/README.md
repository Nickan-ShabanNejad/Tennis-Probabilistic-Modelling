# ATP Edge

Men's singles betting model for every **ATP 250, ATP 500, Masters 1000, Grand Slam and the ATP Finals**:
- win probabilities and fair odds for each match;
- value bets found by comparing every bookmaker's price with the fair price;
- **Grand Slam over/under 3.5 sets**;
- fully automated with GitHub Actions: data refresh, retraining, predictions, bet logging and settlement.

```
pip install -r requirements.txt
streamlit run app/streamlit_app.py      # the app (sidebar → "Demo data" works without keys)
python scripts/run_slate.py             # predict upcoming matches + log bets   (needs keys)
python scripts/update_data.py           # refresh results, rankings, court speed; settle bets
python scripts/train.py                 # rebuild ratings + retrain all models (~3 min)
pytest -q                               # 31 tests
```

## Set it up (about 15 minutes)

1. **Create a new GitHub repository** and upload everything in this folder. That includes the hidden folders
   `.github/` and `.streamlit/`; on a Mac, press Cmd+Shift+. in Finder to see them.
2. **Add your API keys as secrets.** Never put keys in the code or in a chat. In the repo, go to **Settings → Secrets and variables → Actions → New repository secret**:
   - `RAPIDAPI_KEY`: your RapidAPI key, subscribed to *Tennis API – ATP WTA ITF* (Matchstat)
   - `ODDS_API_KEY`: your key from the-odds-api.com
3. **Let the workflows write to the repo:** **Settings → Actions → General → Workflow permissions →
   Read and write**.
4. **Run each workflow once by hand:** **Actions → "Daily data refresh…" → Run workflow**, then **Actions → "Live slate…" → Run workflow**.
   After that they run on their own schedule.
5. **Deploy the app** on share.streamlit.io: choose the repo, set the main file to `app/streamlit_app.py`, and paste
   both keys into the app's **Secrets** box (format in `.streamlit/secrets.toml.example`).

### What runs automatically

| Workflow | When | What it does |
|---|---|---|
| `daily.yml` | 08:20 UTC daily | New results (TennisMyLife and Matchstat), rankings, the court speed index (Tennis Abstract) and closing odds; settles yesterday's bets; rebuilds ratings and retrains every model; runs the tests; commits everything |
| `slate.yml` | Every 4 hours | Pulls upcoming fixtures (Matchstat) and prices (The Odds API); predicts every match in the next 48 hours; logs predictions and recommended bets with a timestamp before the match starts |

**The Odds API quota:** each slate run costs about 1 credit per active tournament, or 2 during a Slam (moneyline plus sets totals).
Every 4 hours fits the free 500-credit plan. On a paid plan, change the cron in `slate.yml` to `*/2`. To also
compare North American books (FanDuel, DraftKings, BetMGM, …), add a repo **variable** `ODDS_API_REGIONS` = `eu,us`. This doubles the cost.

## How a prediction is made

1. **One rating engine for training and live use** (`src/atp_edge/engine.py`). It replays every tour-level match
   since 2000 in date order. Training features and tomorrow's features come from the same code, so the two can't drift apart.
2. **Features:**
   - **Ratings:** dynamic-K Elo, overall and per surface, weighted by margin of victory.
   - **Serve and return:** surface-specific serve and return points won, ace and double-fault rates.
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
4. **Fair price (market-anchored, the default):** the model's probability combined with Pinnacle's no-vig price, using weights
   fitted on historical closing odds. The weight on the model is never negative.
5. **Bet:** the best price across all books is compared with the fair price. A bet is called when EV ≥ 2% (adjustable).
   Stake is ¼-Kelly, capped at 2% of bankroll (1.5% for set totals).

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

| | Log loss | Accuracy |
|---|---|---|
| Model | 0.604 | 66.3% |
| Pinnacle closing price | **0.586** | **68.0%** |

| Rule at Pinnacle closing prices | Bets | Actual ROI |
|---|---|---|
| Model only, EV ≥ 2% & edge ≥ 2% | 13,080 (70% of matches) | −4.1% ± 1.1% |
| Model only, EV ≥ 5% & edge ≥ 5% | 8,751 | −2.8% ± 1.3% |
| Market-anchored (default) | almost none | n/a |

The closing market is more accurate than any public-data model, so this repo doesn't pretend otherwise:

- **Default mode looks for book prices that lag the fair price.** That's line shopping, and it's where public-data edges realistically are.
- **Every recommendation is logged before the match** and graded against the closing price. Closing-line value (CLV) on the
  *Results & CLV* page is the number to watch. Positive CLV over a few hundred bets is the evidence that matters.
- **Model-only mode is still in the sidebar** for research.

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
  predict.py / live.py    slate building and recommendations
  tracking.py             prediction & bet logs, settlement, CLV
  injuries.py             manual injury flags
scripts/                  update_data.py, train.py, run_slate.py
data/seed                 match archive 2000–2026 (bootstrap; refreshed daily)
data/reference            court speed table, injury flags
data/odds                 historical closing odds (tennis-data.co.uk, evaluation only)
data/predictions          latest slate, prediction log, bet log (written by Actions)
```

Data sources: TennisMyLife match archive, Matchstat via RapidAPI, The Odds API, Tennis Abstract surface-speed
ratings (Jeff Sackmann), and tennis-data.co.uk closing odds. Check each provider's terms before commercial use.
This tool estimates probabilities and doesn't guarantee profit.
