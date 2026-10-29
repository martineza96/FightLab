# Fight Lab: UFC Fight Predictor & Simulator

A learning project that goes from data analysis to data science to a web app. It predicts UFC fights from
information known **before** the fight, and simulates them.

- **Predictor:** a calibrated logistic regression on "fighter A minus fighter B" differences, built only from
  earlier fights. On 1,475 unseen 2024–26 fights it picks **63.8%** of winners (log loss 0.634, AUC 0.696), against
  55.1% for "always pick the red corner" and 66–69% for betting favourites.
- **Simulator:** a Markov-chain model that plays a fight out 5 seconds at a time across four positions,
  10,000 times per matchup (61.7% on the same test set).
- **First live check:** on UFC 333 (24 October 2026), predicted a week before with no later data, it went 8 of 14.

**Full write-up** (data → models → simulator → app, with interview Q&A): [`docs/PROJECT_BREAKDOWN.md`](docs/PROJECT_BREAKDOWN.md).

### What's in this repository
| Folder | Contents |
|---|---|
| `notebooks/` | The analysis, in order: 01 understanding the data → 10 the Markov fight simulator |
| `src/ufc/` | The reusable library the notebooks import (cleaning, features, Elo, models, simulator) |
| `data/` | Raw snapshots and the processed tables the notebooks read |
| `tests/` | Tests for the simulator and the geography matching |
| `docs/` | The project breakdown and its figures |

The **Fight Lab web app** (a FastAPI backend and an Ionic React front end that serve these models) isn't in
this repository yet.

## Data source
Fight data comes from ufcstats.com via the public repo
[Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats) (GPL-3.0), which
republishes it with round-by-round stats, strike breakdowns, fighter birthdates and judges' scorecards.
ufcstats.com itself now blocks automated scrapers with a browser challenge, so we don't scrape it directly.

The original 2024 files (`Large set/`, `Medium set/`, … and copies in `data/raw/*.csv`) are kept
for reference and for the cross-check in notebook 01b. The data is redistributed here under the source's GPL-3.0 licence.

## Layout
```
data/raw/greco/<date>/  dated snapshots of the Greco1899 CSVs, never edited
data/raw/*.csv          original 2024 files (ufcstats through 2024-03-23)
data/processed/         pipeline outputs: fights, fighter_fights, fight_rounds, judge_scores,
                        fighters (directory), elo_history, model_table
notebooks/              01 understanding → 01b new source → 02 cleaning → 03 EDA → 04 baseline model
                        → 05 fighter explorer & pound-for-pound rankings
                        → 06 careers, retention & the point of decline
                        → 07 form, "who did they beat", fight chains, method of victory
                        → 08 short records, explainability, error analysis, matchup cards
                        → 09 home advantage (fight location × fighter origin)
                        → 10 Markov-chain fight simulator (Monte Carlo)
src/ufc/                ingest.py (new source), cleaning.py (shared rules), features.py,
                        elo.py (ratings, P4P), fighters.py (search, cards, last fights),
                        careers.py (survival curves, aging curves, early-warning model),
                        form.py (recent form, opponent quality, OWP/OOWP, common opponents),
                        modeling.py (shared train/eval, shrinkage, method model, explanations),
                        geo.py (event country, fighter origin via ESPN + Wikidata, home features),
                        markov.py (Markov-chain fight simulator), viz.py
```

## Fighter explorer (notebook 05)
```python
from ufc.fighters import FighterDB
from ufc import elo
db = FighterDB()
db.search("borz")                         # names + nicknames, accent-insensitive
db.card("Israel Adesanya")                # profile, record, streak, Elo, P4P rank
db.last_fights("Israel Adesanya", n=5)    # results + stats for both fighters
db.percentiles("Israel Adesanya")         # stats vs division peers (100 = best)
elo.p4p(db.hist, "2015-01-01")            # pound-for-pound top 15 on any date
elo.peak_elo(db.hist)                     # all-time peak ratings
```
Every function takes an optional `as_of` date to see a fighter as they were at that time.
Pound-for-pound uses an Elo rating (K=64, finishes ×1.25) over all UFC fights, limited to active fighters
(fought in the last 18 months, at least 5 UFC fights).

## Careers & decline (notebook 06)
- **Retention:** Kaplan-Meier survival curves of UFC career length. Active fighters are treated as censored, not as finished careers.
  Debut winners last about twice as long as debut losers.
- **Aging:** "result minus Elo expectation" by age compares each fighter with their own rating.
  It turns negative at about **31**, is clearly negative from 34, and is about twice as bad from 38.
  Stat curves use the delta method with a regression-to-the-mean correction.
- **Early warning:** age + years active + UFC fights + own/opponent Elo + points below own peak predict "loses 2 of the next 3" (AUC ≈ 0.64 on 2020+ fights).
  Damage and durability stats add nothing beyond those.

## Prior work (sources used in notebook 06)
- Betting markets: [UFCalendar 14-year odds study](https://www.ufcalendar.com/blog/ufc-odds-line-movement-14-year-study), [AgentMMA favorites](https://agentmma.com/mma-lab/ufc-favorites-win-rate)
- Careers and aging: [AgentMMA debut loss](https://agentmma.com/mma-lab/ufc-debut-loss-career-length), [AgentMMA peak age](https://agentmma.com/mma-lab/ufc-fighter-peak-age),
  [UFCalendar decline](https://www.ufcalendar.com/blog/when-do-ufc-fighters-get-washed), [Fight Opinion 9-year rule](https://www.fightopinion.com/2011/06/19/9-year-rule-mma-ufc/),
  [Bloody Elbow longevity](https://bloodyelbow.com/2013/07/18/ufc-going-deeper-into-mma-fighter-longevity-does-age-matter)
- Knockouts: [AgentMMA KO repeat risk](https://agentmma.com/mma-lab/ufc-knockout-loss-repeat-risk), [RAMA 2023](https://doaj.org/article/b12f3505b5494fe98e56eb45bba29a22)
- Ratings and models: [Király & Qian, Elo/Bradley-Terry](https://arxiv.org/abs/1701.08055v1), [Holmes et al. 2024, Bayesian MMA judging](https://livrepository.liverpool.ac.uk/id/eprint/3171986)

## Fight Lab app
The models above power a local web app (FastAPI + Ionic React) with six tabs: the next UFC card with predictions
and their reasons, fighter pages, any-matchup predictions, the simulator with an animated replay, breakdown charts,
and a plain-English model page. A card review grades the model on the last finished card. The app code isn't
in this repository yet.

## Setup
**Conda (recommended for notebooks):**
```bash
conda env create -f environment.yml   # creates the "ufc" environment (first time only)
conda activate ufc
python -m ipykernel install --user --name ufc --display-name "Python (ufc)"
```
Then in **Anaconda Navigator**: Environments → `ufc` → Home → **JupyterLab** → Launch, and open `notebooks/`.
Notebooks use the kernel **"Python (ufc)"**.
After changing `environment.yml`: `conda env update -f environment.yml`.

Run the tests with `PYTHONPATH=src pytest -q tests`.

**Plain venv (scripts / terminal):**
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab            # open notebooks/ and run them in order
```
Refresh to the latest data and rebuild everything without Jupyter:
```bash
PYTHONPATH=src .venv/bin/python -c "from ufc import ingest; ingest.download()"
PYTHONPATH=src .venv/bin/python -m ufc.ingest
```

## Key data gotchas (details in notebooks 01 and 01b)
1. **In-fight stats** (strikes, takedowns, control time) are only known after the fight → used only to build *history*.
2. **Career stats** on fighter pages describe the whole career, including fights after the one you're predicting → rebuilt from prior fights only.
3. **Before 2010-03-21 the winner is always listed in the red corner** → those fights feed history but aren't prediction targets.
4. **Shared names** (two Bruno Silvas, two Jean Silvas) → split by weight class and birthdate.
5. **Scorecards are written loser-first** ("28 - 29" = loser got 28) → mapped to corners using the winner.
6. **Duplicate events / spelling variants** → deduplicated by fight URL; names matched after normalization.

## Prediction results (train 2010-03 → 2023, test 2024-01 → 2026-10, 1,475 fights)
Numbers from the conda `ufc` env. A plain venv with older library versions gives 64.3% for the history model: one fight's difference, from numerics.

| Model | Accuracy | Log loss | AUC |
|---|---|---|---|
| Always pick Red | 55.1% | 0.690 | 0.500 |
| Logistic, minimum features (nb 04) | 61.6% | 0.653 | 0.658 |
| Logistic, + fight history (nb 04) | 64.2% | 0.640 | 0.684 |
| Elo only (nb 05) | 56.0% | 0.680 | 0.578 |
| Logistic + history + Elo (nb 05) | 63.7% | 0.637 | 0.691 |
| History, shrunk toward division average, k=5 (nb 07) | 64.2% | 0.636 | 0.690 |
| **Shrunk history + "who they beat" (nb 07, chosen on 2021–23 validation)** | **63.9%** | **0.635** | **0.694** |
| **+ home crowd abroad / US-event split (nb 09; live app model)** | **63.8%** | **0.634** | **0.696** |
| Shrunk + every new block (not chosen; reference only) | 64.8% | 0.633 | 0.696 |
| Gradient boosting, every block | 64.1% | 0.638 | 0.688 |
| *Leaky: in-fight stats* | *88.4%* | *0.284* | *0.952* |
| *Leaky: career snapshot* | *82.5%* | *0.388* | *0.906* |

**Who and how (7 outcomes, nb 07):** base rates 1.726 log loss · one multinomial model 1.615 · two-step
(P(winner) × winner's shrunk KO/SUB/DEC mix) 1.619. Both pick the exact outcome about 33% of the time vs 22% for base rates.

## Form, opponents & explainability (notebooks 07–08)
- **Your lead-up vs theirs:** fighters who won all of their last 5 win about 62% of the next fight, but only 50% against another 5-of-5 fighter. What matters is the gap.
- **Strength of schedule** (opponents' and opponents' opponents' win rates) is the only new layer whose gain clears the bootstrap noise test.
  Fight chains ("A beat C, C beat B") are right about 57% of the time and add nothing to the model.
- **Shrinkage** (partial pooling of career rates) is the biggest single improvement.
- **What drives predictions:** age first, then strikes absorbed, takedowns, and times finished. Reach barely matters once everything else is known.
- **Short records:** fights involving a UFC debutant are about 57% predictable. The strongest debut signal is matchmaking: prospects are booked to beat veterans.
- **Misses** are mostly close fights, submissions, and younger underdogs. The model is well calibrated (its 75%+ favorites win about 78%).

## Home advantage (notebook 09)
- **Fighter origin:** ESPN citizenship, looked up by name and confirmed by birth date, with gaps filled from Wikidata. Covers 97% of fights since 2005.
  It's stored in `data/processed/fighter_origin.parquet`; rebuild it with `geo.build_origins()`.
- **Home crowd outside the USA:** home fighters beat their own Elo expectation by about **+9 points**, a gain that holds in every era since 2005.
  Judges give them about **+4.6 points** more scorecards than the fight stats earn, in line with published work.
- **"Home" in the USA is a different effect:** Americans at US events *underperform* their ratings against foreign fighters (who are a selected group). So the model splits the two effects.
- **The 2020–21 empty-arena test is inconclusive:** only 12 home-crowd fights abroad took place without fans.
- **For the model:** better on 2021–23 validation (log loss 0.6440 → 0.6415), and neutral to slightly better on the 2024+ test. It's part of the live app model.
  The Predict tab has a fight-location picker, and fighter pages show home/away records.

## Fight simulator (notebook 10)
A simplified version of the best published MMA model: Holmes, McHale & Zychaluk (2023), *A Markov chain model for forecasting results of mixed martial arts contests*, International Journal of Forecasting.

**How it works:**
- A fight is played out in 5-second steps across four positions: distance, clinch, A on top, B on top.
- Each step, fighters land strikes at their rate for that position, and each landed strike may cause a knockdown or a KO.
- On the ground, either fighter may attempt a submission, and an attempt may finish the fight.
- Takedowns, clinches, stand-ups and reversals move the fight between positions.
- Fights that last go to a judge model fitted on real decisions, which picks the official winner in 85% of decisions.
- Every rate is "A's offense × B's defense" relative to the division average, recency-weighted (2-year half-life), and shrunk toward the average.
```python
from ufc import markov as MK
S = MK.Simulator(ff, before="2026-10-18", P=MK.Params(k_min=40, k_ko=400), half_life=2)
MK.calibrate(S, some_fights)                       # match league ground/clinch time and KO/SUB rates
o = S.matchup("Islam Makhachev", "Ilia Topuria", rounds=5, n=10000, log=5, raw=True)
o["p_a"], o["A KO/TKO"], o["finish_round"]         # win %, method, round
MK.play_by_play(o, ("Makhachev", "Topuria"), 0, 0) # one simulated fight, event by event
```

**Test results (2024+, 1,475 fights):**

| Model | Accuracy | Log loss | AUC |
|---|---|---|---|
| Logistic on the same in-fight rates | 60.0% | 0.665 | 0.634 |
| **Markov simulation** | **61.7%** | **0.656** | **0.659** |
| App model (nb 09) | 63.8% | 0.634 | 0.696 |
| App model + Markov log-odds (*not adopted: validation said no*) | 65.8% | 0.629 | 0.701 |

**Findings:**
- **Simulating beats regressing on the same inputs.**
- **It doesn't beat the app model alone.** The app model also knows age, record and matchmaking.
- **The blend is better on test but not on validation,** so by the rule it waits for fresh fights.
- **Who + how:** the hybrid (app model's winner × the simulation's method mix) is best on test (1.599 vs 1.614 for the two-step model). Validation preferred the two-step.
- **Monte Carlo charts in the notebook:** convergence, outcome splits, fight-length, strike and takedown distributions, a round × method heatmap, 200-fight "spaghetti", and a one-fight timeline.
- **In the app:** the Simulate tab, served by `src/ufc/simservice.py`.
