# Fight Lab: The Full Project Breakdown

*From raw fight data to a predictor, a fight simulator, and a web app. Written to be read before an interview.*

> [!NOTE]
> **How to use this document.** Every chapter has the same rhythm. **In one sentence** gives the idea. The body explains **how it works**, with the technical detail. **Say it like this** is how to put it in an interview. **Key numbers** lists the figures worth memorising. If you only have ten minutes, read chapter 1 and the interview questions in chapter 9.
>
> The app code described in chapter 7 (`api/`, `app/`) isn't included in the GitHub repository yet; the notebooks, library and data are.

## Contents

1. [The pitch](#1-the-pitch)
2. [The data](#2-the-data)
3. [Exploring the data](#3-exploring-the-data)
4. [The Predictor](#4-the-predictor)
5. [Careers, ratings and home advantage](#5-careers-ratings-and-home-advantage)
6. [The Simulator](#6-the-simulator)
7. [The app](#7-the-app)
8. [Lessons and next steps](#8-lessons-and-next-steps)
9. [Interview questions](#9-interview-questions)
10. [Glossary](#10-glossary)

## 1. The pitch

> [!NOTE]
> **In one sentence:** Fight Lab predicts UFC fights from information known *before* the fight. It has two models: a calibrated statistical **Predictor** for *who* wins, and a **Simulator** that plays the fight out 10,000 times to show *how* it could go. Both run behind a FastAPI backend and an Ionic React app.

### The 30-second version

I built an end-to-end UFC prediction project. I started with about 8,950 fights of round-by-round data and found three separate ways the raw data leaks the answer. With those fixed, I built a logistic-regression model that picks 63.8% of winners on 2024–26 fights it never saw, against 55% for the naive baseline and 66–69% for betting favourites. Then I implemented a Markov-chain fight simulator based on the best published MMA model, and put both into a web app with live fight cards, explanations of every prediction, and an animated replay of simulated fights.

### The 2-minute version

1. **Data.** ufcstats.com blocks scrapers, so I used a public republication of its data. It covers every UFC fight round by round, with judges' scorecards and birthdates. Much of the early work was data quality: duplicate events, two different fighters both named Bruno Silva, scorecards written loser-first, and a quirk where **the winner is always listed in the red corner before 2010**.
2. **Leakage.** The biggest lesson. Models that use the obvious columns score 82–88%, but they are reading the answer. I rebuilt every feature from fights *before* the one being predicted, and split train and test by time.
3. **Predictor.** A logistic regression on the differences between the two fighters: age, striking and grappling rates, record, strength of schedule, and home crowd. Two ideas made the biggest difference. The first is **shrinkage**: pulling small-sample stats toward the division average. The second is **strength of schedule**: who your opponents beat.
4. **Discipline.** Every modelling choice was made on a 2021–23 validation window. The 2024+ test set was scored once. I even turned down a blend that looked better on test, because validation disagreed.
5. **Simulator.** A Markov chain with four positions (distance, clinch, and either fighter on top) stepping every 5 seconds. Rates are "my offence × your defence", recency-weighted and shrunk. Fights that go the distance are scored by a judge model fitted on 2,600 real decisions. It runs 10,000 vectorised simulations in about half a second.
6. **App.** FastAPI serves the models and Ionic React shows them. Tabs cover the next UFC card, fighter pages, any-matchup predictions, the simulator with an animated replay, a **card review** that grades the model on the last finished card, and a model page that explains everything in plain English.

### The project at a glance

<!-- svg:pipeline -->
```text
 ufcstats data ─► cleaning ─► exploration ─► features ─► Predictor (logistic)  ─┐
 (Greco1899)      (ingest,     (nb 03)        (prior       63.8% / 0.634          ├─► FastAPI ─► Ionic React app
                   nb 01–02)                   fights       Simulator (Markov)  ─┘    (api/)      (app/)
                                               only)        61.7% / 0.656
```

| | |
|---|---|
| **Languages and libraries** | Python 3.12, pandas, NumPy, scikit-learn, matplotlib/seaborn, Jupyter; TypeScript, React 18, Ionic 8, TanStack Query, Vite |
| **Backend** | FastAPI + Uvicorn, with model fits cached on disk under a data fingerprint |
| **Code layout** | `src/ufc/` (reusable library, about 4,000 lines), `notebooks/01–10` (the analysis story), `api/`, `app/`, `tests/` (31 tests) |
| **Data** | about 8,950 UFC fights to October 2026, 2,700+ fighters, round-by-round stats, every judge's scorecard |

> [!IMPORTANT]
> **Key numbers (2024–26 test, 1,475 fights)**
> - Always pick the red corner: **55.1%** accuracy, log loss 0.690
> - **Predictor: 63.8%**, log loss **0.634**, AUC **0.696**
> - Simulator alone: 61.7%, log loss 0.656, AUC 0.659
> - Betting favourites (published): **66–69%**
> - Leaky models that "cheat": 82.5% and 88.4%
> - First live card (UFC 333): **8/14**, inside the 39–89% that 14 fights allow

## 2. The data

> [!NOTE]
> **In one sentence:** The hard part wasn't the model. It was making sure the data only told the model what was knowable before the fight.

### Source

ufcstats.com is the official stats provider, but it now answers automated requests with a "Checking your browser…" challenge. I treated that as a "no" and didn't work around it. Instead I used the public GitHub repo **Greco1899/scrape_ufc_stats** (GPL-3.0), which republishes the same data and keeps it updated. Compared with the original 2024 Kaggle-style files, it has:

| | Original files | Greco1899 snapshot |
|---|---|---|
| Fights | about 7,500, to March 2024 | about 8,950, to October 2026 |
| Stats | whole-fight totals | **round by round** |
| Strike detail | significant strikes only | **head/body/leg and distance/clinch/ground** |
| Age | approximate | **exact date of birth** |
| Judges | none | **every judge's scorecard** |

Before switching sources, I cross-checked every fight the two datasets share. They matched, apart from name changes (Bobby Green is now "King Green") and a few seconds of rounding in control time. The round table also adds up exactly to the fight totals.

### The tables (built by `src/ufc/ingest.py`)

- **`fights`**: one row per fight, with date, division, method, round and corners.
- **`fighter_fights`**: the **long table**, with two rows per fight, one from each fighter's point of view. It holds their own stats and their opponent's (`opp_*`). This is the workhorse, because every per-fighter history is a grouped, time-ordered calculation on it.
- **`fight_rounds`**: one row per fighter per round.
- **`judge_scores`**: every judge's card.

### Leakage: the main story

**Leakage** means the model sees information it wouldn't have at prediction time. I found three kinds:

1. **Career-snapshot columns.** The original files had columns like `wins_total` and `age`, but they were the values *on the day of the scrape*. Jon Jones's 2009 fight carried his 2023 win total, and the average "age at fight" was 38. A model using them already knows how every career turned out.
2. **In-fight stats.** Strikes, takedowns and control time are produced *by* the fight. They're fine for building a fighter's *history* from earlier fights, and useless as features of the fight itself.
3. **The corner artifact.** Before 21 March 2010, ufcstats lists **the winner in the red corner 100% of the time**. A model trained on those rows learns "red always wins", which is the answer key hidden in the column order. Those fights still count toward fighters' histories, but they are never prediction targets.

To show how much this matters, I trained two deliberately leaky models. **In-fight stats scored 88.4%** and **the career snapshot scored 82.5%**, against about 64% for honest models.

> [!TIP]
> **Say it like this:** "When you see a '95% accurate UFC predictor' online, it's almost always leakage: career stats from after the fight, or a random train/test split. I measured it: 82–88% with leaky features, 64% honestly."

### Other data gotchas I fixed

- **Duplicate events.** The same card appeared under two names. Fixed by deduplicating on the fight URL, which is the real ID.
- **Spelling variants.** "YiSak Lee" vs "Yi Sak Lee". Names are normalised by removing accents, spaces, hyphens and suffixes, with a short manual nickname list. I tried fuzzy matching and **rejected** it: it matched "Muhammad Saidov" to a different fighter, "Muhammad Said".
- **Different people with one name.** There are two Bruno Silvas and two Jean Silvas. Each fight goes to the profile whose weight fits the division and whose birthdate gives a plausible age.
- **Loser-first scorecards.** "28 – 29" means the *loser* got 28. Fixed by mapping cards to corners using the winner.
- **Mojibake.** "Cláudia" had been saved as UTF-8 and read back as Cyrillic cp1251, becoming "ClГЎudia". Fixed by reversing the encoding.
- **Weight classes.** More than 100 labels ("Road to UFC 3 Flyweight Tournament TitleBout", …) were collapsed into 15 divisions.

> [!WARNING]
> **Clean labels beat more data.** Removing the 1,290 pre-2010 training rows (where red always wins) *improved* every honest model.

## 3. Exploring the data

> [!NOTE]
> **In one sentence:** Exploration asked one question, "what, known before a fight, relates to winning?", and it found that age matters most, reach and stance matter a little, and the red-corner edge is shrinking.

A useful trick: to study any attribute without the red-corner bias, use the **long table**. Every fight contributes one win and one loss, so the baseline is exactly 50%, and "fighters with a 10 cm reach edge won 55%" means something.

<!-- img:red_corner_by_year -->

**What I found:**

- **Red corner:** about **58%** since 2010, drifting down to **55.1%** in 2024–26. The red corner is usually the favourite or champion, so this is matchmaking, not a magic colour. A model that leans on "pick red" would slowly get worse. That's **concept drift**, and it's why a time-based split matters.
- **Age** is the strongest physical signal. A fighter **8+ years older** than their opponent wins only about **a third** of the time.
- **Reach:** a big edge (**15+ cm**) is worth about **59%**. Height adds little on top, because the two overlap.
- **Stance:** southpaws beat orthodox fighters about **52–53%** of the time. It's small but consistent.
- **Experience:** UFC debutants win only about **45%**. The win rate climbs with experience, then dips for 11+ fight veterans, who are usually older.
- **Collinearity:** fight counts, minutes and win-type counts all measure experience. That makes linear coefficients unstable, which pointed toward regularisation and, later, shrinkage.

<!-- img:physical_edges -->

## 4. The Predictor

> [!NOTE]
> **In one sentence:** A logistic regression on "fighter A minus fighter B" differences, built from each fighter's history before the fight, with noisy small-sample stats pulled toward the division average.

### The evaluation setup (decided before any modelling)

<!-- svg:timeline -->
```text
 2010-03 ───────────── training ───────────── 2020 | 2021–23 validation | 2024-01 ── test (1,475 fights) ── 2026-10
                                                   (all choices made here)  (scored once, at the end)
```

- **Train** on fights from March 2010 (when the corners become trustworthy) to 2023. Earlier fights still feed fighter histories.
- **Validate** on 2021–23. Every choice is made here: which feature blocks, shrinkage strength, simulator settings.
- **Test** on 2024 to October 2026. This is what "predicting the future" really looks like.
- **Metrics:**
  - **Accuracy** is intuitive but crude.
  - **Log loss** punishes confident mistakes. It's the main metric, because the goal is *honest probabilities*.
  - **AUC** measures how well the model ranks fighters.

### Feature engineering

- **Differences, not levels.** Every feature is A's value minus B's, for example age difference and strikes-absorbed difference. The model sees the gap.
- **Prior fights only.** Every career rate (strikes landed and absorbed per minute, takedowns per 15 minutes, defence %, times finished, win streak) is recomputed as of the day before each fight.
- **Corner-swap augmentation.** Each training fight is duplicated with the corners swapped and the target flipped, plus a `red_corner` column (+1 or −1). The fighter features must then explain the result *symmetrically*, and the red-corner effect (matchmaking) is isolated in one coefficient.

### Shrinkage (empirical Bayes): the biggest single gain

<!-- svg:shrinkage -->
```text
 shrunk rate = (n · fighter's rate + k · division average) / (n + k)      with k = 5 fights

 1 UFC fight   → about 83% division average, 17% fighter
 20 UFC fights → about 80% fighter, 20% division average
```

A fighter with 2 UFC fights might show 9 strikes absorbed per minute. That's mostly noise. Shrinkage is **partial pooling**: it trusts each fighter's numbers in proportion to how much evidence there is. *k* was chosen on validation. This single change improved log loss about as much as all the new feature blocks combined, without adding a feature.

### Feature blocks tested (notebooks 05, 07, 09)

| Block | Idea | Verdict |
|---|---|---|
| **Elo rating** | A chess-style rating: start at 1500, K = 64, finishes move ratings 25% more | Good for rankings, but only 56.0% alone. Adds nothing once the other features are in |
| **Recent form** | Last-5 results by method, finish and decision streaks | Helps a little, within noise |
| **Who they beat** | The streak you snapped, the average Elo of opponents you beat | Helps a little; chosen by validation |
| **Strength of schedule (OWP/OOWP)** | Opponents' win rate, and their opponents' win rate (the RPI idea from college sports) | **The only block whose gain clears the bootstrap noise test** |
| **Fight chains ("MMA math")** | A beat C, C beat B | Right about 57% of the time when it exists; adds nothing |
| **Home crowd** | Fighting in your home country | Split into "home abroad" and "US event" (chapter 5); improved validation, so adopted |

Blocks were added **greedily on the validation score**. The bootstrap 95% interval for each change in log loss tells real gains from luck.

<!-- img:ablation -->

> [!WARNING]
> **The price of honesty.** Including *every* block scores 0.633 log loss on test, slightly better than the chosen model's 0.635. But the only way to know that is to look at the test set. Picking by test score is quietly cheating, because with enough tries something always "helps". I kept the validation choice.

### Results ladder (2024–26 test, 1,475 fights)

| Model | Accuracy | Log loss | AUC |
|---|---|---|---|
| Always pick red | 55.1% | 0.690 | 0.500 |
| Logistic, minimum features (nb 04) | 61.6% | 0.653 | 0.658 |
| + fight history (nb 04) | 64.2% | 0.640 | 0.684 |
| Elo only (nb 05) | 56.0% | 0.680 | 0.578 |
| History, shrunk k = 5 (nb 07) | 64.2% | 0.636 | 0.690 |
| Shrunk + "who they beat" (nb 07, chosen on validation) | 63.9% | 0.635 | 0.694 |
| **+ home crowd split (nb 09), the live model** | **63.8%** | **0.634** | **0.696** |
| Gradient boosting, every block | 64.1% | 0.638 | 0.688 |

> [!TIP]
> **Why accuracy went *down* slightly while the model got better:** accuracy only counts which side of 50% you land on. Log loss and AUC measure the probabilities themselves, and those kept improving. One more correct pick out of 1,475 fights is 0.07 points of accuracy, which is noise.

**Why logistic regression over gradient boosting?** Boosting scored worse on log loss (0.638). With about 6,000 training fights and features that are mostly linear in effect, the extra flexibility mostly fits noise. Logistic regression is also **exactly explainable**: every prediction is "starting point + each feature's push" in log-odds, and that's what the app shows as plain-English "drivers".

### Who *and how*

There are seven outcomes: red or blue by KO/TKO, submission or decision, plus "other". I compared three approaches:

| Approach | 7-outcome log loss |
|---|---|
| Base rates (always the historical mix) | 1.726 |
| One multinomial logistic model | 1.615 |
| **Two-step:** P(winner) × the winner's shrunk KO/SUB/DEC mix | 1.619 |

Both models pick the exact outcome about **33%** of the time, against **22%** for base rates. I chose the two-step model because it's simpler and is exactly the shape a simulator needs. Fighters keep winning the way they've been winning: with 2+ KO wins in their last five, a fighter wins by KO about **46%** of the time, against 27% with none.

### How it decides, and where it's wrong (notebook 08)

<!-- img:feature_importance -->

- **Most important:** **age** first, with each year younger worth about 1–1.5 points. Next come strikes absorbed, takedowns per 15 minutes, and times finished. **Reach barely matters** once everything else is known: ±15 cm moves the needle about 1 point.
- **Calibration:** the model's 70–75% favourites won **74%**, and its 75%+ favourites won **78%**. So most misses are honest upsets, not a broken model.
- **Where it misses:**
  - Confidence is the big divider: about 54% accuracy on 50–60% calls, against about 76% on 70%+ calls.
  - Debut fights are the hardest group, at about 57%.
  - Submissions are harder to see coming than knockouts (about 59% vs 67%).
  - In **38%** of misses the underdog was the *younger* fighter, against 28% of hits.
- **A surprise about debut fights:** in fights with a debutant, *more* UFC experience predicts *losing*. That's matchmaking: the UFC books prospects it believes in against veterans on the way down. The model learned the matchmaker's opinion.

<!-- img:calibration -->

## 5. Careers, ratings and home advantage

> [!NOTE]
> **In one sentence:** Side analyses that became app features: pound-for-pound ratings, when fighters decline, how long careers last, and whether the home crowd matters.

### Elo and pound-for-pound (notebook 05)

Every fighter starts at 1500. After each fight the winner takes points from the loser, more when the result is a surprise (K = 64, finishes × 1.25). Weight class plays no part, so the top active ratings form a **pound-for-pound** list. "Active" means fought in the last 18 months, with at least 5 UFC fights. A "time machine" shows P4P on any date. The lesson: **a good ranking tool isn't automatically a good predictor**. Elo alone picks 56.0%.

### When do fighters decline? (notebook 06)

- **The survivor problem:** comparing different fighters at different ages flatters age, because fighters who decline early leave the sample.
- **The fix:** compare each fighter with *themselves*, using **result minus Elo-expected win chance**.
- **The answer:** the average turns negative at about **31**. It is clearly negative from **34** and roughly twice as bad from **38**. A 35-year-old's rating runs about 5 points per fight too high; a 40-year-old's about 12.
- **For individual stats** I used the baseball **delta method**, with a regression-to-the-mean correction. Without it, the curve "declines" from age 21, purely through selection: a lucky year buys the next one.

<!-- img:aging_vs_elo -->

### How long do careers last? (notebook 06)

**Kaplan–Meier survival analysis** handles **censoring**. About a quarter of fighters are still active, so we only know their careers are *at least* this long. Dropping them biases careers short, and so does treating them as finished. Debut winners' careers last roughly **twice as long** as debut losers', which replicates a published finding.

An **early-warning model** predicts "loses 2 of the next 3". Age, years active, fight count, own Elo, opponent's Elo and "points below own peak" give **AUC ≈ 0.64**. Damage and trend stats add nothing on top.

### Home advantage (notebook 09)

- **Fighter origin:** ESPN citizenship plus a Wikidata fill. It covers 97% of fights since 2005.
- **Outside the USA:** home fighters beat their own Elo expectation by about **+9 points**, a gain that holds in every era. Judges give them about **+4.6 points** more scorecards than their fight stats earn, close to a published +3.6.
- **Inside the USA, "home" fighters *under*-perform.** Almost every American is "home" at US events, so "home" really means "American vs a foreigner". Foreigners on US cards are a selected group. One plain "home" flag cancels out, so the model uses **two** features.
- **The COVID empty-arena natural experiment was inconclusive.** Only 12 home-crowd fights abroad took place without fans. I reported that honestly rather than over-claiming.
- **For the model:** it improved validation (0.6440 → 0.6415) and was neutral to slightly better on test. In plain terms, a home crowd abroad turns a 50/50 fight into about 57/43.

## 6. The Simulator

> [!NOTE]
> **In one sentence:** Instead of mapping stats straight to a probability, the Simulator plays the fight out 5 seconds at a time across four positions, 10,000 times, and counts what happens.

### Why this model

I researched published UFC models. The most credible is **Holmes, McHale & Zychaluk (2023), "A Markov chain model for forecasting results of mixed martial arts contests", *International Journal of Forecasting* 39(2)**. The authors report predictions "comparable to the bookmakers" that made money when used for betting. Its inputs are ufcstats-style box scores, which I already had. I built a simplified version.

### The Markov chain

<!-- svg:markov -->
```text
              takedown                         reversal
  DISTANCE ◄──────────► CLINCH       A ON TOP ◄─────────► B ON TOP
     │  ▲  clinch / break   │           ▲  │                  │
     │  └──── stand-up ─────┼───────────┘  └── stand-up ──────┘
     └──────── takedown ────┴─────────────►  (ground)
  every state can end in:  KO/TKO (from a landed strike) · SUB (from a ground attempt) · DECISION (time runs out → judge model)
```

- **States:** distance, clinch, A on top, B on top. There are also **absorbing** end states: KO, submission, decision.
- **Time:** 5-second steps, so 60 per round, 180 in a 3-round fight and 300 in a 5-round fight. Each round restarts at distance.
- **Each step:**
  - Both fighters may land significant strikes at their rate *for that position*.
  - Each landed strike carries a small chance of a knockdown and of a KO.
  - On the ground, either fighter may attempt a submission.
  - The position may change: takedowns, clinch entries and breaks, stand-ups, reversals.
- **"Markov"** means the next step depends only on the current state, not on how the fight got there. That's a simplification: real fighters tire and adjust.

### Where the rates come from

ufcstats doesn't record how long a fight spent in each position, so I **estimate it from the box score**:
- Control time is split between ground and clinch in proportion to where strikes and takedowns happened.
- Ground time is split between the fighters by their own control time.
- Everything else is distance.

The simulator uses exactly the same accounting, so a rate means the same thing in the data and in the simulation.

**Offence meets defence (the log5 / odds-ratio rule):**

```text
 rate(A lands at distance vs B) = league × (A's offence ÷ league) × (B's "allowed" ÷ league)
```

A great striker against a hittable opponent lands well above the average; against a great defender they're pulled back toward it. Every rate is also:

- **Recency-weighted**, with a **2-year half-life**: a fight from 2 years ago counts half as much as last month's. That lets cross-era matchups (GSP vs today's champions) use each fighter's form at the time.
- **Shrunk** toward the division average: **40 minutes** of "average fighting" are added to every career, and **400 landed strikes** for KO power and chin. A debutant is exactly division-average.
- **Chosen on 2021–23 validation**, scored by log loss, at 400 simulations per fight with shared random numbers.

### Calibration: does the simulated UFC look like the real one?

A few league-wide settings aren't directly observed: how long ground spells last, how often fighters clinch, and an overall scale on KO and submission hazards. They're tuned so a sample of **2015–20** fights (well before validation and test) matches real ground time, clinch time, KO rate and submission rate. Then the **untuned** numbers act as a free check:

| Check (not tuned) | Simulated | Real |
|---|---|---|
| Significant strikes per minute | 6.73 | 7.13 (about 6% low) |
| Fight length (minutes) | 10.95 | 10.85 (within 0.1) |
| Takedowns per 15 minutes | 3.06 | 2.82 (about 8% high) |

<!-- img:finish_rounds -->

### The judge model

Fights that go the distance need a scorer. A **logistic regression on 2,600 real decisions** uses the differences in strikes, knockdowns, takedowns, minutes on top and submission attempts. It's symmetric and has no intercept, so there's no corner bias. It picks the official winner **85.3%** of the time from box-score totals alone. Its coefficients say what the judges reward: **one knockdown ≈ 8.2 strikes**, **a minute on top ≈ 4.3 strikes**, **a takedown ≈ 2.3 strikes**.

### Monte Carlo: why 10,000 fights

A **Monte Carlo** estimate is an average over many random runs. After 100 simulated fights the win estimate still wobbles by several points; after 10,000 the Monte Carlo error is under a point. The simulation is **vectorised in NumPy**: all 10,000 fights advance together, one 5-second step at a time, as arrays. 10,000 five-round fights take about 0.6 seconds, fast enough for a live API.

> [!TIP]
> **Say it like this:** "10,000 runs make the *simulation* precise, not the *model* right. Monte Carlo error is under a point; the model's real uncertainty is much bigger than that."

<!-- img:mc_convergence -->

<!-- img:round_method_heatmap -->

### Results (2024–26 test)

| Model | Accuracy | Log loss | AUC |
|---|---|---|---|
| Logistic on the *same* in-fight rates | 60.0% | 0.665 | 0.634 |
| **Markov simulation** | **61.7%** | **0.656** | **0.659** |
| Predictor (live app model) | 63.8% | 0.634 | 0.696 |

1. **Simulating beats regressing on the same inputs.** The structure itself (positions, finishes, judges) adds real information.
2. **It doesn't beat the Predictor alone.** The simulation sees only *how* fighters fight. It doesn't see age (the strongest feature), record, quality of opposition, or matchmaking.
3. **Its probabilities are honest as they are.** A recalibration fitted on validation changes almost nothing.

### The blend I didn't adopt

Adding the simulation's verdict (its log-odds) as a feature of the Predictor gave:

- **Validation (2021–23):** very slightly *worse*, 0.6420 vs 0.6415.
- **Test (2024+):** clearly better: **65.8%**, log loss 0.629, AUC 0.701, with a 95% interval on the log-loss change entirely below zero.

By the rule set at the start, I did **not** adopt it. Switching because of the test result would be choosing on the test set. The fair next step is to re-check on a fresh season of fights.

> [!TIP]
> **This is the story to tell in interviews.** "I had a model that was two points better on my test set and I didn't ship it, because my validation data disagreed and I'd committed to choosing on validation. If I start picking on the test set, my test number stops meaning anything."

### Who and how, revisited

- **Validation** prefers the Predictor's two-step method model (1.645 vs 1.659 for the hybrid).
- **On test**, the **hybrid** (Predictor's winner × the simulation's method mix) is best (1.599 vs 1.614). It picks the exact outcome 34% of the time vs 32%, and the simulation gets the *method* right more often than anything else (51%).
- So the simulation's view of finishing mechanics is at least as good as career method mixes, and may be better. That's another candidate for a fresh-data check.

### Limitations

- No fatigue, cuts or between-round adjustments. Real finishes cluster earlier than simulated ones.
- Position times are estimated from control time.
- Rates don't adjust for the quality of past opponents.
- Debutants are division-average.

<!-- img:fight_timeline -->

## 7. The app

> [!NOTE]
> **In one sentence:** A thin FastAPI layer over the same Python library the notebooks use, and an Ionic React front end that makes every model output readable for fans.

### Architecture

<!-- svg:architecture -->
```text
 Ionic React + Vite (app/)  ──/api/*──►  FastAPI (api/main.py)  ──►  src/ufc library  ──►  data/processed/*.parquet
 TanStack Query caching                  JSON, startup model fits       predict, simservice,     + caches keyed on a
                                                                        fighters, elo, upcoming  data fingerprint
                                                ▲
                                        ESPN public scoreboard (next cards + results)
```

- **One library, two consumers.** The notebooks and the API import the same `src/ufc` code, so the website uses exactly the feature pipeline that was evaluated. There is no reimplementation to drift.
- **The live prediction trick.** `predict.py` builds a placeholder "next fight" row for every fighter with the training feature code, then refits the final model on all fights.
- **Corner-neutral.** Schedules don't say who is in the red corner, so live predictions don't depend on corner order.
- **Unknown fighters** are treated as UFC debuts, predicted from ESPN age, height and reach, and flagged.
- **Caching.** Model fits, simulator calibration and precomputed charts are stored under `data/processed/` with a fingerprint of the data files, so they rebuild automatically when the data changes.
- **Endpoints:** fighters (search, profile, fight detail), rankings (P4P, peak), insights, model report, `POST /api/predict`, `POST /api/simulate`, upcoming events, card review (`GET /api/review/{latest|id}`).

### The tabs

| Tab | What it shows |
|---|---|
| **Next card** | Upcoming UFC events, each bout with a win-probability bar, the most likely outcome and the top 3 drivers |
| **Fighters** | 2,700+ fighters: record, Elo, P4P rank, stats as division percentiles, last 5 fights round by round, career stage |
| **Predict** | Any two fighters: win %, a diverging driver chart, who-and-how odds, tale of the tape, fight-location picker |
| **Simulate** | 10,000 simulations: an animated replay (cage view, scoreboard, position timeline, 1–8× speed), outcome odds, round × method heatmap, convergence, strike-race "spaghetti", distributions, position mix, matchup "fight DNA" |
| **Breakdowns** | P4P on any date, all-time peaks, aging curve, KO risk, survival curves, trends |
| **Card review** | The last finished card graded: picks vs results by confidence band, the reasons that pointed to each winner, and tale-of-the-tape edges vs history (reached from a banner on Next card) |
| **Model** | Both models explained step by step in plain English, scores vs baselines and betting markets, calibration, weak spots |

**Engineering basics:** a `Makefile` (`make dev`, `make test`, `make build`), **31 pytest tests** covering the API, geography matching, the simulator and the card review. The simulator tests check that a fighter against themselves comes out at about 50%, that swapping corners mirrors the result, that probabilities sum to 1, and that a fixed seed is reproducible. The front end is type-checked with TypeScript, and pages were checked at phone and desktop widths in light and dark mode.

### Grading the model on a real card

After UFC 333 (24 October 2026) I added a **Card review** page: the predictions for the last finished card, scored against what actually happened. Our data stopped on 17 October, so these were genuine pre-fight predictions, made exactly like the Next card tab's.

- **Results** come from ESPN's public scoreboard: who won each bout, and the round and clock it ended. A fight that reaches the final bell is a decision; anything earlier is a finish. ESPN doesn't give the method, so KO vs submission isn't checked.
- **Scoring.** A pick is correct when the model's favourite won. Picks are grouped by confidence: toss-ups (under 55%), leans (55–65%) and confident picks (over 65%). Log loss and Brier score are computed too.
- **What pointed to the winner.** Every prediction is an exact sum of feature pushes (chapter 4), so for each factor the review counts how often it pushed toward the fighter who won.
- **Tale-of-the-tape edges.** Model aside: how often the younger, longer, taller, more experienced, hotter or higher-rated fighter won, next to the same rate over every UFC fight since 2015.

**No hindsight allowed.** A review is honest only while the data stops before the card. Once the dataset includes UFC 333, "predicting" it again would use its own result: the leakage from chapter 2, sneaking back in. So the first review of each card is frozen to `data/processed/reviews/<event>.json` and served from there forever. If the data already contains a card that was never frozen, the API refuses (HTTP 409) rather than show a hindsight prediction. A test checks both paths.

<!-- img:card_review -->

**What UFC 333 showed:**

- **8 of 14 correct (57%).** Toss-ups went 3/3, leans 3/7 and confident picks 2/4. It called decision vs finish in 10 of 14.
- **Experience beat youth.** The fighter with more UFC fights won 10 of 13, against 48% since 2015. The younger fighter won only 5 of 14, against 58%. Age is the Predictor's strongest feature, so that night it carried more push toward the *loser* than any other factor.
- **"Quality losses"** (only ever losing to strong opponents) pointed to the winner in 5 of 6 bouts.

**Is 57% bad?** One card is a tiny sample. A model that's truly 63.8% accurate lands anywhere between about 39% and 89% on a 14-fight card, purely by chance. That's the sampling error: 1.96 × √(0.638 × 0.362 / 14) ≈ ±25 points. So 8 of 14 is a clue, not a verdict. The real test is a track record built over many cards.

> [!TIP]
> **Say it like this:** "On the first real card the model went 8 for 14. That's below its 64% average, but well inside the 39–89% that 14 fights allow. The interesting part was *why*: it leaned on youth, and that night the more experienced fighters kept winning. Every review is frozen before the data updates, so the model can never grade itself with hindsight."

> [!IMPORTANT]
> **Key numbers (UFC 333, 24 October 2026)**
> - Picks: **8/14** (57%); toss-ups 3/3, leans 3/7, confident 2/4
> - Decision vs finish: **10/14**
> - Card log loss: 0.731 (test set: 0.634)
> - More UFC fights won **10/13** (48% since 2015)
> - Younger fighter won **5/14** (58% since 2015)
> - Expected range for 14 fights at 63.8%: **39–89%**

## 8. Lessons and next steps

> [!NOTE]
> **In one sentence:** Most of the value came from honesty about the data and the evaluation, not from fancier models.

### Lessons

1. **Leakage hides in data-collection quirks**, not just obvious columns. Check the target against time, source, and every "harmless" grouping.
2. **Decide the evaluation rules first, then stick to them**, even when the test set tempts you.
3. **Clean labels beat more data.** Dropping 1,290 corrupted rows improved every model.
4. **Smarter handling of small samples (shrinkage) beat new features.**
5. **Simple and calibrated beats complex.** Logistic regression beat gradient boosting, and it is explainable.
6. **Structure is information.** The simulator beats a regression on the same inputs because it knows how fights work.
7. **Concept drift is real.** The red-corner edge fell from 58.4% to 55.1%.
8. **Judge a model over many cards, not one night.** UFC 333 went 8/14, inside the ±25 points one card allows.

### What I'd do next

- **Re-check the blend and the hybrid method model** on a fresh season of fights. Adopt them only if they win on data neither model has seen.
- **Add fatigue and age to the simulator:** rates that decay by round, and the aging curve from notebook 06.
- **Betting odds as a benchmark:** a fair head-to-head against closing lines, fight by fight.
- **Bayesian hierarchical rates** in place of fixed-*k* shrinkage, so each stat learns its own amount of pooling.
- **A live track record:** review every card before the data refreshes, then re-check calibration, the blend and the hybrid on those frozen predictions.
- **Deployment:** GitHub, a hosted API, and a mobile build with Capacitor.

## 9. Interview questions

**Q: Walk me through the project.**
A: Use the 2-minute version in chapter 1: data → leakage → Predictor → validation discipline → Simulator → app.

**Q: How do you know your model isn't overfit?**
A: Three ways:
- The split is by time. Training runs through 2023, choices are made on 2021–23 validation, and the 2024+ test is scored once.
- Gains are checked with a bootstrap interval on the change in log loss.
- The model is calibrated on unseen fights: its 75%+ favourites win 78%.

**Q: Why is your accuracy only 64%? I've seen 90% models.**
A: Those are almost always leaky. I built two deliberately leaky models that reach 82–88%. Honest published work and betting markets sit around 60–70%; closing favourites win 66–69%. Fights are close to the limit of what records can predict.

**Q: Explain log loss to a non-technical person.**
A: It's a penalty for being confidently wrong. Say 90% and lose, and you pay a lot; say 55% and lose, and you pay a little. It rewards probabilities you can trust, not just correct picks.

**Q: Why logistic regression instead of XGBoost?**
A: I tried gradient boosting, and it was worse on log loss (0.638 vs 0.634). With about 6,000 training fights and mostly linear effects, the extra flexibility fits noise. Logistic regression also gives exact per-feature explanations, which the app shows.

**Q: What is shrinkage and why did it help so much?**
A: A fighter with two fights has extreme, noisy rates. Shrinkage blends each fighter's rate with the division average, weighted by sample size: (n·x + k·μ)/(n + k) with k = 5. It's partial pooling, the core idea of empirical Bayes. It was the single biggest gain.

**Q: What's a Markov chain, in your simulator?**
A: The fight is always in one of four positions. Every 5 seconds there are probabilities for strikes, finishes and position changes that depend only on the current position and the two fighters' rates. Run it until a finish or the final bell, then repeat 10,000 times.

**Q: Why run 10,000 simulations?**
A: The win estimate is an average of random outcomes, so its error shrinks like 1/√n. At 10,000 the Monte Carlo error is under a point, much smaller than the model's real uncertainty. More runs make the simulation precise, not more correct.

**Q: The blend was better on the test set. Why not use it?**
A: Because validation said no, and I'd committed to choosing on validation. Choosing on test turns the test set into a second validation set and makes the reported number optimistic. It's the first thing to re-check on fresh fights.

**Q: What was the hardest bug?**
A: The pre-2010 red-corner artifact. Nothing looked wrong until I plotted red's win rate by year and saw exactly 100% before March 2010. The old data lists the winner first. It's a reminder to check the target against time.

**Q: What surprised you?**
A: Two things. Reach, the most-quoted stat on broadcasts, barely matters once age and style are known. And in debut fights, more experience predicts *losing*, because the model is learning the matchmakers' opinion of their prospects.

**Q: How does the home-crowd finding hold up?**
A: Outside the USA, home fighters beat their ratings by about 9 points, and judges give them about 4.6 points more cards than the stats earn. Inside the USA "home" means "American vs a foreigner", a selection effect, so the model uses two separate features. The empty-arena natural experiment was too small to settle it, and I said so.

**Q: How did it do on a real card?**
A: The first card I reviewed, UFC 333, went 8 of 14. That's below the 63.8% test score, but well inside the 39–89% range that 14 fights allow by chance. The review showed why: the model leaned on youth, and that night the fighter with more UFC experience won 10 of 13.

**Q: How do you stop hindsight leaking into a review?**
A: A review is built only while the data stops before the card, and the first build is frozen to disk. If the data already contains the card and nothing was frozen, the API refuses rather than re-predict with the result baked in. A test covers both paths.

**Q: How would you deploy and maintain this?**
A:
- Schedule the data download and rebuild.
- Rely on the fingerprinted caches so fits rebuild only when the data changes.
- Host FastAPI behind a small server, and build the Ionic app for web and mobile.
- Monitor calibration on new fights to catch drift, using the frozen card reviews. Re-check the blend each season.

**Q: What would you do with more time?**
A: Fatigue and aging in the simulator, a fight-by-fight comparison with betting odds, and Bayesian hierarchical rates.

**Q: What did you personally learn?**
A: Most of the gains came from data quality and evaluation discipline, not algorithms. Also, how to turn a research paper into working code and then into something a fan can understand.

## 10. Glossary

### Accuracy
The share of fights where the model's favourite won. Easy to read, but it ignores how confident the model was.

### AUC
Area under the ROC curve: how well the model *ranks* fights. Pick one fight the red corner won and one it lost at random; AUC is the chance the model gave red a higher probability in the fight red won. 0.5 is random, 1.0 is perfect.

### Bootstrap interval
Resample the test fights with replacement many times and recompute a metric each time. The middle 95% of the results is a confidence interval. If the interval for an improvement includes zero, it could be luck.

### Calibration
Whether probabilities mean what they say: do 70% favourites win about 70% of the time?

### Censoring
In survival analysis, cases where the event hasn't happened yet (an active fighter's career hasn't ended). Kaplan–Meier uses them correctly instead of dropping them.

### Concept drift
When the relationship between features and outcome changes over time, like the shrinking red-corner edge.

### Elo
A rating that moves after every fight by how surprising the result was. Ours starts at 1500 with K = 64, and finishes count 25% more.

### Leakage
Information in the features that wouldn't be available at prediction time. It makes models look far better than they are.

### Log loss
The average penalty −log(probability given to what actually happened). Lower is better. It punishes confident mistakes hardest.

### Log5
A rule for combining one side's offence with the other side's defence, both relative to the league average. It comes from baseball.

### Logistic regression
A model that adds up weighted features into log-odds and converts them to a probability. It's simple, fast and exactly explainable.

### Markov chain
A random process that moves between states, where the next step depends only on the current state.

### Monte Carlo
Estimating a probability by simulating a random process many times and counting the outcomes.

### OWP and OOWP
Opponents' win percentage, and opponents' opponents' win percentage. Strength-of-schedule measures borrowed from college sports' RPI.

### Sampling error
How much a score moves by chance because only a few fights were seen. For an accuracy p over n fights, the 95% range is about p ± 1.96·√(p(1−p)/n): ±25 points for one 14-fight card, ±2.5 points for the 1,475-fight test set.

### Shrinkage
Pulling noisy estimates from small samples toward a group average, in proportion to how little data there is. Also called partial pooling or empirical Bayes.

### Validation set
Data held out from training that is used to make modelling choices, kept separate from the test set, which is used only once to report the final score.
