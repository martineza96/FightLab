"""Precomputed data for the app: breakdown charts, model report card, career-stage risk.

Everything here is the notebooks' analysis (05–08) packaged as JSON-ready dicts. The slow
parts are computed once and cached under data/processed/api/, keyed by the data snapshot.
"""
import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from ufc import careers as C
from ufc import form as F
from ufc import modeling as M
from ufc.features import HIST_FEATURES, MIN_FEATURES
from ufc.predict import LOCATION_COLS, _fingerprint, add_location, next_fight_rows

WARN_FEATURES = ["age", "years_active", "prior_fights", "elo_pre", "opp_elo_pre", "below_peak"]


def clean(o):
    """Make pandas / numpy objects JSON-safe (NaN -> None, Timestamps -> ISO strings)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return clean(o.to_dict(orient="records"))
    if isinstance(o, pd.Series):
        return clean(o.to_dict())
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return None if pd.isna(o) else o.isoformat()[:10]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if o is pd.NaT or (not isinstance(o, str) and o is not None and pd.api.types.is_scalar(o) and pd.isna(o)):
        return None
    return o


def _cached(processed, name, build):
    path = Path(processed) / "api" / f"{name}_{_fingerprint(processed)}.json"
    if path.exists():
        return json.loads(path.read_text())
    data = clean(build())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


# ---------------------------------------------------------------------------
# Breakdowns (notebooks 06–07)
# ---------------------------------------------------------------------------
def build_insights(processed="data/processed"):
    p = Path(processed)
    ff = pd.read_parquet(p / "fighter_fights.parquet")
    hist = pd.read_parquet(p / "elo_history.parquet")
    fighters = pd.read_parquet(p / "fighters.parquet")
    perf = C.per_fight_perf(ff, hist)
    car = C.career_table(perf, fighters)

    fa = C.form_by(perf, "age", lo=20, hi=41)
    age_form = [{"age": int(a), "label": "41+" if a >= 41 else str(int(a)), "mean": r["mean"] * 100,
                 "lo": r.lo * 100, "hi": r.hi * 100, "n": int(r["size"])} for a, r in fa.iterrows()]

    q = perf[perf.age.notna()].copy()
    q["band"] = pd.cut(q.age, [0, 25, 28, 31, 34, 37, 40, 60], right=False,
                       labels=["<25", "25–27", "28–30", "31–33", "34–36", "37–39", "40+"])
    q["prior_ko"] = (q.groupby("fighter").ko_loss.cumsum() - q.ko_loss).clip(upper=2)
    grid = q.pivot_table(index="band", columns="prior_ko", values="ko_loss", aggfunc="mean", observed=True)
    ko_by_age = [{"band": str(b), "ko0": grid.loc[b, 0] * 100, "ko1": grid.loc[b, 1] * 100, "ko2": grid.loc[b, 2] * 100}
                 for b in grid.index]
    band_stats = q.groupby("band", observed=True).agg(win=("win", "mean"), ko=("ko_loss", "mean"), n=("ko_loss", "size"))
    by_age_band = [{"band": str(b), "win_rate": r.win * 100, "ko_loss_rate": r.ko * 100, "n": int(r.n)} for b, r in band_stats.iterrows()]

    m = car[car.debut_date >= "2001-01-01"]
    def km_series(groups):
        out = []
        for label, g in groups:
            km = C.kaplan_meier(g.n_fights, g.retired)
            km = km[km.index <= 25]
            out.append({"label": label, "n": int(len(g)), "median": C.km_median(km),
                        "points": [{"t": float(t), "s": r.surv * 100, "lo": r.lo * 100, "hi": r.hi * 100} for t, r in km.iterrows()]})
        return out
    lost = m[m.debut_result == "L"]
    band = pd.cut(m.debut_age, [0, 25, 30, 99], right=False, labels=["debut < 25", "debut 25–29", "debut 30+"])
    retention = {
        "by_debut_result": km_series([("won debut", m[m.debut_result == "W"]), ("lost debut", lost)]),
        "by_debut_loss": km_series([(f"lost by {k}", lost[lost.debut_method == k]) for k in ["DEC", "SUB", "KO/TKO"]]),
        "by_debut_age": km_series([(str(b), m[band == b]) for b in band.cat.categories]),
    }

    fights = ff.drop_duplicates("fight_id")
    yrs = fights[fights.date.dt.year.between(2001, fights.date.max().year - 1)]
    share = pd.crosstab(yrs.date.dt.year, yrs.method_group, normalize="index") * 100
    med_age = perf[perf.date.dt.year.between(2001, fights.date.max().year - 1)].groupby(
        [perf.date.dt.year, "gender"]).age.median().unstack()
    trends = [{"year": int(y), "dec": share.at[y, "DEC"], "ko": share.at[y, "KO/TKO"], "sub": share.at[y, "SUB"],
               "age_men": med_age.at[y, "Men"] if "Men" in med_age else None,
               "age_women": med_age.at[y, "Women"] if "Women" in med_age else None} for y in share.index]

    # form (notebook 07): win rate by last-5 record, and method mix by recent KO wins
    t, _ = M.load_model_table(processed)
    A = M.augment(t)
    five = A[(A.r_l5_fights == 5)]
    w5 = (five.r_l5_win_rate * 5).round()
    g = five.red_win.groupby(w5)
    last5 = [{"wins": int(k), "win_rate": v * 100, "n": int(n)} for (k, v), n in zip(g.mean().items(), g.size())]
    won = A[A.red_win == 1]
    kb = pd.cut(won.r_l5_ko_w, [-1, 0, 1, 99], labels=["0", "1", "2+"])
    mix = pd.crosstab(kb, won.method_group, normalize="index") * 100
    method_by_ko = [{"ko_wins_last5": str(k), "ko": mix.at[k, "KO/TKO"], "sub": mix.at[k, "SUB"], "dec": mix.at[k, "DEC"]}
                    for k in mix.index]
    return {"age_form": age_form, "ko_by_age": ko_by_age, "by_age_band": by_age_band, "retention": retention,
            "trends": trends, "last5": last5, "method_by_ko": method_by_ko,
            "peak_age_median": float(car[car.retired & (car.n_fights >= 8)].peak_age.median())}


# ---------------------------------------------------------------------------
# Model report card (notebooks 07–08)
# ---------------------------------------------------------------------------
def build_model_report(processed="data/processed"):
    t, _ = M.load_model_table(processed)
    ts, _ = M.load_model_table(processed, shrink_k=5)
    p = Path(processed)
    ts = add_location(ts, pd.read_parquet(p / "fighter_fights.parquet"), pd.read_parquet(p / "fights.parquet"),
                      pd.read_parquet(p / "fighter_origin.parquet"))
    base = M.diffs(MIN_FEATURES + HIST_FEATURES) + M.FLAGS + ["red_corner"]
    opp_cols = base + M.diffs(F.OPP_FEATURES)
    final_cols = opp_cols + LOCATION_COLS
    tr, te = t[t.date < M.SPLIT], t[t.date >= M.SPLIT]
    s_tr, s_te = ts[ts.date < M.SPLIT], ts[ts.date >= M.SPLIT].reset_index(drop=True)

    rows = [{"model": "Always pick the red corner", "accuracy": (te.red_win == 1).mean(),
             "log_loss": log_loss(te.red_win, np.full(len(te), tr.red_win.mean())), "roc_auc": 0.5}]
    for name, a, b, cols in [("Elo rating only", tr, te, ["elo_pre_diff", "red_corner"]),
                             ("Career history (notebook 04)", tr, te, base),
                             ("+ shrinkage (notebook 07)", s_tr, s_te, base),
                             ("+ who they beat (notebook 07)", s_tr, s_te, opp_cols),
                             ("+ home crowd (final model, notebook 09)", s_tr, s_te, final_cols)]:
        rows.append({"model": name, **M.scores(b.red_win, M.predict(M.fit(M.logistic(), a, cols), b, cols))})

    final = M.fit(M.logistic(), s_tr, final_cols)
    p = M.predict(final, s_te, final_cols)
    fav = np.maximum(p, 1 - p)
    correct = (p > 0.5) == (s_te.red_win.values == 1)
    bins = pd.cut(fav, [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1.0])
    cal = pd.DataFrame({"pred": fav, "won": correct.astype(float)}).groupby(bins, observed=True).agg(
        pred=("pred", "mean"), won=("won", "mean"), n=("won", "size"))
    calibration = [{"predicted": r.pred * 100, "actual": r.won * 100, "n": int(r.n)} for _, r in cal.iterrows()]

    exp = []
    for b, g in s_te.assign(c=correct).groupby("experience", observed=True):
        exp.append({"bucket": str(b), "accuracy": g.c.mean() * 100, "n": int(len(g))})

    families = {
        "Age, height, reach": ["age_at_fight", "height", "reach"],
        "Striking stats": ["slpm", "sapm", "sig_acc", "sig_def", "kd_per15"],
        "Record & streaks": ["n_fights", "win_pct", "ko_wins", "sub_wins", "dec_wins", "been_finished", "win_streak", "lose_streak"],
        "Grappling stats": ["td_per15", "td_acc", "td_def", "sub_per15", "ctrl_share"],
        "Who they beat": F.OPP_FEATURES,
        "Activity (layoff, cage time)": ["days_since_last", "ufc_minutes"],
        "Home crowd / location": ["home_abroad", "us_event"],
    }
    X = s_te.assign(red_corner=1)[final_cols]
    base_ll = log_loss(s_te.red_win, final.predict_proba(X)[:, 1])
    rng = np.random.default_rng(0)
    fam = []
    for name, feats in families.items():
        cols = [f + "_diff" for f in feats]
        inc = []
        for _ in range(15):
            Xs = X.copy()
            Xs[cols] = X[cols].values[rng.permutation(len(Xs))]
            inc.append(log_loss(s_te.red_win, final.predict_proba(Xs)[:, 1]) - base_ll)
        fam.append({"family": name, "importance": float(np.mean(inc)) * 1000, "sd": float(np.std(inc)) * 1000})
    fam.sort(key=lambda d: -d["importance"])
    return {"test_period": f"{M.SPLIT:%Y-%m-%d} → {s_te.date.max():%Y-%m-%d}", "test_fights": int(len(s_te)),
            "metrics": [{**r, "accuracy": r["accuracy"] * 100} for r in rows],
            "betting_reference": {"label": "Betting favorites (published studies)", "accuracy_low": 66, "accuracy_high": 69},
            "calibration": calibration, "by_experience": exp, "families": fam,
            "method_model": {"base_rates": 1.726, "two_step": 1.619, "exact_outcome_right": 32, "base_exact": 22}}


# ---------------------------------------------------------------------------
# Career-stage risk for each fighter's NEXT fight (notebook 06's early-warning model)
# ---------------------------------------------------------------------------
def build_career_risk(processed="data/processed"):
    p = Path(processed)
    ff0 = pd.read_parquet(p / "fighter_fights.parquet")
    hist = pd.read_parquet(p / "elo_history.parquet")
    fighters = pd.read_parquet(p / "fighters.parquet")
    directory = fighters.drop_duplicates("name").set_index("name")
    perf = C.per_fight_perf(ff0, hist)
    car = C.career_table(perf, fighters)
    w = C.warning_table(perf, car)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from ufc.features import Clipper
    model = make_pipeline(Clipper(), SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(w[WARN_FEATURES], w.bad_stretch.astype(int))
    base_rate = float(w.bad_stretch.mean())

    ff_aug, hist_aug, _ = next_fight_rows(ff0, hist, directory)
    perf_aug = C.per_fight_perf(ff_aug, hist_aug)
    wa = C.warning_table(perf_aug, car, labeled_only=False)
    nxt = wa[wa.fight_id > ff0.fight_id.max()].set_index("fighter")
    risk = pd.Series(model.predict_proba(nxt[WARN_FEATURES])[:, 1], index=nxt.index)
    fa = C.form_by(perf, "age", lo=20, hi=41)["mean"]
    out = {}
    for name, r in nxt.iterrows():
        a = r.age
        out[name] = {"risk": risk[name], "base_rate": base_rate, "age": a, "below_peak": r.below_peak,
                     "peak_elo": car.at[name, "peak_elo"] if name in car.index else None,
                     "age_form": fa.get(min(max(math.floor(a), 20), 41)) * 100 if pd.notna(a) else None}
    return out


class AppData:
    def __init__(self, processed="data/processed"):
        self.processed = processed
        self.insights = _cached(processed, "insights", lambda: build_insights(processed))
        self.model_report = _cached(processed, "model_report", lambda: build_model_report(processed))
        self.career_risk = _cached(processed, "career_risk", lambda: build_career_risk(processed))
