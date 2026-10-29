"""Career arcs: retention (survival), aging curves, and early signs of decline.

Everything here is built from `fighter_fights` (one row per fighter per fight) and
`elo_history`. Only who won matters for careers, so the pre-2010 red-corner quirk
doesn't affect anything in this module and all fights back to 1994 are used.
"""
import numpy as np
import pandas as pd

from ufc import elo

RETIRED_AFTER_DAYS = 548      # no UFC fight for 18 months = career over (or left the UFC)
WIN_SCORE = {"W": 1.0, "L": 0.0, "D": 0.5}


# ---------------------------------------------------------------------------
# Per-fight table
# ---------------------------------------------------------------------------
def per_fight_perf(ff, hist):
    """One row per fighter per fight, in career order, with performance signals.

    Signals (all describe *this* fight, so they're only usable as history for later fights):
      elo_surprise   result minus the Elo-expected win probability (+ = beat expectations)
      sig_diff_pm    significant strikes landed minus absorbed, per minute
      head_abs_pm    head strikes absorbed per minute
      str_def        share of the opponent's significant strikes that missed
      td_def         share of the opponent's takedown attempts that failed (NaN if none)
      kd_received    times knocked down
    """
    cols = ["fight_id", "fighter", "elo_pre", "elo_post", "opp_elo_pre"]
    p = ff.merge(hist[cols], on=["fight_id", "fighter"], how="left")
    p = p.sort_values(["fighter", "date", "fight_id"]).reset_index(drop=True)

    g = p.groupby("fighter", sort=False)
    p["fight_no"] = g.cumcount() + 1
    p["debut_date"] = g.date.transform("min")
    p["years_active"] = (p.date - p.debut_date).dt.days / 365.25
    p["age"] = p.age_at_fight

    p["score"] = p.result.map(WIN_SCORE)                      # NaN for no contests
    p["win"] = p.result.map({"W": 1.0, "L": 0.0})             # draws/NC excluded
    p["ko_loss"] = ((p.result == "L") & (p.method_group == "KO/TKO")).astype(int)
    p["elo_surprise"] = p.score - elo.expected(p.elo_pre, p.opp_elo_pre)

    minutes = p.fight_seconds / 60
    p["minutes"] = minutes
    p["sig_diff_pm"] = (p.sig_str - p.opp_sig_str) / minutes
    p["head_abs_pm"] = p.opp_head / minutes
    p["str_def"] = 1 - p.opp_sig_str / p.opp_sig_str_att.where(p.opp_sig_str_att > 0)
    p["td_def"] = 1 - p.opp_td / p.opp_td_att.where(p.opp_td_att > 0)
    p["kd_received"] = p.opp_kd
    return p


# ---------------------------------------------------------------------------
# Career table + survival
# ---------------------------------------------------------------------------
def career_table(perf, fighters, data_end=None):
    """One row per fighter: debut, length, whether the career has ended, and the Elo peak.

    `retired` = no UFC fight in the last 18 months of data. Careers still going are
    *censored*: we know they lasted at least `n_fights`, not how long they'll last.
    """
    data_end = pd.Timestamp(data_end) if data_end is not None else perf.date.max()
    g = perf.groupby("fighter", sort=False)
    first = g.head(1).set_index("fighter")
    last = g.tail(1).set_index("fighter")

    c = pd.DataFrame({
        "gender": first.gender,
        "weight_class": g.weight_class.agg(lambda s: s.mode().iat[0]),
        "debut_date": first.date,
        "debut_age": first.age,
        "debut_result": first.result,
        "debut_method": first.method_group,
        "last_date": last.date,
        "last_age": last.age,
        "last_result": last.result,
        "n_fights": g.size(),
        "wins": g.win.sum(),
        "ko_losses": g.ko_loss.sum(),
    })
    c["years"] = (c.last_date - c.debut_date).dt.days / 365.25
    c["retired"] = (data_end - c.last_date).dt.days > RETIRED_AFTER_DAYS

    # Elo peak: the highest post-fight rating (ties -> first time it was reached)
    i_peak = perf.groupby("fighter", sort=False).elo_post.idxmax()
    pk = perf.loc[i_peak].set_index("fighter")
    c["peak_elo"] = pk.elo_post
    c["peak_date"] = pk.date
    c["peak_age"] = pk.age
    c["peak_fight_no"] = pk.fight_no
    after = perf.merge(pk[["fight_no"]].rename(columns={"fight_no": "peak_no"}),
                       left_on="fighter", right_index=True)
    after = after[after.fight_no > after.peak_no]
    c["fights_after_peak"] = after.groupby("fighter").size().reindex(c.index).fillna(0).astype(int)
    c["win_rate_after_peak"] = after.groupby("fighter").win.mean().reindex(c.index)

    dob = fighters.drop_duplicates("name").set_index("name").dob
    c["dob"] = dob.reindex(c.index)
    return c


def kaplan_meier(durations, events, z=1.96):
    """Kaplan-Meier survival curve with a Greenwood confidence band.

    durations: how long each career lasted (e.g. number of UFC fights)
    events:    True if the career ended, False if it's still going (censored)
    Returns a DataFrame indexed by time t with S(t) = P(career lasts longer than t).
    """
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=bool)
    times = np.unique(d[e])
    at_risk = np.array([(d >= t).sum() for t in times])
    ended = np.array([((d == t) & e).sum() for t in times])
    surv = np.cumprod(1 - ended / at_risk)
    with np.errstate(divide="ignore", invalid="ignore"):
        gw = np.cumsum(ended / (at_risk * (at_risk - ended)))
        se = np.nan_to_num(surv * np.sqrt(gw))
    out = pd.DataFrame({"at_risk": at_risk, "ended": ended, "surv": surv,
                        "lo": np.clip(surv - z * se, 0, 1), "hi": np.clip(surv + z * se, 0, 1)},
                       index=pd.Index(times, name="t"))
    # start the curve at S(0) = 1
    return pd.concat([pd.DataFrame({"at_risk": len(d), "ended": 0, "surv": 1.0, "lo": 1.0, "hi": 1.0},
                                   index=pd.Index([0.0], name="t")), out])


def km_median(km):
    """Smallest t where survival drops to 50% or below (NaN if it never does)."""
    hit = km.index[km.surv <= 0.5]
    return float(hit[0]) if len(hit) else np.nan


def km_restricted_mean(km, horizon):
    """Area under the survival curve up to `horizon` = expected career length capped at horizon."""
    s = km.surv[km.index <= horizon]
    t = np.append(s.index.values, horizon)
    return float(np.sum(np.diff(t) * s.values))


# ---------------------------------------------------------------------------
# Aging curves (delta method)
# ---------------------------------------------------------------------------
METRICS = {
    # name: (label, higher_is_better)
    "win": ("win rate", True),
    "elo_surprise": ("results vs Elo expectation", True),
    "sig_diff_pm": ("sig. strike differential / min", True),
    "head_abs_pm": ("head strikes absorbed / min", False),
    "str_def": ("striking defense", True),
    "td_def": ("takedown defense", True),
    "kd_received": ("knockdowns received / fight", False),
}


def _bin(perf, by):
    if by == "age":
        return np.floor(perf.age)
    if by == "years_active":
        return np.floor(perf.years_active)
    if by == "fight_no":
        return (perf.fight_no - 1) // 3                   # blocks of 3 fights: 0 = fights 1-3, 1 = 4-6, ...
    raise ValueError(by)


def form_by(perf, by="age", lo=None, hi=None, min_fights=60):
    """Average result vs Elo expectation in each bin, with a 95% interval.

    Elo already tracks each fighter's current level, so this is a *within-fighter* rate of
    change: positive = fighters in this bin are improving faster than their rating
    follows, negative = their rating overstates them (they're declining). ×64 converts it
    to roughly how many Elo points per fight the rating is out of date.
    """
    b = _bin(perf, by)
    if lo is not None or hi is not None:
        b = b.clip(lo, hi)
    s = perf.groupby(b).elo_surprise.agg(["mean", "sem", "size"])
    s = s[s["size"] >= min_fights]
    s["lo"], s["hi"] = s["mean"] - 1.96 * s["sem"], s["mean"] + 1.96 * s["sem"]
    if by == "fight_no":
        s.index = s.index * 3 + 1                            # label blocks by their first fight
    s.index.name = by
    return s


def fighter_bins(perf, by="age"):
    """Aggregate each fighter's fights within one bin (e.g. one year of age).

    Rates are pooled (total strikes / total minutes) so a 30-second fight doesn't count
    as much as a 25-minute one.
    """
    p = perf.assign(bin=_bin(perf, by)).dropna(subset=["bin"])
    p = p.assign(sig_net=p.sig_str - p.opp_sig_str,
                 opp_td_att_=p.opp_td_att.where(p.opp_td_att > 0))
    g = p.groupby(["fighter", "bin"])
    s = g[["minutes", "sig_net", "opp_head", "opp_sig_str", "opp_sig_str_att",
           "opp_td", "opp_td_att_", "kd_received"]].sum(min_count=1)
    out = pd.DataFrame({
        "n": g.size(),
        "win": g.win.mean(),
        "elo_surprise": g.elo_surprise.mean(),
        "sig_diff_pm": s.sig_net / s.minutes,
        "head_abs_pm": s.opp_head / s.minutes,
        "str_def": 1 - s.opp_sig_str / s.opp_sig_str_att.where(s.opp_sig_str_att > 0),
        "td_def": 1 - s.opp_td / s.opp_td_att_,
        "kd_received": s.kd_received / g.size(),
    })
    return out.reset_index()


def _pairs(fb, metric, shrink):
    """Same fighter, consecutive bins (b, b+1): the change and its weight.

    Fighters who had a lucky year are the ones who get to fight the next year, so the raw
    change is biased downward (regression to the mean). With `shrink`, the starting value
    is first pulled toward the bin average by its reliability r (the year-to-year
    correlation), which removes most of that luck before taking the difference.
    """
    a = fb[["fighter", "bin", "n", metric]]
    b = a.assign(bin=a.bin - 1)
    m = a.merge(b, on=["fighter", "bin"], suffixes=("", "_next")).dropna(subset=[metric, metric + "_next"])
    m["to_bin"] = m.bin + 1
    m["w"] = 2 / (1 / m.n + 1 / m.n_next)                    # harmonic mean of fight counts
    start = m[metric]
    if shrink:
        mu = fb.groupby("bin").apply(lambda d: np.average(d[metric].dropna(), weights=d.n[d[metric].notna()])
                                     if d[metric].notna().any() else np.nan, include_groups=False)
        r = np.clip(np.corrcoef(m[metric], m[metric + "_next"])[0, 1], 0, 1)
        start = m.bin.map(mu) + r * (m[metric] - m.bin.map(mu))
    m["delta"] = m[metric + "_next"] - start
    return m[["fighter", "to_bin", "delta", "w"]]


def _curve(pairs, bins):
    agg = pairs.assign(wd=pairs.delta * pairs.w).groupby("to_bin")[["wd", "w"]].sum()
    step = (agg.wd / agg.w).reindex(bins[1:])
    return pd.Series(np.concatenate([[0.0], step.cumsum().values]), index=bins)


def aging_curve(perf, metric="sig_diff_pm", by="age", shrink=True, min_pairs=40,
                n_boot=200, seed=0, lo=None, hi=None):
    """Within-fighter aging curve via the delta method (from baseball aging research).

    For every fighter who fought in two consecutive bins (e.g. at age 30 and 31) we take
    the change in their own numbers. Averaging those changes compares each fighter only
    with themselves, so fighters who are simply better (and therefore last longer)
    don't distort the curve the way a plain average by age does.

    Returns DataFrame indexed by bin: curve (change relative to the curve's best bin),
    lo/hi (bootstrap 90% band, resampling fighters), pairs (number of fighter pairs into
    the bin), and naive (plain average by bin, also shifted so its best bin is 0).
    """
    fb = fighter_bins(perf, by)
    pairs = _pairs(fb, metric, shrink)
    counts = pairs.groupby("to_bin").size()
    ok = counts[counts >= min_pairs].index
    if lo is not None:
        ok = ok[ok >= lo]
    if hi is not None:
        ok = ok[ok <= hi]
    ok = np.array(sorted(ok))
    runs = np.split(ok, np.where(np.diff(ok) != 1)[0] + 1)   # keep the longest consecutive run
    run = max(runs, key=len)
    bins = np.concatenate([[run[0] - 1], run])
    pairs = pairs[pairs.to_bin.isin(run)]

    higher = METRICS[metric][1]
    best = (lambda s: s.max()) if higher else (lambda s: s.min())

    curve = _curve(pairs, bins)
    rng = np.random.default_rng(seed)
    ids = pairs.fighter.unique()
    by_f = dict(tuple(pairs.groupby("fighter")))
    boots = []
    for _ in range(n_boot):
        bp = pd.concat([by_f[f] for f in rng.choice(ids, len(ids), replace=True)], ignore_index=True)
        c = _curve(bp, bins)
        boots.append(c - best(c))
    boots = pd.concat(boots, axis=1) if boots else pd.DataFrame(index=bins)

    nv = fb[fb.bin.isin(bins)].dropna(subset=[metric])
    naive = nv.groupby("bin").apply(lambda d: np.average(d[metric], weights=d.n), include_groups=False)
    naive = naive.reindex(bins)

    out = pd.DataFrame({
        "curve": curve - best(curve),
        "lo": boots.quantile(0.05, axis=1) if n_boot else np.nan,
        "hi": boots.quantile(0.95, axis=1) if n_boot else np.nan,
        "pairs": counts.reindex(bins).fillna(0).astype(int),
        "naive": naive - best(naive),
    })
    if by == "fight_no":
        out.index = out.index * 3 + 1                        # label blocks by their first fight
    out.index.name = by
    return out


# ---------------------------------------------------------------------------
# Early-warning features + label
# ---------------------------------------------------------------------------
WARN_FEATURES = [
    "age", "years_active", "prior_fights", "elo_pre", "opp_elo_pre", "below_peak",
    "prior_ko_losses", "ko_loss_last2", "loss_streak", "head_abs_cum",
    "head_abs_pm_last3", "sig_diff_trend", "str_def_trend", "days_since_last",
]


def warning_table(perf, careers, min_prior=5, window=3, labeled_only=True):
    """One row per fight by a veteran, with only pre-fight features, plus the label.

    Label `bad_stretch`: in this fight and the next (window-1) fights the fighter loses
    at least 2, *or* the career ends with a loss before the window fills.
    Rows whose window is still open (career ongoing, not enough fights yet) are dropped,
    unless labeled_only=False (useful for scoring a fighter's latest fights).
    """
    p = perf.copy()
    g = p.groupby("fighter", sort=False)
    prev = lambda col: g[col].shift(1)

    p["prior_fights"] = p.fight_no - 1
    p["prior_peak"] = g.elo_post.cummax().groupby(p.fighter).shift(1)
    p["below_peak"] = (p.prior_peak - p.elo_pre).clip(lower=0)
    p["prior_ko_losses"] = g.ko_loss.cumsum() - p.ko_loss
    p["ko_loss_last2"] = prev("ko_loss").fillna(0) + g.ko_loss.shift(2).fillna(0)
    p["head_abs_cum"] = g.opp_head.cumsum() - p.opp_head.fillna(0)
    p["days_since_last"] = (p.date - prev("date")).dt.days

    # losing streak going into the fight
    is_loss = (p.result == "L").astype(int)
    block = (is_loss == 0).groupby(p.fighter).cumsum()
    streak_after = is_loss.groupby([p.fighter, block]).cumsum()
    p["loss_streak"] = streak_after.groupby(p.fighter).shift(1).fillna(0)

    # recent form vs career form, using prior fights only
    def rolling_prior(col, n):
        return g[col].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
    def career_prior(col):
        return g[col].transform(lambda s: s.shift(1).expanding().mean())
    p["head_abs_pm_last3"] = rolling_prior("head_abs_pm", 3)
    p["sig_diff_trend"] = rolling_prior("sig_diff_pm", 3) - career_prior("sig_diff_pm")
    p["str_def_trend"] = rolling_prior("str_def", 3) - career_prior("str_def")

    # label: losses in this fight + the next (window-1)
    loss = (p.result == "L").astype(float)
    fwd = sum(g.result.shift(-k).eq("L").astype(float) for k in range(1, window))
    n_left = g.fight_no.transform("max") - p.fight_no + 1          # fights from here to the end
    p["losses_window"] = loss + fwd
    retired = p.fighter.map(careers.retired)
    last_result = p.fighter.map(careers.last_result)
    full = n_left >= window
    p["bad_stretch"] = np.where(full, p.losses_window >= 2,
                                np.where(retired, (last_result == "L") | (p.losses_window >= 2), np.nan))
    # date the window closes (for a clean time split)
    p["label_date"] = g.date.shift(-(window - 1)).fillna(g.date.transform("max"))

    keep = (p.prior_fights >= min_prior) & (p.result != "N")
    if labeled_only:
        keep &= p.bad_stretch.notna()
    return p.loc[keep].reset_index(drop=True)


def check_warning_no_lookahead(perf, table, n_checks=150, seed=0):
    """Recompute a few rows' features by brute force from strictly earlier fights."""
    rng = np.random.default_rng(seed)
    for i in rng.choice(len(table), min(n_checks, len(table)), replace=False):
        row = table.iloc[i]
        past = perf[(perf.fighter == row.fighter) & (perf.fight_no < row.fight_no)]
        assert row.prior_fights == len(past)
        assert row.prior_ko_losses == past.ko_loss.sum()
        assert np.isclose(row.head_abs_cum, past.opp_head.sum())
        assert np.isclose(row.below_peak, max(past.elo_post.max() - row.elo_pre, 0))
        assert (past.date <= row.date).all()
    return True


# ---------------------------------------------------------------------------
# Two-phase (rise then fall) fit
# ---------------------------------------------------------------------------
def two_phase_fit(x, y, min_side=3):
    """Fit y = a + b·x + c·max(0, x − k) for every candidate breakpoint k; keep the best.

    Returns dict(k, slope_before, slope_after, sse, gain) where gain is how much the hinge
    reduces squared error vs a single straight line (0 = no evidence of a turn).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X1 = np.column_stack([np.ones_like(x), x])
    sse_line = np.sum((y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]) ** 2)
    best = None
    for k in x[min_side - 1:len(x) - min_side]:
        X = np.column_stack([X1, np.maximum(0, x - k)])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        sse = np.sum((y - X @ coef) ** 2)
        if best is None or sse < best["sse"]:
            best = {"k": k, "slope_before": coef[1], "slope_after": coef[1] + coef[2], "sse": sse}
    if best is None:
        return None
    best["gain"] = 1 - best["sse"] / sse_line if sse_line > 0 else 0.0
    return best
