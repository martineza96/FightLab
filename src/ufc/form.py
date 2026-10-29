"""Recent form, opponent quality, strength of schedule and common opponents.

Four layers of "what did this fighter do before tonight?", all built from fights strictly
before the current one:

1. recent_form        the fighter's own last 5 fights, split by result and method
2. opponent_quality   who they beat / lost to, measured as those opponents were *at the time*
3. schedule_strength  opponents' records (OWP) and opponents' opponents' records (OOWP)
4. common_opponents   shared past opponents and previous meetings ("MMA math")

Every function takes the long fighter-fight table (`fighter_fights.parquet`, optionally
with Elo columns merged in) and returns one row per (fight_id, fighter).
"""
import numpy as np
import pandas as pd

SCORE = {"W": 1.0, "L": 0.0, "D": 0.5}          # no contests -> NaN (ignored)
KEY = ["fight_id", "fighter"]


def _sorted(ff):
    return ff.sort_values(["fighter", "date", "fight_id"]).reset_index(drop=True)


def _roll_prior(g, col, n, how="sum"):
    """Rolling stat over the previous n fights (current fight excluded)."""
    return g[col].transform(lambda s: getattr(s.shift(1).rolling(n, min_periods=1), how)())


def _run_length(flags, fighter):
    """Length of the current run of True values entering each fight."""
    f = flags.astype(int)
    block = (f == 0).groupby(fighter).cumsum()
    run_after = f.groupby([fighter, block]).cumsum()
    return run_after.groupby(fighter).shift(1).fillna(0)


# ---------------------------------------------------------------------------
# Layer 1: the fighter's own lead-up
# ---------------------------------------------------------------------------
FORM_FEATURES = ["l5_fights", "l5_win_rate", "l5_ko_w", "l5_sub_w", "l5_dec_w",
                 "l5_ko_l", "l5_sub_l", "l5_dec_l", "finish_streak", "dec_streak",
                 "form_ewm", "l5_avg_min"]


def recent_form(ff, n=5, halflife=2):
    """Last-n record split by method, method streaks, and a recency-weighted score."""
    p = _sorted(ff)
    win, loss = p.result == "W", p.result == "L"
    for m, tag in [("KO/TKO", "ko"), ("SUB", "sub"), ("DEC", "dec")]:
        p[f"_w_{tag}"] = (win & (p.method_group == m)).astype(float)
        p[f"_l_{tag}"] = (loss & (p.method_group == m)).astype(float)
    p["_win"], p["_one"] = win.astype(float), 1.0
    p["_min"] = p.fight_seconds / 60
    p["_score"] = p.result.map(SCORE)
    g = p.groupby("fighter", sort=False)

    out = p[KEY].copy()
    out["l5_fights"] = _roll_prior(g, "_one", n).fillna(0)
    out["l5_win_rate"] = _roll_prior(g, "_win", n) / out.l5_fights.where(out.l5_fights > 0)
    for tag in ["ko", "sub", "dec"]:
        out[f"l5_{tag}_w"] = _roll_prior(g, f"_w_{tag}", n).fillna(0)
        out[f"l5_{tag}_l"] = _roll_prior(g, f"_l_{tag}", n).fillna(0)
    out["finish_streak"] = _run_length(p._w_ko.astype(bool) | p._w_sub.astype(bool), p.fighter)
    out["dec_streak"] = _run_length(p._w_dec.astype(bool), p.fighter)
    out["form_ewm"] = g["_score"].transform(lambda s: s.shift(1).ewm(halflife=halflife, ignore_na=True).mean())
    out["l5_avg_min"] = _roll_prior(g, "_min", n, "mean")
    return out


# ---------------------------------------------------------------------------
# Layer 2: who they beat, as those opponents were at the time
# ---------------------------------------------------------------------------
OPP_FEATURES = ["beat_streak_max", "last_opp_streak", "last_result_w", "quality_wins",
                "quality_losses", "opp_form_avg"]


def opponent_quality(ff, form, n=5):
    """Summaries of the last n opponents, each measured *going into* that fight.

    Needs `opp_elo_pre` in ff (merge elo_history first) and the `recent_form` output.
    beat_streak_max = the longest win streak this fighter snapped in the last n fights.
    last_opp_streak = win streak the most recent opponent was on (whatever the result).
    """
    p = _sorted(ff)
    streak = _win_streaks(p)
    opp_state = p[KEY].assign(_streak=streak.values).merge(form[KEY + ["l5_win_rate"]], on=KEY)
    opp_state = opp_state.rename(columns={"fighter": "opponent", "_streak": "opp_streak",
                                          "l5_win_rate": "opp_l5"})
    p = p.merge(opp_state, on=["fight_id", "opponent"], how="left")
    p = _sorted(p)
    win, loss = p.result == "W", p.result == "L"
    p["_beat_streak"] = p.opp_streak.where(win)
    p["_q_win"] = p.opp_elo_pre.where(win)
    p["_q_loss"] = p.opp_elo_pre.where(loss)
    p["_last_w"] = win.astype(float).where(p.result.isin(["W", "L", "D"]))
    g = p.groupby("fighter", sort=False)

    out = p[KEY].copy()
    out["beat_streak_max"] = _roll_prior(g, "_beat_streak", n, "max")
    out["last_opp_streak"] = g.opp_streak.shift(1)
    out["last_result_w"] = g._last_w.shift(1)
    out["quality_wins"] = _roll_prior(g, "_q_win", n, "mean")
    out["quality_losses"] = _roll_prior(g, "_q_loss", n, "mean")
    out["opp_form_avg"] = _roll_prior(g, "opp_l5", n, "mean")
    return out


def _win_streaks(p):
    """Win streak entering each fight (losses and draws reset it, no contests don't)."""
    def one(res):
        w, out = 0, []
        for r in res:
            out.append(w)
            if r == "W":
                w += 1
            elif r in ("L", "D"):
                w = 0
        return out
    return p.groupby("fighter", sort=False).result.transform(lambda s: pd.Series(one(s.values), index=s.index))


# ---------------------------------------------------------------------------
# Layer 3: strength of schedule (OWP / OOWP)
# ---------------------------------------------------------------------------
SOS_FEATURES = ["owp", "oowp"]


def prior_pairs(ff):
    """Every (current fight, earlier opponent) pair for each fighter.

    Columns: fight_id, fighter, date (of the current fight), opp (an earlier opponent),
    opp_fight_id (when they met), score (fighter's result vs opp: 1 / 0 / 0.5).
    """
    p = _sorted(ff)[["fight_id", "fighter", "date", "opponent", "result"]]
    p = p.assign(i=p.groupby("fighter").cumcount())
    m = p.merge(p, on="fighter", suffixes=("", "_prev"))
    m = m[m.i_prev < m.i]
    return pd.DataFrame({"fight_id": m.fight_id.values, "fighter": m.fighter.values, "date": m.date.values,
                         "opp": m.opponent_prev.values, "opp_fight_id": m.fight_id_prev.values,
                         "score": m.result_prev.map(SCORE).values})


def _record_timeline(ff):
    """Each fighter's cumulative wins / decided fights *after* every fight, by date."""
    p = _sorted(ff)
    dec = p.result.isin(["W", "L", "D"]).astype(float)
    t = pd.DataFrame({"fighter": p.fighter, "date": p.date,
                      "cum_w": (p.result == "W").astype(float).groupby(p.fighter).cumsum(),
                      "cum_n": dec.groupby(p.fighter).cumsum()})
    return t.groupby(["fighter", "date"], as_index=False).last().sort_values("date")


def _asof_rate(pairs, timeline, key="opp"):
    """Win rate of `key` using only their fights strictly before `date`."""
    q = pairs[[key, "date"]].reset_index().sort_values("date")
    tl = timeline.rename(columns={"fighter": key})
    m = pd.merge_asof(q, tl, on="date", by=key, allow_exact_matches=False)
    rate = (m.cum_w / m.cum_n).where(m.cum_n > 0)
    return pd.Series(rate.values, index=m["index"]).reindex(pairs.index)


def schedule_strength(ff, pairs=None):
    """OWP = average win rate of past opponents, as of tonight (their fights before today).

    OOWP = average OWP of past opponents, taken from each opponent's most recent fight
    before tonight (their opponents' records as of that fight). One level of recursion,
    the same idea as the RPI ratings used in college sports.
    """
    pairs = prior_pairs(ff) if pairs is None else pairs
    tl = _record_timeline(ff)
    pairs = pairs.assign(opp_rate=_asof_rate(pairs, tl))
    owp = pairs.groupby(KEY).opp_rate.mean().rename("owp").reset_index()

    # OOWP: each opponent's OWP at their latest fight before tonight
    own = _sorted(ff)[KEY + ["date"]].merge(owp, on=KEY, how="left")
    own = own.dropna(subset=["owp"]).sort_values("date").rename(columns={"fighter": "opp", "owp": "opp_owp"})
    q = pairs[["opp", "date"]].reset_index().sort_values("date")
    m = pd.merge_asof(q, own[["opp", "date", "opp_owp"]], on="date", by="opp", allow_exact_matches=False)
    pairs["opp_owp"] = pd.Series(m.opp_owp.values, index=m["index"]).reindex(pairs.index)
    oowp = pairs.groupby(KEY).opp_owp.mean().rename("oowp").reset_index()

    out = _sorted(ff)[KEY].merge(owp, on=KEY, how="left").merge(oowp, on=KEY, how="left")
    return out


# ---------------------------------------------------------------------------
# Layer 4: common opponents and previous meetings
# ---------------------------------------------------------------------------
COMMON_FEATURES = ["n_common", "common_net", "chain_wins", "rematch_net"]


def common_opponents(ff, pairs=None):
    """For each fight, compare both fighters' results against shared earlier opponents.

    common_net  = mean over shared opponents C of (my score vs C) − (their score vs C)
    chain_wins  = number of C with "I beat C and C beat them" minus the reverse
    rematch_net = my wins − losses against tonight's opponent in earlier meetings
    All are from this fighter's point of view, so the two corners are mirror images.
    """
    pairs = prior_pairs(ff) if pairs is None else pairs
    p = _sorted(ff)[KEY + ["opponent"]]
    mine = pairs.merge(p, on=KEY)                                   # adds tonight's opponent
    rematch = mine[mine.opp == mine.opponent]
    rematch_net = (rematch.score * 2 - 1).groupby([rematch.fight_id, rematch.fighter]).sum().rename("rematch_net")

    # each fighter's average score vs every earlier opponent (some opponents were met twice)
    per = mine.groupby(KEY + ["opponent", "opp"], as_index=False).score.mean()
    theirs = per.rename(columns={"fighter": "opponent", "opponent": "fighter", "score": "their_score"})
    both = per.merge(theirs, on=["fight_id", "fighter", "opponent", "opp"])
    both = both[(both.opp != both.fighter) & (both.opp != both.opponent)]
    g = both.assign(diff=both.score - both.their_score,
                    chain=(both.score == 1).astype(int) * (both.their_score == 0) -
                          (both.score == 0).astype(int) * (both.their_score == 1)).groupby(KEY)
    common = pd.DataFrame({"n_common": g.size(), "common_net": g["diff"].mean(), "chain_wins": g.chain.sum()})

    out = p[KEY].merge(common.reset_index(), on=KEY, how="left").merge(rematch_net.reset_index(), on=KEY, how="left")
    out["n_common"] = out.n_common.fillna(0)
    out["chain_wins"] = out.chain_wins.fillna(0)
    out["rematch_net"] = out.rematch_net.fillna(0)
    return out


# ---------------------------------------------------------------------------
# Style + physical
# ---------------------------------------------------------------------------
STYLE_FEATURES = ["dist_share", "clinch_share", "ground_share", "reach_height", "southpaw"]


def style_physical(ff):
    """Where a fighter lands strikes (prior fights only), plus reach-to-height and stance."""
    p = _sorted(ff)
    g = p.fighter
    out = p[KEY].copy()
    landed = p.sig_str.fillna(0)
    prior_landed = landed.groupby(g).cumsum() - landed
    for col, name in [("distance", "dist_share"), ("clinch", "clinch_share"), ("ground", "ground_share")]:
        v = p[col].fillna(0)
        out[name] = ((v.groupby(g).cumsum() - v) / prior_landed).where(prior_landed >= 10)
    out["reach_height"] = p.reach / p.height
    out["southpaw"] = p.stance.isin(["Southpaw"]).astype(float).where(p.stance.notna())
    return out


# ---------------------------------------------------------------------------
# Everything together
# ---------------------------------------------------------------------------
ALL_NEW = FORM_FEATURES + OPP_FEATURES + SOS_FEATURES + COMMON_FEATURES + STYLE_FEATURES + ["elo_pre"]


def build_all(ff, hist):
    """Long table with every new feature, keyed by (fight_id, fighter)."""
    e = ff.merge(hist[KEY + ["elo_pre", "opp_elo_pre"]], on=KEY, how="left")
    pairs = prior_pairs(e)
    form = recent_form(e)
    parts = [form, opponent_quality(e, form), schedule_strength(e, pairs), common_opponents(e, pairs),
             style_physical(e), e[KEY + ["elo_pre"]]]
    out = parts[0]
    for x in parts[1:]:
        out = out.merge(x, on=KEY, how="left")
    return out


def check_no_lookahead_form(ff, hist, feats, n_checks=120, seed=0):
    """Recompute a sample of rows by brute force from strictly earlier fights."""
    e = _sorted(ff.merge(hist[KEY + ["elo_pre", "opp_elo_pre"]], on=KEY, how="left"))
    e["i"] = e.groupby("fighter").cumcount()
    f = feats.set_index(KEY)
    rng = np.random.default_rng(seed)
    for idx in rng.choice(len(e), n_checks, replace=False):
        row = e.iloc[idx]
        past = e[(e.fighter == row.fighter) & (e.i < row.i)]
        last5 = past.tail(5)
        got = f.loc[(row.fight_id, row.fighter)]
        assert got.l5_fights == len(last5)
        ko = ((last5.result == "W") & (last5.method_group == "KO/TKO")).sum()
        assert got.l5_ko_w == ko
        qw = last5[last5.result == "W"].opp_elo_pre.mean()
        assert (np.isnan(qw) and np.isnan(got.quality_wins)) or np.isclose(qw, got.quality_wins)
        # OWP: earlier opponents' win rates using only their fights before tonight
        rates = []
        for opp in past.opponent:
            o = e[(e.fighter == opp) & (e.date < row.date) & e.result.isin(["W", "L", "D"])]
            rates.append((o.result == "W").mean() if len(o) else np.nan)
        owp = np.nanmean(rates) if len(rates) and not np.all(np.isnan(rates)) else np.nan
        assert (np.isnan(owp) and np.isnan(got.owp)) or np.isclose(owp, got.owp)
        assert (past.date <= row.date).all()
    return True
