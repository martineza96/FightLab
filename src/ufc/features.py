"""Pre-fight features built ONLY from information available before the opening bell.

Core idea: for every (fighter, fight) row in the long table, aggregate that fighter's
earlier fights (strictly smaller fight_id). Cumulative sum minus the current row gives
"everything before this fight" without any look-ahead.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Minimum features: who is fighting, in what kind of bout, and very basic record/physicals.
MIN_FEATURES = ["age_at_fight", "height", "reach", "n_fights", "win_pct"]

# First historical layer: career-to-date style and outcome profile.
HIST_FEATURES = ["slpm", "sapm", "sig_acc", "sig_def", "td_per15", "td_acc", "td_def",
                 "sub_per15", "kd_per15", "ctrl_share", "ko_wins", "sub_wins", "dec_wins",
                 "been_finished", "win_streak", "lose_streak", "days_since_last",
                 "ufc_minutes"]

CONTEXT = ["weight_class", "gender", "is_title_bout", "total_rounds"]


def _prior_sum(ff, col):
    """Sum of `col` over the fighter's earlier fights (current fight excluded)."""
    v = ff[col].fillna(0)
    return v.groupby(ff["fighter"]).cumsum() - v


def _prior_ratio(ff, num, den, scale=1.0):
    """Prior sum(num) / prior sum(den), using only fights where both are recorded."""
    ok = ff[num].notna() & ff[den].notna()
    n = ff[num].where(ok, 0)
    d = ff[den].where(ok, 0)
    g = ff["fighter"]
    pn = n.groupby(g).cumsum() - n
    pd_ = d.groupby(g).cumsum() - d
    return (scale * pn / pd_).where(pd_ > 0)


def _streaks(results):
    """Win/lose streak entering each fight (draws reset both, no-contests are ignored)."""
    w = l = 0
    out_w, out_l = [], []
    for r in results:
        out_w.append(w)
        out_l.append(l)
        if r == "W":
            w, l = w + 1, 0
        elif r == "L":
            w, l = 0, l + 1
        elif r == "D":
            w, l = 0, 0
        # "N" (no contest) leaves both streaks unchanged
    return out_w, out_l


def add_history(ff):
    """Add prior-fight features to the long fighter-fight table."""
    ff = ff.sort_values(["fighter", "fight_id"]).reset_index(drop=True)
    g = ff.groupby("fighter")
    ff["minutes"] = ff["fight_seconds"] / 60
    for r in "WLD":
        ff[f"is_{r}"] = (ff["result"] == r).astype(int)
    win = ff["result"] == "W"
    loss = ff["result"] == "L"
    ff["is_ko_win"] = (win & (ff.method_group == "KO/TKO")).astype(int)
    ff["is_sub_win"] = (win & (ff.method_group == "SUB")).astype(int)
    ff["is_dec_win"] = (win & (ff.method_group == "DEC")).astype(int)
    ff["is_finished"] = (loss & ff.method_group.isin(["KO/TKO", "SUB"])).astype(int)

    # record
    ff["n_fights"] = g.cumcount()
    ff["wins"] = _prior_sum(ff, "is_W")
    ff["losses"] = _prior_sum(ff, "is_L")
    ff["win_pct"] = (ff["wins"] / ff["n_fights"]).where(ff["n_fights"] > 0)
    ff["ko_wins"] = _prior_sum(ff, "is_ko_win")
    ff["sub_wins"] = _prior_sum(ff, "is_sub_win")
    ff["dec_wins"] = _prior_sum(ff, "is_dec_win")
    ff["been_finished"] = _prior_sum(ff, "is_finished")
    ff["ufc_minutes"] = _prior_sum(ff, "minutes")
    ff["days_since_last"] = g["date"].diff().dt.days
    ws, ls = [], []
    for _, res in g["result"]:
        w, l = _streaks(res)
        ws += w
        ls += l
    ff["win_streak"], ff["lose_streak"] = ws, ls

    # style rates (same definitions ufcstats uses, but computed only from PAST fights)
    ff["slpm"] = _prior_ratio(ff, "sig_str", "minutes")
    ff["sapm"] = _prior_ratio(ff, "opp_sig_str", "minutes")
    ff["sig_acc"] = _prior_ratio(ff, "sig_str", "sig_str_att")
    ff["sig_def"] = 1 - _prior_ratio(ff, "opp_sig_str", "opp_sig_str_att")
    ff["td_per15"] = _prior_ratio(ff, "td", "minutes", 15)
    ff["td_acc"] = _prior_ratio(ff, "td", "td_att")
    ff["td_def"] = 1 - _prior_ratio(ff, "opp_td", "opp_td_att")
    ff["sub_per15"] = _prior_ratio(ff, "sub_att", "minutes", 15)
    ff["kd_per15"] = _prior_ratio(ff, "kd", "minutes", 15)
    ff["ctrl_share"] = _prior_ratio(ff, "ctrl_sec", "fight_seconds")

    return ff.sort_values(["fight_id", "corner"]).reset_index(drop=True)


def build_model_table(fights, ff_hist, reliable_corners_only=True, extra=()):
    """One row per decided fight: context + red_/blue_ features + red-minus-blue diffs.

    By default drops pre-2010-03 fights, where the red slot is always the winner
    (see cleaning.CORNERS_RELIABLE_FROM). Their history still feeds later fights.
    `extra` names additional per-fighter columns already present in ff_hist.
    """
    feats = MIN_FEATURES + HIST_FEATURES + [f for f in extra if f not in MIN_FEATURES + HIST_FEATURES]
    side = ff_hist.set_index(["fight_id", "corner"])[feats + ["stance"]]
    red = side.xs("Red", level="corner").add_prefix("r_")
    blue = side.xs("Blue", level="corner").add_prefix("b_")

    t = fights.set_index("fight_id")[["date", "event", "r_fighter", "b_fighter", "winner",
                                      "red_win", "corners_reliable"] + CONTEXT].join(red).join(blue)
    t = pd.concat([t, pd.DataFrame({f"{f}_diff": t[f"r_{f}"] - t[f"b_{f}"] for f in feats}, index=t.index)], axis=1).copy()
    # Missing-ness is information (debutants have no history, old-era fighters lack reach).
    t["r_debut"] = (t["r_n_fights"] == 0).astype(int)
    t["b_debut"] = (t["b_n_fights"] == 0).astype(int)
    t["reach_missing"] = (t["r_reach"].isna() | t["b_reach"].isna()).astype(int)
    t["stance_matchup"] = t["r_stance"].fillna("Unknown") + "_v_" + t["b_stance"].fillna("Unknown")
    t = t[t["winner"].isin(["Red", "Blue"])]  # drop draws and no-contests
    if reliable_corners_only:
        t = t[t["corners_reliable"]]
    t = t.reset_index().copy()
    t["red_win"] = t["red_win"].astype(int)
    return t


def swap_corners(t):
    """Return a copy with red/blue swapped and target flipped (for symmetric training)."""
    # rename r_x <-> b_x in one go (instead of column-by-column assignment)
    pairs = {c: "b_" + c[2:] for c in t.columns if c.startswith("r_") and "b_" + c[2:] in t.columns}
    mapping = {**pairs, **{b: r for r, b in pairs.items()}}
    s = t.rename(columns=mapping)[list(t.columns)].copy()
    diff_cols = [c for c in t.columns if c.endswith("_diff")]
    s[diff_cols] = -t[diff_cols]
    s["red_win"] = 1 - t["red_win"]
    s["winner"] = t["winner"].map({"Red": "Blue", "Blue": "Red"})
    if "stance_matchup" in t.columns:
        s["stance_matchup"] = s["r_stance"].fillna("Unknown") + "_v_" + s["b_stance"].fillna("Unknown")
    return s


def check_no_lookahead(ff_hist, n_checks=200, seed=0):
    """Recompute a few rows by brute force from strictly-earlier fights and compare."""
    rng = np.random.default_rng(seed)
    for i in rng.choice(len(ff_hist), n_checks, replace=False):
        row = ff_hist.iloc[i]
        past = ff_hist[(ff_hist.fighter == row.fighter) & (ff_hist.fight_id < row.fight_id)]
        assert row.n_fights == len(past)
        assert row.wins == (past.result == "W").sum()
        if len(past):
            assert (past.date <= row.date).all()
    return True


class Clipper(BaseEstimator, TransformerMixin):
    """Clip each column to quantiles learned on the training data (winsorizing).

    Fighters with one very short fight get extreme rates (e.g. 80 strikes/min);
    clipping keeps them from dominating a linear model. NaNs pass through.
    """

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lo_ = X.quantile(self.lower)
        self.hi_ = X.quantile(self.upper)
        return self

    def transform(self, X):
        return pd.DataFrame(X).clip(self.lo_, self.hi_, axis=1)
