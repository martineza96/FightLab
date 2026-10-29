"""Markov-chain fight simulator, after Holmes, McHale & Zychaluk (2023, Int. J. Forecasting).

A fight is a chain of 5-second steps. Each step the fight is in one of four positions:

    0 distance · 1 clinch · 2 A on top (ground) · 3 B on top (ground)

and in each step:
  * both fighters may land significant strikes (Poisson, at their rate for that position);
    every landed strike carries a small chance of a knockdown and of a KO/TKO finish
  * on the ground, either fighter may attempt a submission, which may finish the fight
  * the position may change: takedowns, clinch entries/breaks, stand-ups, reversals
Rounds restart at distance. A fight that survives every round goes to the judges, modeled
by a logistic regression fitted on real decisions (strikes, knockdowns, takedowns, top time).

Every rate is "fighter A's offense × fighter B's defense" (odds-ratio / log5 rule) relative to
the division average, from UFC fights *before* the bout, shrunk toward the division average by
pseudo-exposure (a debutant is an exactly average member of their division).

ufcstats doesn't record time spent in each position, so it is estimated per fight:
control time is split into ground and clinch by where the strike attempts (and takedowns)
happened, and ground time is split between the fighters by their control time. The
simulator uses the same accounting, so rates and simulated time are consistent.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DT = 5.0                       # seconds per step
STEPS_PER_ROUND = int(300 / DT)
POS = ["distance", "clinch", "A top", "B top"]

# per-fighter quantities summed over career (offense "o_", allowed "a_")
PER_MIN = {  # rate name -> (count column, exposure column)  exposure in minutes
    "dist": ("dist_land", "t_dist"), "clinch": ("clinch_land", "t_clinch"),
    "gtop": ("ground_land", "t_top"), "td": ("td", "t_stand"),
    "sub": ("sub_att", "t_ground"), "rev": ("rev", "t_bot"),
}
PER_EVENT = {  # rate name -> (count, exposure)  exposure in events
    "ko": ("ko_win", "sig_land"), "kd": ("kd", "stand_land"), "subconv": ("sub_win", "sub_att"),
    "topdur": ("t_top", "top_entries"),        # minutes on top per top spell (inverse = stand-up hazard)
}
JUDGE_FEATS = ["sig_diff", "kd_diff", "td_diff", "top_diff", "subatt_diff"]


# ---------------------------------------------------------------------------
# 1. Per-fight position-time accounting
# ---------------------------------------------------------------------------
def fight_exposures(ff):
    """Add estimated position times (minutes) and the counts used by the simulator to fighter_fights."""
    d = ff.copy()
    T = d.fight_seconds.astype(float)
    C = (d.ctrl_sec + d.opp_ctrl_sec).clip(upper=T)
    g = d.ground_att + d.opp_ground_att + 2 * (d.td + d.opp_td)
    c = d.clinch_att + d.opp_clinch_att
    w_ground = np.where(g + c > 0, g / (g + c).where(g + c > 0, 1), 0.5)
    t_ground = C * w_ground
    t_clinch = C - t_ground
    share = np.where(C > 0, d.ctrl_sec / C.where(C > 0, 1), 0.5)
    out = pd.DataFrame({
        "t_total": T / 60, "t_dist": (T - C) / 60, "t_clinch": t_clinch / 60,
        "t_top": t_ground * share / 60, "t_bot": t_ground * (1 - share) / 60,
    }, index=d.index)
    out["t_stand"] = out.t_dist + out.t_clinch
    out["t_ground"] = out.t_top + out.t_bot
    for side, pre in [("o_", ""), ("a_", "opp_")]:
        out[side + "dist_land"] = d[pre + "distance"]
        out[side + "clinch_land"] = d[pre + "clinch"]
        out[side + "ground_land"] = d[pre + "ground"]
        out[side + "sig_land"] = d[pre + "sig_str"]
        out[side + "stand_land"] = d[pre + "distance"] + d[pre + "clinch"]
        out[side + "kd"] = d[pre + "kd"]
        out[side + "td"] = d[pre + "td"]
        out[side + "sub_att"] = d[pre + "sub_att"]
        out[side + "rev"] = d[pre + "rev"]
    won, lost = d.result == "W", d.result == "L"
    out["o_ko_win"] = (won & (d.method_group == "KO/TKO")).astype(float)
    out["a_ko_win"] = (lost & (d.method_group == "KO/TKO")).astype(float)
    out["o_sub_win"] = (won & (d.method_group == "SUB")).astype(float)
    out["a_sub_win"] = (lost & (d.method_group == "SUB")).astype(float)
    # exposures seen from each side: my offense uses my top time; what I allow uses opponent's top time
    out["o_t_top"], out["a_t_top"] = out.t_top, out.t_bot
    out["o_t_bot"], out["a_t_bot"] = out.t_bot, out.t_top
    out["o_top_entries"] = d.td + d.rev
    out["a_top_entries"] = d.opp_td + d.opp_rev
    for col in ["t_dist", "t_clinch", "t_stand", "t_ground"]:
        out["o_" + col] = out["a_" + col] = out[col]
    # fights without usable stats (some early events) contribute nothing
    out["valid"] = out.notna().all(axis=1) & (T > 0)
    return out.where(out.valid, 0).assign(valid=out.valid)


# ---------------------------------------------------------------------------
# 2. League (division) averages and career sums before each fight
# ---------------------------------------------------------------------------
def _rate(num, den):
    return num / den if den > 0 else 0.0


def league_table(ff, ex, before):
    """Division-average rates from fights before `before` (a Timestamp). Index = weight class, plus 'ALL'."""
    m = (ff.date < before) & (ff.date >= "2005-01-01")
    d = pd.concat([ff.loc[m, ["weight_class"]], ex.loc[m]], axis=1)
    rows = {}
    for wc, g in list(d.groupby("weight_class")) + [("ALL", d)]:
        s = g.sum(numeric_only=True)
        r = {k: _rate(s["o_" + c], s["o_" + e]) for k, (c, e) in PER_MIN.items()}
        r.update({k: _rate(s["o_" + c], s["o_" + e]) for k, (c, e) in PER_EVENT.items()})
        r["n"] = len(g)
        tt = s.t_total
        r.update({"share_dist": s.t_dist / tt, "share_clinch": s.t_clinch / tt, "share_ground": s.t_ground / tt,
                  "slpm": s.o_sig_land / tt, "td_per15": 15 * s.o_td / tt})
        rows[wc] = r
    lg = pd.DataFrame(rows).T
    small = lg.n < 300
    lg.loc[small, lg.columns.drop("n")] = lg.loc["ALL", lg.columns.drop("n")].values
    return lg


def career_sums(ff, ex, half_life=None):
    """Sums of every o_/a_ column over each fighter's previous fights (this fight excluded).

    half_life (years): if set, a fight from h years before counts 0.5^(h / half_life), so the
    rates follow a fighter's recent form (and decline) instead of their whole career equally.
    """
    cols = [c for c in ex.columns if c.startswith(("o_", "a_"))]
    d = pd.concat([ff[["fighter", "date", "fight_id"]], ex[cols]], axis=1).sort_values(["fighter", "date", "fight_id"])
    if not half_life:
        cs = d.groupby("fighter")[cols].cumsum() - d[cols]
    else:
        out = np.zeros((len(d), len(cols)))
        X = d[cols].to_numpy(float)
        years = (d.date - pd.Timestamp("1990-01-01")).dt.days.to_numpy() / 365.25
        pos = 0
        for n in d.groupby("fighter", sort=False).size().reindex(d.fighter.unique()).to_numpy():
            yy, xx = years[pos:pos + n], X[pos:pos + n]
            w = 0.5 ** ((yy[:, None] - yy[None, :]) / half_life) * np.tri(n, n, -1)   # strictly earlier fights
            out[pos:pos + n] = w @ xx
            pos += n
        cs = pd.DataFrame(out, index=d.index, columns=cols)
    cs["n_prev"] = d.groupby("fighter").cumcount()
    return cs.reindex(ff.index)


# ---------------------------------------------------------------------------
# 3. Matchup parameters: shrink each fighter's rates, combine offense × defense
# ---------------------------------------------------------------------------
@dataclass
class Params:
    k_min: float = 15.0        # pseudo-minutes of division-average fighting added to every career
    k_ko: float = 150.0        # pseudo landed strikes for knockdown / KO-per-strike rates
    k_sub: float = 4.0         # pseudo submission attempts for submission conversion
    clinch_out: float = 1 / 25  # per second: mean clinch spell ~25 s
    up_mult: float = 1.0       # calibrates league ground time
    clinch_mult: float = 1.0   # calibrates league clinch time
    ko_mult: float = 1.0       # calibrates league KO rate
    sub_mult: float = 1.0      # calibrates league submission rate
    extra: dict = field(default_factory=dict)


def _shrunk(cs, L, side, name, k_exp, table):
    c, e = table[name]
    num, den = cs[side + c].fillna(0).values, cs[side + e].fillna(0).values
    return (num + k_exp * L[name]) / (den + k_exp)


def fighter_rates(cs, L, P: Params):
    """Shrunk offense (o_) and allowed (a_) rates for each row of career sums `cs`; L = league row per fight."""
    out = {}
    share = {"dist": "share_dist", "clinch": "share_clinch", "gtop": "share_ground", "td": None, "sub": "share_ground", "rev": "share_ground"}
    for side in ["o_", "a_"]:
        for name, (c, e) in PER_MIN.items():
            sh = L["share_ground"] / 2 if name in ("gtop", "rev") else (L[share[name]] if share[name] else L.share_dist + L.share_clinch)
            out[side + name] = _shrunk(cs, L, side, name, P.k_min * sh, PER_MIN)
        out[side + "ko"] = _shrunk(cs, L, side, "ko", P.k_ko, PER_EVENT)
        out[side + "kd"] = _shrunk(cs, L, side, "kd", P.k_ko, PER_EVENT)
        out[side + "subconv"] = _shrunk(cs, L, side, "subconv", P.k_sub, PER_EVENT)
        out[side + "topdur"] = _shrunk(cs, L, side, "topdur", P.k_min * L.share_ground / 2 / L.topdur.clip(lower=1e-6), PER_EVENT)
    # how much of a fighter's standing time is spent in the clinch, relative to the division
    lc = L.share_clinch / (L.share_dist + L.share_clinch)
    k = P.k_min * (L.share_dist + L.share_clinch)
    own = (cs.o_t_clinch.fillna(0).values + k * lc) / (cs.o_t_stand.fillna(0).values + k)
    out["clinch_prop"] = own / lc
    return pd.DataFrame(out, index=cs.index)


def combine(ra, rb, L, P: Params):
    """Per-matchup step probabilities for A (index-aligned rows of ra/rb/L). Returns dict of arrays.

    rate_A = league × (A offense / league) × (B allowed / league)
    """
    def oddr(a, b, l):
        l = np.asarray(l, float)
        return np.where(l > 0, np.asarray(a) * np.asarray(b) / np.where(l > 0, l, 1), 0.0)

    def side(r1, r2):
        p = {}
        p["dist"] = oddr(r1.o_dist, r2.a_dist, L["dist"])
        p["clinch"] = oddr(r1.o_clinch, r2.a_clinch, L["clinch"])
        p["gtop"] = oddr(r1.o_gtop, r2.a_gtop, L["gtop"])
        p["td"] = oddr(r1.o_td, r2.a_td, L["td"])
        p["sub"] = oddr(r1.o_sub, r2.a_sub, L["sub"]) * P.sub_mult
        p["rev"] = oddr(r1.o_rev, r2.a_rev, L["rev"])
        p["ko"] = np.clip(oddr(r1.o_ko, r2.a_ko, L["ko"]) * P.ko_mult, 0, 0.2)
        p["kd"] = np.clip(oddr(r1.o_kd, r2.a_kd, L["kd"]), 0, 0.3)
        p["subconv"] = np.clip(oddr(r1.o_subconv, r2.a_subconv, L["subconv"]), 0, 0.95)
        # stand-up hazard while r1 is on top: short top spells for r1 (offense) and quick escapes for r2 (allowed)
        p["up"] = P.up_mult / np.clip(oddr(r1.o_topdur, r2.a_topdur, L["topdur"]), 1e-3, None)
        return p
    pa, pb = side(ra, rb), side(rb, ra)
    # clinch entry: league hazard scaled by how much both fighters' fights are spent clinching
    prop = np.sqrt(ra.clinch_prop.values * rb.clinch_prop.values)
    shc, shd = L.share_clinch.values, L.share_dist.values
    clinch_in = P.clinch_mult * P.clinch_out * shc / np.maximum(shd, 1e-6) * prop  # per second
    return pa, pb, clinch_in


# ---------------------------------------------------------------------------
# 4. Judges
# ---------------------------------------------------------------------------
class JudgeModel:
    """P(A wins the decision) from the fight's stat differences (A minus B)."""

    def fit(self, ff, ex, before):
        m = (ff.date < before) & (ff.method_group == "DEC") & ff.result.isin(["W", "L"]) & (ff.corner == "Red") & ex.valid
        X = self.features(ex.loc[m, "o_sig_land"] - ex.loc[m, "a_sig_land"], ex.loc[m, "o_kd"] - ex.loc[m, "a_kd"],
                          ex.loc[m, "o_td"] - ex.loc[m, "a_td"], ex.loc[m, "t_top"] - ex.loc[m, "t_bot"],
                          ex.loc[m, "o_sub_att"] - ex.loc[m, "a_sub_att"])
        y = (ff.loc[m, "result"] == "W").astype(int)
        # symmetric: add the mirror so there's no corner bias
        self.lr = LogisticRegression(C=1.0, fit_intercept=False, max_iter=2000).fit(np.vstack([X, -X]), np.r_[y, 1 - y])
        self.n = int(m.sum())
        self.acc = float(((self.lr.predict_proba(X)[:, 1] > 0.5) == y).mean())
        return self

    @staticmethod
    def features(sig, kd, td, top, sub):
        return np.column_stack([np.asarray(x, float) for x in (sig, kd, td, top, sub)])

    def proba(self, X):
        z = X @ self.lr.coef_[0]
        return 1 / (1 + np.exp(-z))

    @property
    def coef(self):
        return pd.Series(self.lr.coef_[0], index=JUDGE_FEATS)


# ---------------------------------------------------------------------------
# 5. Monte Carlo
# ---------------------------------------------------------------------------
def simulate(pa, pb, clinch_in, rounds, judge, P: Params, n=2000, seed=0, log=0, raw=False):
    """Simulate n fights for each of M matchups (arrays of length M in pa/pb/clinch_in/rounds).

    Returns a dict of (M,)-arrays of outcome probabilities plus expected stats, and, if log>0,
    full step-by-step event logs for the first `log` simulations of each matchup.
    raw=True also returns every simulation's ending, round, minutes, stats and P(A wins the
    decision) as (M, n) arrays under out["raw"] (for distribution plots).
    """
    rng = np.random.default_rng(seed)
    M = len(rounds)
    rounds = np.asarray(rounds, int)
    R = rounds.max()
    col = lambda x: np.asarray(x, float)[:, None]
    f = DT / 60
    A = {k: col(v) for k, v in pa.items()}
    B = {k: col(v) for k, v in pb.items()}
    cin = col(clinch_in) * DT
    cout = P.clinch_out * DT

    pos = np.zeros((M, n), np.int8)
    alive = np.ones((M, n), bool)
    end = np.zeros((M, n), np.int8)        # 0 decision, 1 A KO, 2 A SUB, 3 B KO, 4 B SUB
    end_round = np.zeros((M, n), np.int8)
    st = {k: np.zeros((M, n), np.float32) for k in ["a_sig", "b_sig", "a_kd", "b_kd", "a_td", "b_td", "a_top", "b_top",
                                                    "a_sub", "b_sub", "t_dist", "t_clinch"]}
    logs = []
    for step in range(R * STEPS_PER_ROUND):
        rnd = step // STEPS_PER_ROUND
        if step % STEPS_PER_ROUND == 0:
            pos[:] = 0
        live = alive & (rnd < rounds)[:, None]
        if not live.any():
            break
        d, c, at, bt = pos == 0, pos == 1, pos == 2, pos == 3
        stand = d | c
        lam_a = np.where(d, A["dist"], np.where(c, A["clinch"], np.where(at, A["gtop"], 0.0))) * f
        lam_b = np.where(d, B["dist"], np.where(c, B["clinch"], np.where(bt, B["gtop"], 0.0))) * f
        sa = rng.poisson(lam_a) * live
        sb = rng.poisson(lam_b) * live
        st["a_sig"] += sa
        st["b_sig"] += sb
        u = rng.random((6, M, n))
        kda = stand & (sa > 0) & (u[0] < 1 - (1 - A["kd"]) ** sa)
        kdb = stand & (sb > 0) & (u[1] < 1 - (1 - B["kd"]) ** sb)
        st["a_kd"] += kda
        st["b_kd"] += kdb
        koa = (sa > 0) & (u[2] < 1 - (1 - A["ko"]) ** sa)
        kob = (sb > 0) & (u[3] < 1 - (1 - B["ko"]) ** sb)
        ground = at | bt
        suba = ground & live & (u[4] < A["sub"] * f)
        subb = ground & live & (u[5] < B["sub"] * f)
        st["a_sub"] += suba
        st["b_sub"] += subb
        v = rng.random((3, M, n))
        sub_win_a = suba & (v[0] < A["subconv"])
        sub_win_b = subb & (v[1] < B["subconv"])
        a_first = v[2] < 0.5
        fin_a_ko = koa & ~(kob & ~a_first)
        fin_b_ko = kob & ~fin_a_ko
        fin_a_sub = sub_win_a & ~fin_a_ko & ~fin_b_ko
        fin_b_sub = sub_win_b & ~fin_a_ko & ~fin_b_ko & ~fin_a_sub
        done = live & (fin_a_ko | fin_b_ko | fin_a_sub | fin_b_sub)
        end[done & fin_a_ko] = 1
        end[done & fin_a_sub] = 2
        end[done & fin_b_ko] = 3
        end[done & fin_b_sub] = 4
        end_round[done] = rnd + 1
        alive &= ~done
        live &= ~done
        # time in position this step
        st["t_dist"] += (d & live) * f
        st["t_clinch"] += (c & live) * f
        st["a_top"] += (at & live) * f
        st["b_top"] += (bt & live) * f
        # transitions (at most one per step)
        w = rng.random((M, n))
        tda, tdb = A["td"] * f, B["td"] * f
        new = pos.copy()
        # standing: takedown A, takedown B, clinch in/out
        p1 = tda
        p2 = p1 + tdb
        p3 = p2 + np.where(d, cin, cout)
        sel = stand & live
        new = np.where(sel & (w < p1), 2, new)
        new = np.where(sel & (w >= p1) & (w < p2), 3, new)
        new = np.where(sel & (w >= p2) & (w < p3), np.where(d, 1, 0), new)
        st["a_td"] += sel & (w < p1)
        st["b_td"] += sel & (w >= p1) & (w < p2)
        # ground: stand-up, reversal
        upa, upb = A["up"] * f, B["up"] * f      # hazard while A (resp. B) is on top
        reva, revb = A["rev"] * f, B["rev"] * f  # A reverses from bottom (B on top), and vice versa
        new = np.where(at & live & (w < upa), 0, new)
        new = np.where(at & live & (w >= upa) & (w < upa + revb), 3, new)
        new = np.where(bt & live & (w < upb), 0, new)
        new = np.where(bt & live & (w >= upb) & (w < upb + reva), 2, new)
        if log:
            logs.append({"pos": pos[:, :log].copy(), "sa": sa[:, :log].copy(), "sb": sb[:, :log].copy(),
                         "kda": kda[:, :log].copy(), "kdb": kdb[:, :log].copy(),
                         "suba": suba[:, :log].copy(), "subb": subb[:, :log].copy(),
                         "tda": (sel & (w < p1))[:, :log].copy(), "tdb": (sel & (w >= p1) & (w < p2))[:, :log].copy(),
                         "end": np.where(done, end, 0)[:, :log].copy()})
        pos = new.astype(np.int8)

    went = alive
    feats = np.stack([st["a_sig"] - st["b_sig"], st["a_kd"] - st["b_kd"], st["a_td"] - st["b_td"],
                      st["a_top"] - st["b_top"], st["a_sub"] - st["b_sub"]], axis=-1)
    p_dec_a = judge.proba(feats.reshape(-1, feats.shape[-1])).reshape(M, n)
    out = {
        "A KO/TKO": (end == 1).mean(1), "A SUB": (end == 2).mean(1), "A DEC": (went * p_dec_a).mean(1),
        "B KO/TKO": (end == 3).mean(1), "B SUB": (end == 4).mean(1), "B DEC": (went * (1 - p_dec_a)).mean(1),
    }
    out["p_a"] = out["A KO/TKO"] + out["A SUB"] + out["A DEC"]
    out["finish_round"] = np.stack([((end > 0) & (end_round == r + 1)).mean(1) for r in range(R)], 1)
    out["a_finish_round"] = np.stack([(((end == 1) | (end == 2)) & (end_round == r + 1)).mean(1) for r in range(R)], 1)
    out["b_finish_round"] = np.stack([(((end == 3) | (end == 4)) & (end_round == r + 1)).mean(1) for r in range(R)], 1)
    mins = (st["t_dist"] + st["t_clinch"] + st["a_top"] + st["b_top"])
    for k in st:
        out["mean_" + k] = st[k].mean(1)
    out["mean_minutes"] = mins.mean(1)
    if log:
        out["logs"] = logs
        out["log_end"] = end[:, :log]
        out["log_end_round"] = end_round[:, :log]
        out["log_p_dec_a"] = p_dec_a[:, :log]
    if raw:
        out["raw"] = {"end": end, "end_round": end_round, "minutes": mins, "p_dec_a": p_dec_a,
                      # P(A wins) for each simulation: 1/0 when finished, the judges' probability otherwise
                      "a_wins": np.where(end == 0, p_dec_a, (end == 1) | (end == 2)).astype(float),
                      **{k: st[k] for k in st}}
    return out


# ---------------------------------------------------------------------------
# 6. Putting it together: predictions for real fights (walk-forward) and any matchup
# ---------------------------------------------------------------------------
OUT7 = ["A KO/TKO", "A SUB", "A DEC", "B KO/TKO", "B SUB", "B DEC"]


class Simulator:
    """Holds the data, league table and judge model fitted on fights before `before`."""

    def __init__(self, ff, before, P: Params | None = None, half_life=None):
        self.ff = ff.sort_values(["fighter", "date", "fight_id"]).reset_index(drop=True)
        self.half_life = half_life
        self.ex = fight_exposures(self.ff)
        self.cs = career_sums(self.ff, self.ex, half_life)
        self.before = pd.Timestamp(before)
        self.L = league_table(self.ff, self.ex, self.before)
        self.judge = JudgeModel().fit(self.ff, self.ex, self.before)
        self.P = P or Params()

    def _league_rows(self, wcs):
        wcs = pd.Series(wcs).where(pd.Series(wcs).isin(self.L.index), "ALL")
        return self.L.loc[wcs.values].reset_index(drop=True)

    def params_for(self, cs_a, cs_b, wcs):
        L = self._league_rows(wcs)
        ra = fighter_rates(cs_a.reset_index(drop=True), L, self.P)
        rb = fighter_rates(cs_b.reset_index(drop=True), L, self.P)
        return combine(ra, rb, L, self.P)

    def fight_params(self, fights):
        """Matchup parameters for real fights (red = A), each fighter's history strictly before the fight."""
        key = self.ff.set_index(["fight_id", "corner"]).index
        idx = pd.Series(np.arange(len(self.ff)), index=key)
        ia = idx.reindex(pd.MultiIndex.from_arrays([fights.fight_id, np.repeat("Red", len(fights))])).values
        ib = idx.reindex(pd.MultiIndex.from_arrays([fights.fight_id, np.repeat("Blue", len(fights))])).values
        return self.params_for(self.cs.iloc[ia], self.cs.iloc[ib], fights.weight_class.values)

    def run_fights(self, fights, n=1000, seed=0, chunk=400):
        """Simulate every fight in `fights` (needs fight_id, weight_class, total_rounds). One row per fight."""
        res = []
        for i in range(0, len(fights), chunk):
            f = fights.iloc[i:i + chunk]
            pa, pb, cin = self.fight_params(f)
            o = simulate(pa, pb, cin, f.total_rounds.clip(1, 5).values, self.judge, self.P, n=n, seed=seed + i)
            res.append(pd.DataFrame({k: o[k] for k in OUT7 + ["p_a", "mean_a_sig", "mean_b_sig", "mean_a_td", "mean_b_td",
                                                                 "mean_t_dist", "mean_t_clinch", "mean_a_top", "mean_b_top",
                                                                 "mean_a_kd", "mean_b_kd", "mean_a_sub", "mean_b_sub", "mean_minutes"]},
                                     index=f.index))
        return pd.concat(res)

    # --- any two fighters, as of today -------------------------------------------------
    def career_now(self, name):
        cols = [c for c in self.ex.columns if c.startswith(("o_", "a_"))]
        rows = self.ff.fighter == name
        w = np.ones(int(rows.sum()))
        if self.half_life:
            # relative to the fighter's own last fight: a retired great is simulated at their final form,
            # not discounted for the years since they retired
            d = self.ff.loc[rows, "date"]
            years = (d.max() - d).dt.days.to_numpy() / 365.25
            w = 0.5 ** (years / self.half_life)
        s = (self.ex.loc[rows, cols] * w[:, None]).sum()
        s["n_prev"] = int(rows.sum())
        return s.to_frame().T

    def matchup(self, a, b, weight_class=None, rounds=3, n=10000, seed=0, log=0, raw=False):
        ca, cb = self.career_now(a), self.career_now(b)
        if weight_class is None:
            last = self.ff[self.ff.fighter == a].sort_values("date").tail(1)
            weight_class = last.weight_class.iloc[0] if len(last) else "ALL"
        pa, pb, cin = self.params_for(ca, cb, [weight_class])
        return simulate(pa, pb, cin, [rounds], self.judge, self.P, n=n, seed=seed, log=log, raw=raw)


def outcome_probs(sim, other=0.003):
    """7-outcome table in modeling.OUTCOMES order (Red = A), with a small 'Other' share."""
    o = sim[OUT7].div(sim[OUT7].sum(axis=1), axis=0) * (1 - other)
    o.columns = [c.replace("A ", "Red ").replace("B ", "Blue ") for c in OUT7]
    o["Other"] = other
    return o


def calibrate(sim_obj, fights, n=400, seed=0, iters=3, verbose=True):
    """Tune the five league multipliers so simulated averages match the real ones on `fights`.

    Targets: ground share, clinch share, KO rate, submission rate (all league-wide).
    Fixed-point updates; a few iterations are enough.
    """
    f = sim_obj.ff
    real = fights.merge(f[f.corner == "Red"][["fight_id"]], on="fight_id")
    ex = sim_obj.ex
    m = f.fight_id.isin(fights.fight_id) & (f.corner == "Red")
    tt = ex.loc[m, "t_total"].sum()
    tgt = {"ground": ex.loc[m, "t_ground"].sum() / tt, "clinch": ex.loc[m, "t_clinch"].sum() / tt,
           "ko": (fights.method_group == "KO/TKO").mean(), "sub": (fights.method_group == "SUB").mean()}
    P = sim_obj.P
    for it in range(iters):
        s = sim_obj.run_fights(fights, n=n, seed=seed)
        mins = s.mean_minutes.sum()
        got = {"ground": (s.mean_a_top + s.mean_b_top).sum() / mins, "clinch": s.mean_t_clinch.sum() / mins,
               "ko": (s["A KO/TKO"] + s["B KO/TKO"]).mean(), "sub": (s["A SUB"] + s["B SUB"]).mean()}
        if verbose:
            print(f"iter {it}: " + "  ".join(f"{k} {got[k]:.3f}/{tgt[k]:.3f}" for k in tgt))
        P.up_mult *= (got["ground"] / tgt["ground"]) ** 1.2
        P.clinch_mult *= (tgt["clinch"] / got["clinch"])
        P.ko_mult *= tgt["ko"] / got["ko"]
        P.sub_mult *= tgt["sub"] / got["sub"]
    return tgt, got


def play_by_play(sim, names, m=0, j=0):
    """Readable event list for simulated fight j of matchup m (needs simulate(..., log>j)).

    Returns (events DataFrame [round, clock, event], per-round strike table, result text).
    """
    a, b = names
    logs = sim["logs"]
    ev, per_round = [], {}
    prev = 0
    for step, L in enumerate(logs):
        rnd, sec = step // STEPS_PER_ROUND + 1, (step % STEPS_PER_ROUND) * DT
        clock = f"{int(sec // 60)}:{int(sec % 60):02d}"
        pos = int(L["pos"][m, j])
        if step % STEPS_PER_ROUND == 0:
            prev = 0
        pr = per_round.setdefault(rnd, {a: 0, b: 0})
        pr[a] += int(L["sa"][m, j])
        pr[b] += int(L["sb"][m, j])
        if pos != prev:
            text = {(0, 1): "Clinch against the fence", (1, 0): "They break apart",
                    (2, 0): f"{b} gets back up", (3, 0): f"{a} gets back up",
                    (2, 3): f"{b} reverses — now on top", (3, 2): f"{a} reverses — now on top"}.get((prev, pos))
            if text:
                ev.append((rnd, clock, text))
        for who, k in [(a, "kda"), (b, "kdb")]:
            if L[k][m, j]:
                ev.append((rnd, clock, f"{who} scores a KNOCKDOWN"))
        for who, k in [(a, "tda"), (b, "tdb")]:
            if L[k][m, j]:
                ev.append((rnd, clock, f"{who} lands a takedown"))
        for who, k in [(a, "suba"), (b, "subb")]:
            if L[k][m, j]:
                ev.append((rnd, clock, f"{who} attempts a submission"))
        e = int(L["end"][m, j])
        if e:
            ev.append((rnd, clock, {1: f"{a} wins by KO/TKO", 2: f"{a} wins by submission",
                                    3: f"{b} wins by KO/TKO", 4: f"{b} wins by submission"}[e]))
            break
        prev = pos
    end = int(sim["log_end"][m, j])
    if end:
        result = ev[-1][2] + f" in round {ev[-1][0]} ({ev[-1][1]})"
    else:
        p = float(sim["log_p_dec_a"][m, j])
        result = f"Goes the distance: judges favor {a if p >= 0.5 else b} ({max(p, 1 - p):.0%} likely)"
    strikes = pd.DataFrame(per_round).T.rename_axis("round")
    return pd.DataFrame(ev, columns=["round", "clock", "event"]), strikes, result
