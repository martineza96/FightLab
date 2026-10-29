"""Elo ratings for every UFC fighter, and pound-for-pound rankings built on them.

Elo in one line: after each fight the winner takes points from the loser, and the
amount depends on how surprising the result was. Beating a much higher-rated
opponent gains a lot; beating a newcomer gains little. Weight class plays no part,
which makes it a natural pound-for-pound measure.

Pre-fight ratings only use earlier fights, so they're safe to use as model features.
"""
from pathlib import Path

import numpy as np
import pandas as pd

INIT = 1500
# Chosen by tune() on 2010-03 → 2023 fights (lowest log loss); see notebook 05.
# A bigger K for newcomers was also tried; it barely helped, so it was left out.
DEFAULT_K = 64
DEFAULT_FINISH_MULT = 1.25


def expected(r_a, r_b):
    """Probability that A beats B."""
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


def compute_elo(fights, k=DEFAULT_K, finish_mult=DEFAULT_FINISH_MULT, init=INIT):
    """Walk fights in time order; return pre/post ratings per fight (one row per fight)."""
    f = fights.sort_values("fight_id")
    ratings = {}
    out = np.empty((len(f), 4))
    for i, (r, b, w, m) in enumerate(zip(f.r_fighter, f.b_fighter, f.winner, f.method_group)):
        ra, rb = ratings.get(r, init), ratings.get(b, init)
        score = {"Red": 1.0, "Blue": 0.0, "Draw": 0.5}.get(w)  # no-contest -> None
        if score is None:
            out[i] = ra, rb, ra, rb
            continue
        kk = k * (finish_mult if w != "Draw" and m in ("KO/TKO", "SUB") else 1.0)
        delta = kk * (score - expected(ra, rb))
        ratings[r], ratings[b] = ra + delta, rb - delta
        out[i] = ra, rb, ra + delta, rb - delta
    return pd.DataFrame(out, columns=["r_elo_pre", "b_elo_pre", "r_elo_post", "b_elo_post"],
                        index=f.fight_id).reset_index()


def history(fights, per_fight):
    """Long table: one row per fighter per fight with ratings before and after."""
    f = fights.merge(per_fight, on="fight_id")
    rows = []
    for me, opp, color in [("r", "b", "Red"), ("b", "r", "Blue")]:
        rows.append(pd.DataFrame({
            "fight_id": f.fight_id, "date": f.date, "event": f.event, "corner": color,
            "fighter": f[f"{me}_fighter"], "opponent": f[f"{opp}_fighter"],
            "result": np.select([f.winner == color, f.winner == "Draw", f.winner == "NC"],
                                ["W", "D", "N"], "L"),
            "method_group": f.method_group, "weight_class": f.weight_class, "gender": f.gender,
            "is_title_bout": f.is_title_bout,
            "elo_pre": f[f"{me}_elo_pre"], "elo_post": f[f"{me}_elo_post"],
            "opp_elo_pre": f[f"{opp}_elo_pre"]}))
    return pd.concat(rows, ignore_index=True).sort_values(["fight_id", "corner"]).reset_index(drop=True)


def evaluate(fights, per_fight, start, end):
    """Log loss / accuracy of 'higher Elo wins' on decided fights in [start, end)."""
    f = fights.merge(per_fight, on="fight_id")
    f = f[(f.date >= start) & (f.date < end) & f.winner.isin(["Red", "Blue"])]
    p = expected(f.r_elo_pre, f.b_elo_pre).clip(1e-6, 1 - 1e-6)
    y = (f.winner == "Red").astype(int)
    acc = np.where(p == 0.5, 0.5, (p > 0.5) == y).mean()  # coin flip when ratings are equal
    return {"fights": len(f), "accuracy": acc,
            "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()}


def tune(fights, ks=(24, 32, 40, 48, 64, 80), mults=(1.0, 1.25, 1.5, 2.0),
         start="2010-03-21", end="2024-01-01"):
    rows = []
    for k in ks:
        for m in mults:
            rows.append({"k": k, "finish_mult": m,
                         **evaluate(fights, compute_elo(fights, k, m), start, end)})
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def _record(h):
    c = h.result.value_counts()
    rec = f"{c.get('W', 0)}-{c.get('L', 0)}"
    if c.get("D", 0):
        rec += f"-{c['D']}"
    if c.get("N", 0):
        rec += f" ({c['N']} NC)"
    return rec


def p4p(hist, as_of=None, n=15, gender="Men", min_fights=5, active_days=548):
    """Pound-for-pound top n as of a date: highest current Elo among active fighters.

    'Active' = fought within `active_days` (~18 months) before `as_of`. Retirements aren't
    in the data, so recently retired fighters can linger until the window passes.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else hist.date.max()
    h = hist[hist.date <= as_of]
    if gender:
        h = h[h.gender == gender]
    last = h.sort_values("fight_id").groupby("fighter").tail(1).set_index("fighter")
    counts = h.groupby("fighter").size()
    ok = last[(last.date >= as_of - pd.Timedelta(days=active_days))
              & (counts.reindex(last.index) >= min_fights)]
    top = ok.sort_values("elo_post", ascending=False).head(n)
    if top.empty:                       # e.g. women before 2015: nobody with 5+ UFC fights yet
        return pd.DataFrame(columns=["fighter", "elo", "ufc_record", "division", "last_fight", "last_result"],
                            index=pd.Index([], name="rank"))
    recs = h[h.fighter.isin(top.index)].groupby("fighter").apply(_record, include_groups=False)
    return pd.DataFrame({
        "rank": range(1, len(top) + 1), "fighter": top.index, "elo": top.elo_post.round(0).astype(int),
        "ufc_record": recs.reindex(top.index).values, "division": top.weight_class.values,
        "last_fight": top.date.dt.date.values,
        "last_result": (top.result + " vs " + top.opponent).values}).set_index("rank")


def p4p_rank(hist, fighter, as_of=None, gender=None, **kw):
    """Fighter's P4P rank as of a date (None if not ranked among the top 100)."""
    gender = gender or hist.loc[hist.fighter == fighter, "gender"].iloc[-1]
    table = p4p(hist, as_of, n=100, gender=gender, **kw)
    hit = table.index[table.fighter == fighter]
    return int(hit[0]) if len(hit) else None


def peak_elo(hist, n=15, gender="Men", min_fights=5):
    """All-time leaderboard of each fighter's highest-ever rating."""
    h = hist[hist.gender == gender] if gender else hist
    counts = h.groupby("fighter").size()
    h = h[h.fighter.map(counts) >= min_fights]
    best = h.loc[h.groupby("fighter").elo_post.idxmax()].sort_values("elo_post", ascending=False).head(n)
    recs = [_record(h[(h.fighter == r.fighter) & (h.fight_id <= r.fight_id)]) for r in best.itertuples()]
    return pd.DataFrame({
        "rank": range(1, len(best) + 1), "fighter": best.fighter.values,
        "peak_elo": best.elo_post.round(0).astype(int).values, "peak_date": best.date.dt.date.values,
        "record_at_peak": recs, "division": best.weight_class.values}).set_index("rank")


def run(fights, out_dir="data/processed"):
    hist = history(fights, compute_elo(fights))
    hist.to_parquet(Path(out_dir) / "elo_history.parquet", index=False)
    return hist
