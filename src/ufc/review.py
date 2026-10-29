"""Look back at a finished card: which picks were right, and what pointed to the winners.

The predictions are made exactly like the Next card's (upcoming.predict_card), using only
fights before the event. That is only honest while our data stops before the event, so the
first review built for an event is frozen to data/processed/reviews/<event_id>.json and
served from there forever after; once the data includes the event, a review that was never
frozen is refused rather than rebuilt with hindsight.

Indicators:
  * driver hit rate - for each model factor, how often the fighter it pushed toward won
  * tale-of-the-tape edges - how often the younger / longer / more experienced fighter won,
    next to the same rate over every UFC fight since 2015 (one card is a tiny sample)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ufc import upcoming
from ufc.appdata import clean
from ufc.predict import _nice

MIN_PUSH = 0.02          # ignore factors that barely moved a prediction (log-odds)
HIST_FROM = "2015-01-01"
BANDS = [("Toss-up", 0.0, 0.55), ("Lean", 0.55, 0.65), ("Confident", 0.65, 1.01)]
FACTOR_NAMES = {"debut": "UFC debut", "home_abroad": "Home crowd", "us_event": "Visitor at a US event",
                "n_fights": "UFC experience", "ko_wins": "KO wins", "sub_wins": "submission wins",
                "dec_wins": "decision wins", "been_finished": "times finished", "win_streak": "win streak",
                "lose_streak": "losing streak", "ufc_minutes": "cage time", "owp": "opponents' win rate (OWP)",
                "oowp": "opponents' opponents (OOWP)"}
# tale-of-the-tape edges: stat, label for the fighter with the edge, +1 = more is the edge
EDGES = [("age_at_fight", "Younger fighter", -1), ("reach", "Longer reach", 1), ("height", "Taller fighter", 1),
         ("n_fights", "More UFC fights", 1), ("win_streak", "Longer win streak", 1), ("elo_pre", "Higher Elo", 1)]


class LeakError(Exception):
    """The data already contains this event, so a fresh 'prediction' would use its result."""


def factor_name(f):
    s = FACTOR_NAMES.get(f, _nice(f))
    return s[0].upper() + s[1:]


def historical_edges(processed):
    """Over UFC fights since 2015: share won by the fighter with each edge (ties / unknowns skipped)."""
    p = Path(processed)
    t = pd.read_parquet(p / "model_table.parquet")
    t = t[t.date >= HIST_FROM]
    elo = pd.read_parquet(p / "elo_history.parquet").pivot_table(index="fight_id", columns="corner", values="elo_pre")
    t = t.assign(elo_pre_diff=t.fight_id.map(elo.get("Red")) - t.fight_id.map(elo.get("Blue")))
    out = {}
    for stat, _, sign in EDGES:
        d = t[f"{stat}_diff"] * sign
        ok = d.notna() & (d != 0)
        won = (d[ok] > 0) == t.red_win[ok].astype(bool)
        out[stat] = {"rate": float(won.mean()), "n": int(ok.sum())}
    return out


def _band(p):
    return next(name for name, lo, hi in BANDS if lo <= p < hi)


def review_card(event, predictor, matcher, processed, cache_dir="data/processed/cache"):
    card = upcoming.predict_card(event, predictor, matcher, cache_dir=cache_dir)
    bouts, factors, edges = [], {}, {s: {"n": 0, "won": 0} for s, _, _ in EDGES}
    for b in card["bouts"]:
        pred = b["prediction"]
        win_idx = next((i for i, f in enumerate(b["fighters"]) if f.get("winner")), None)
        drivers = pred.pop("all_drivers")
        stats = pred.pop("stats")
        fav_side = "red" if pred["p_red"] >= 0.5 else "blue"
        p_fav = max(pred["p_red"], pred["p_blue"])
        top = pred["outcomes"][0] if pred["outcomes"] else None
        row = {**b, "decided": win_idx is not None, "pick_side": fav_side, "p_pick": p_fav, "band": _band(p_fav)}
        if win_idx is None:                    # draw / no contest / not reported
            row["drivers"] = drivers[:6]
            bouts.append(row)
            continue
        win_side = ["red", "blue"][win_idx]
        result = b.get("result") or {}
        row.update(winner_side=win_side, winner=b["fighters"][win_idx]["name"], correct=win_side == fav_side,
                   p_winner=pred["p_red"] if win_side == "red" else pred["p_blue"])
        if top and result:
            row["method_check"] = {"predicted": top["method"], "predicted_decision": top["method"] == "DEC",
                                   "actual_decision": result.get("decision"),
                                   "match": (top["method"] == "DEC") == bool(result.get("decision"))}
        for d in drivers:
            d["pointed_to_winner"] = d["side"] == win_side
            if d["size"] < MIN_PUSH:
                continue
            g = factors.setdefault(d["feature"], {"feature": d["feature"], "label": factor_name(d["feature"]),
                                                   "bouts": 0, "hits": 0, "push_to_winner": 0.0, "push_to_loser": 0.0})
            g["bouts"] += 1
            g["hits"] += d["pointed_to_winner"]
            g["push_to_winner" if d["pointed_to_winner"] else "push_to_loser"] += d["size"]
        row["drivers"] = drivers[:6]
        row["edges"] = []
        for stat, label, sign in EDGES:
            w, l = stats[win_side].get(stat), stats["blue" if win_side == "red" else "red"].get(stat)
            if w is None or l is None or w == l:
                continue
            had = (w - l) * sign > 0
            edges[stat]["n"] += 1
            edges[stat]["won"] += had
            row["edges"].append({"stat": stat, "label": label, "winner_had_it": bool(had)})
        bouts.append(row)

    decided = [b for b in bouts if b["decided"]]
    pw = np.clip([b["p_winner"] for b in decided], 1e-6, 1 - 1e-6)
    hits = [b for b in decided if b["correct"]]
    misses = [b for b in decided if not b["correct"]]
    hist = historical_edges(processed)
    summary = {
        "bouts": len(bouts), "decided": len(decided), "correct": len(hits),
        "accuracy": len(hits) / len(decided) if decided else None,
        "log_loss": float(-np.mean(np.log(pw))) if decided else None,
        "brier": float(np.mean((1 - pw) ** 2)) if decided else None,
        "avg_conf_correct": float(np.mean([b["p_pick"] for b in hits])) if hits else None,
        "avg_conf_missed": float(np.mean([b["p_pick"] for b in misses])) if misses else None,
        "bands": [{"band": name, "bouts": sum(b["band"] == name for b in decided),
                   "correct": sum(b["band"] == name and b["correct"] for b in decided)} for name, _, _ in BANDS],
        "method_match": sum(b.get("method_check", {}).get("match", False) for b in decided),
        "method_checked": sum("method_check" in b for b in decided),
        "biggest_upset": min(decided, key=lambda b: b["p_winner"])["id"] if misses else None,
        "safest_hit": max(hits, key=lambda b: b["p_winner"])["id"] if hits else None,
    }
    fl = sorted(factors.values(), key=lambda g: (-(g["push_to_winner"] - g["push_to_loser"]), -g["bouts"]))
    for g in fl:
        g["hit_rate"] = g["hits"] / g["bouts"]
    el = [{"stat": s, "label": label, "n": edges[s]["n"], "won": edges[s]["won"],
           "rate": edges[s]["won"] / edges[s]["n"] if edges[s]["n"] else None,
           "historical": hist[s]["rate"], "historical_n": hist[s]["n"]} for s, label, _ in EDGES]
    card = {k: v for k, v in card.items() if k != "bouts"}
    return clean({**card, "bouts": bouts, "summary": summary, "factors": fl, "edges": el,
                  "data_through": predictor.latest, "built_at": pd.Timestamp.now(tz="UTC").isoformat()[:19]})


def load_or_build(event, predictor, matcher, processed, cache_dir="data/processed/cache"):
    """The frozen review if one exists; otherwise build + freeze it, if the data predates the event."""
    path = Path(processed) / "reviews" / f"{event['id']}.json"
    if path.exists():
        return json.loads(path.read_text())
    event_day = pd.Timestamp(event["date"]).tz_localize(None).normalize()
    if predictor.latest >= event_day:
        raise LeakError(f"our data (through {predictor.latest:%Y-%m-%d}) already includes {event['name']}, "
                        "and no pre-fight review was saved")
    out = review_card(event, predictor, matcher, processed, cache_dir=cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out
