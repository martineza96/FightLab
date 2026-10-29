"""Predict fights that haven't happened yet, and explain each prediction.

How a future fight gets features: every fighter gets one placeholder "next fight" row,
dated after the last fight in the data, appended to the fighter-fight table. The same
builders used for training (features.add_history, form.build_all) then compute that row
from earlier fights only, so a live prediction uses exactly the training-time features.
Only age and days-since-last-fight depend on the actual fight date; they're adjusted on
the fly.

The model is notebook 07's final model (career rates shrunk toward the division mean +
"who they beat"), refit on every fight for live use. Predictions are corner-neutral
(the red-corner term is switched off), because upcoming-card listings don't say who is red.
"""
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ufc import form as F
from ufc import geo
from ufc import modeling as M
from ufc.features import HIST_FEATURES, MIN_FEATURES, add_history, build_model_table

SHRINK_K = 5
METHOD_K = 6
FEATS = MIN_FEATURES + HIST_FEATURES + F.OPP_FEATURES
LOCATION_COLS = ["home_abroad_diff", "us_event_diff"]          # notebook 09
FINAL_COLS = M.diffs(MIN_FEATURES + HIST_FEATURES) + M.FLAGS + ["red_corner"] + M.diffs(F.OPP_FEATURES) + LOCATION_COLS
EDGE_STATS = ["age_at_fight", "height", "reach", "n_fights", "win_streak", "elo_pre"]   # raw values for card reviews
STATE_COLS = sorted(set(MIN_FEATURES + HIST_FEATURES + F.ALL_NEW + ["weight_class", "gender", "stance"]))

# feature -> (phrase when the favored fighter has MORE of it, phrase when LESS, value format)
PHRASES = {
    "age_at_fight": ("Older", "Younger", "{:.1f} yrs"),
    "height": ("Taller", "Shorter", "{:.0f} cm"),
    "reach": ("Longer reach", "Shorter reach", "{:.0f} cm"),
    "n_fights": ("More UFC fights", "Fewer UFC fights", "{:.0f}"),
    "win_pct": ("Higher UFC win rate", "Lower UFC win rate", "{:.0%}"),
    "slpm": ("Lands more strikes / min", "Lands fewer strikes / min", "{:.2f}"),
    "sapm": ("Absorbs more strikes / min", "Absorbs fewer strikes / min", "{:.2f}"),
    "sig_acc": ("More accurate striker", "Less accurate striker", "{:.0%}"),
    "sig_def": ("Better striking defense", "Worse striking defense", "{:.0%}"),
    "td_per15": ("More takedowns / 15 min", "Fewer takedowns / 15 min", "{:.2f}"),
    "td_acc": ("Better takedown accuracy", "Worse takedown accuracy", "{:.0%}"),
    "td_def": ("Better takedown defense", "Worse takedown defense", "{:.0%}"),
    "sub_per15": ("More submission attempts", "Fewer submission attempts", "{:.2f} / 15 min"),
    "kd_per15": ("More knockdowns", "Fewer knockdowns", "{:.2f} / 15 min"),
    "ctrl_share": ("More control time", "Less control time", "{:.0%}"),
    "ko_wins": ("More KO wins", "Fewer KO wins", "{:.0f}"),
    "sub_wins": ("More submission wins", "Fewer submission wins", "{:.0f}"),
    "dec_wins": ("More decision wins", "Fewer decision wins", "{:.0f}"),
    "been_finished": ("Finished more often", "Finished less often", "{:.0f} times"),
    "win_streak": ("Longer win streak", "Shorter win streak", "{:.0f}"),
    "lose_streak": ("Longer losing streak", "Shorter losing streak", "{:.0f}"),
    "days_since_last": ("Longer layoff", "Shorter layoff", "{:.0f} days"),
    "ufc_minutes": ("More cage time", "Less cage time", "{:.0f} min"),
    "beat_streak_max": ("Snapped a longer win streak", "Snapped a shorter win streak", "{:.0f} fights"),
    "last_opp_streak": ("Last opponent was hotter", "Last opponent was colder", "{:.0f}-fight streak"),
    "last_result_w": ("Won last fight", "Lost last fight", "{:.0f}"),
    "quality_wins": ("Beat higher-rated opponents", "Beat lower-rated opponents", "Elo {:.0f}"),
    "quality_losses": ("Lost only to better opponents", "Lost to weaker opponents", "Elo {:.0f}"),
    "opp_form_avg": ("Faced opponents in better form", "Faced opponents in worse form", "{:.0%}"),
}


def next_fight_rows(ff0, hist, directory):
    """Append one placeholder 'next fight' per fighter (dated the day after the data ends).

    Returns (fighter-fight table + placeholders, Elo table + placeholders, each fighter's last row).
    Placeholder Elo-before = the fighter's current rating; result and stats are empty.
    """
    latest = ff0.date.max()
    last = ff0.sort_values(["date", "fight_id"]).groupby("fighter").tail(1).set_index("fighter")
    names = last.index
    nxt = pd.DataFrame({
        "fight_id": ff0.fight_id.max() + 1 + np.arange(len(names)),
        "date": latest + pd.Timedelta(days=1), "event": "next", "corner": "Red",
        "fighter": names, "opponent": "__next__", "result": np.nan, "method_group": np.nan,
        "weight_class": last.weight_class.values, "gender": last.gender.values,
        "is_title_bout": 0, "finish_round": 0, "total_rounds": 3, "fight_seconds": np.nan,
        "height": directory.height.reindex(names).values, "reach": directory.reach.reindex(names).values,
        "stance": directory.stance.reindex(names).values,
        "age_at_fight": ((latest - directory.dob.reindex(names)).dt.days / 365.25).values,
    })
    elo_now = hist.sort_values("fight_id").groupby("fighter").elo_post.last()
    h_next = pd.DataFrame({"fight_id": nxt.fight_id.values, "fighter": names,
                           "elo_pre": elo_now.reindex(names).values, "elo_post": np.nan, "opp_elo_pre": np.nan,
                           "date": nxt.date.values})
    ff_aug = pd.concat([ff0, nxt], ignore_index=True)
    hist_aug = pd.concat([hist, h_next], ignore_index=True)
    return ff_aug, hist_aug, last


def _fingerprint(processed):
    h = hashlib.md5()
    for f in ["fighter_fights.parquet", "elo_history.parquet", "fighters.parquet", "fights.parquet", "fighter_origin.parquet"]:
        p = Path(processed) / f
        h.update(f"{f}:{p.stat().st_mtime_ns}:{p.stat().st_size}".encode())
    return h.hexdigest()[:12]


class Predictor:
    """Fits (or loads) the live model and holds every fighter's 'next fight' state."""

    def __init__(self, processed="data/processed", cache=True):
        self.processed = Path(processed)
        key = _fingerprint(processed)
        path = self.processed / "models" / f"predictor_{key}.joblib"
        if cache and path.exists():
            self.__dict__.update(joblib.load(path))
            return
        self._build()
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({k: v for k, v in self.__dict__.items() if k != "processed"}, path)

    # ------------------------------------------------------------------ build
    def _build(self):
        p = self.processed
        ff0 = pd.read_parquet(p / "fighter_fights.parquet")
        hist = pd.read_parquet(p / "elo_history.parquet")
        fights = pd.read_parquet(p / "fights.parquet")
        directory = pd.read_parquet(p / "fighters.parquet").drop_duplicates("name").set_index("name")
        self.latest = ff0.date.max()

        ff_aug, hist_aug, last = next_fight_rows(ff0, hist, directory)
        ff = add_history(ff_aug).merge(F.build_all(ff_aug, hist_aug), on=F.KEY, how="left")
        real = ff.fight_id <= ff0.fight_id.max()
        raw_state = ff[~real].set_index("fighter")
        ff = M.shrink(ff, M.RATE_FEATURES, "n_fights", "weight_class", SHRINK_K, ref=ff[real])
        state = ff[~real].set_index("fighter")

        # live model: every fight with clean corners, train through the latest date
        t = build_model_table(fights, ff[real], extra=F.ALL_NEW)
        origin = pd.read_parquet(p / "fighter_origin.parquet")
        t = add_location(t, ff0, fights, origin)
        self.model = M.fit(M.logistic(), t, FINAL_COLS)
        self.country = origin.set_index("name").country.dropna()
        decided = t[t.winner.isin(["Red", "Blue"])]
        meth = fights.set_index("fight_id").method_group.reindex(decided.fight_id)
        self.method_prior = meth[meth.isin(M.METHODS)].value_counts(normalize=True).to_dict()
        self.train_size = len(t)

        self.state = state[STATE_COLS].copy()
        self.raw_state = raw_state[STATE_COLS].copy()
        self.last_date = last.date
        self.dob = directory.dob
        self.elo_now = hist.sort_values("fight_id").groupby("fighter").elo_post.last()
        self.directory = directory[["nickname", "weight_class", "height", "reach", "stance", "dob", "ufc_fights"]]
        # neutral "debutant" state: no history; rates shrink fully to the division mean
        self.division_means = ff[real].groupby("weight_class")[M.RATE_FEATURES].mean()

    # ------------------------------------------------------------------ state
    def fighter_state(self, name, date, profile=None, weight_class=None):
        """Feature values going into a fight on `date`. Unknown names = UFC debut.

        profile (debutants only): dict with optional dob / height / reach / stance.
        """
        date = pd.Timestamp(date)
        if name in self.state.index:
            s = self.state.loc[name].copy()
            raw = self.raw_state.loc[name].copy()
            dob = self.dob.get(name)
            if pd.notna(dob):
                s["age_at_fight"] = raw["age_at_fight"] = (date - dob).days / 365.25
            s["days_since_last"] = raw["days_since_last"] = (date - self.last_date[name]).days
            return s, raw, False
        s = pd.Series(np.nan, index=STATE_COLS, dtype=object)
        for c in ["n_fights", "ko_wins", "sub_wins", "dec_wins", "been_finished", "win_streak", "lose_streak",
                  "ufc_minutes", "l5_fights", "l5_ko_w", "l5_sub_w", "l5_dec_w", "l5_ko_l", "l5_sub_l", "l5_dec_l"]:
            s[c] = 0.0
        wc = weight_class if weight_class in self.division_means.index else None
        raw = s.copy()
        for c in M.RATE_FEATURES:
            s[c] = self.division_means.loc[wc, c] if wc else self.division_means[c].mean()
        profile = profile or {}
        if profile.get("dob") is not None and pd.notna(profile.get("dob")):
            s["age_at_fight"] = raw["age_at_fight"] = (date - pd.Timestamp(profile["dob"])).days / 365.25
        for k in ["height", "reach", "stance"]:
            if profile.get(k) is not None:
                s[k] = raw[k] = profile[k]
        s["weight_class"] = raw["weight_class"] = weight_class
        return s.astype(object), raw.astype(object), True

    def _row(self, rs, bs, location=(0.0, 0.0)):
        row = {}
        for f in FEATS:
            rv, bv = pd.to_numeric(rs.get(f), errors="coerce"), pd.to_numeric(bs.get(f), errors="coerce")
            row[f"r_{f}"], row[f"b_{f}"] = rv, bv
            row[f"{f}_diff"] = rv - bv
        row["r_debut"] = int((rs.get("n_fights") or 0) == 0)
        row["b_debut"] = int((bs.get("n_fights") or 0) == 0)
        row["reach_missing"] = int(pd.isna(row["r_reach"]) or pd.isna(row["b_reach"]))
        row["red_corner"] = 0           # corner-neutral
        row["home_abroad_diff"], row["us_event_diff"] = location
        return pd.DataFrame([row])

    # ------------------------------------------------------------------ predict
    def fighter_country(self, name, profile=None):
        if name in self.country.index:
            return self.country[name]
        return geo.norm_country((profile or {}).get("country"))

    def predict(self, red, blue, date=None, rounds=3, red_profile=None, blue_profile=None,
                weight_class=None, n_drivers=8, country=None):
        """country = where the fight takes place (None = unknown / neutral)."""
        date = pd.Timestamp(date) if date is not None else self.latest + pd.Timedelta(days=1)
        rs, rraw, rdeb = self.fighter_state(red, date, red_profile, weight_class)
        bs, braw, bdeb = self.fighter_state(blue, date, blue_profile, weight_class)
        country = geo.norm_country(country)
        rc, bc = self.fighter_country(red, red_profile), self.fighter_country(blue, blue_profile)
        X = self._row(rs, bs, location_features(country, rc, bc))
        p_red = float(self.model.predict_proba(X[FINAL_COLS])[:, 1][0])

        # method: P(winner) × winner's shrunk method mix
        mdf = X.copy()
        for side, s in [("r_", rs), ("b_", bs)]:
            for c in ["l5_ko_l", "l5_sub_l", "l5_dec_l"]:
                mdf[side + c] = pd.to_numeric(s.get(c), errors="coerce")
        probs = M.method_probs(mdf, np.array([p_red]), self.method_prior, k=METHOD_K, loser_weight=0).iloc[0]
        outcomes = [{"winner": red if o.startswith("Red") else blue if o.startswith("Blue") else None,
                     "corner": o.split()[0].lower() if o != "Other" else None,
                     "method": o.split(" ", 1)[1] if o != "Other" else "Other", "p": float(v)}
                    for o, v in probs.items()]

        drivers = self._drivers(X, rraw, braw, rs, bs, red, blue, (country, rc, bc))
        flags = []
        for name, deb, s in [(red, rdeb, rs), (blue, bdeb, bs)]:
            n = float(pd.to_numeric(s.get("n_fights"), errors="coerce") or 0)
            if deb or n == 0:
                flags.append(f"{name}: UFC debut, so no UFC history; prediction leans on age and size")
            elif n < 3:
                flags.append(f"{name}: only {int(n)} UFC fight{'s' if n > 1 else ''}; stats are pulled toward the division average")
        if X.reach_missing.iloc[0]:
            flags.append("Reach unknown for at least one fighter")
        return {
            "red": red, "blue": blue, "date": date.date().isoformat(), "rounds": rounds,
            "p_red": p_red, "p_blue": 1 - p_red,
            "favorite": red if p_red >= 0.5 else blue,
            "confidence": abs(p_red - 0.5) * 2,
            "outcomes": sorted(outcomes, key=lambda o: -o["p"]),
            "drivers": drivers[:n_drivers], "all_drivers": drivers,
            "tale": [{"stat": "From", "red": rc or "–", "blue": bc or "–"}] + self._tale(rraw, braw, red, blue, rdeb, bdeb),
            "location": {"country": country, "red_home": bool(country and rc == country), "blue_home": bool(country and bc == country)},
            "debut": {"red": rdeb, "blue": bdeb}, "flags": flags,
            "stats": {side: {k: _num(raw.get(k)) for k in EDGE_STATS} for side, raw in [("red", rraw), ("blue", braw)]},
        }

    def _drivers(self, X, rraw, braw, rs, bs, red, blue, where=(None, None, None)):
        contrib, b0 = M.contributions(self.model, X[FINAL_COLS])
        c = contrib.iloc[0]
        groups = {}
        for col, v in c.items():
            if col == "red_corner":
                continue
            base = col.replace("missing: ", "").replace("_diff", "")
            if base in ("r_debut", "b_debut"):
                base = "debut"
            if base == "reach_missing":
                base = "reach"
            groups[base] = groups.get(base, 0.0) + float(v)
        out = []
        for f, push in groups.items():
            if abs(push) < 1e-4:
                continue
            fav, oth = (red, blue) if push > 0 else (blue, red)
            fv, ov = (rraw.get(f), braw.get(f)) if push > 0 else (braw.get(f), rraw.get(f))
            fv, ov = pd.to_numeric(fv, errors="coerce"), pd.to_numeric(ov, errors="coerce")
            # a debutant's rates are the division average (shrinkage); say so instead of "unknown"
            fs, os_ = (rs.get(f), bs.get(f)) if push > 0 else (bs.get(f), rs.get(f))
            f_avg, o_avg = pd.isna(fv) and pd.notna(pd.to_numeric(fs, errors="coerce")), pd.isna(ov) and pd.notna(pd.to_numeric(os_, errors="coerce"))
            if f_avg:
                fv = pd.to_numeric(fs, errors="coerce")
            if o_avg:
                ov = pd.to_numeric(os_, errors="coerce")
            if f in ("home_abroad", "us_event"):
                event_c, rc, bc = where
                fav_c = rc if push > 0 else bc
                if f == "home_abroad":
                    label, detail = f"Home crowd ({event_c})", "home fighters outside the USA beat their ratings by ~9 pts (notebook 09)"
                else:
                    label = "Visiting fighter at a US event" if fav_c != "USA" else "Home fighter at a US event"
                    detail = "foreign fighters have beaten their ratings at US events (notebook 09)"
                out.append({"feature": f, "label": label, "detail": detail, "favors": fav, "side": "red" if push > 0 else "blue",
                            "push": push, "size": abs(push)})
                continue
            if f == "debut":
                fav_debut = bool(X.r_debut.iloc[0]) if push > 0 else bool(X.b_debut.iloc[0])
                if fav_debut:
                    label, detail = "Debuting prospect", "newcomers booked against UFC veterans win more than records suggest"
                else:
                    label, detail = "Opponent is a UFC debutant", ""
            elif f in PHRASES:
                more, less, fmt = PHRASES[f]
                if pd.isna(fv) or pd.isna(ov):
                    debut_side = bool(X.r_debut.iloc[0] or X.b_debut.iloc[0])
                    label = f"No UFC {_nice(f)} yet (debut)" if debut_side else f"{_nice(f).capitalize()} unknown for one fighter"
                    detail = ""
                else:
                    label = more if fv > ov else less if fv < ov else _nice(f)
                    detail = (f"{fmt.format(fv)}{' (division avg)' if f_avg else ''} vs "
                              f"{fmt.format(ov)}{' (division avg)' if o_avg else ''}")
            else:
                label, detail = _nice(f), ""
            out.append({"feature": f, "label": label, "detail": detail, "favors": fav,
                        "side": "red" if push > 0 else "blue", "push": push, "size": abs(push)})
        return sorted(out, key=lambda d: -d["size"])

    def _tale(self, r, b, red, blue, rdeb, bdeb):
        rows = [("Age", "age_at_fight", "{:.1f}"), ("Height (cm)", "height", "{:.0f}"), ("Reach (cm)", "reach", "{:.0f}"),
                ("Stance", "stance", "{}"), ("UFC fights", "n_fights", "{:.0f}"), ("UFC win rate", "win_pct", "{:.0%}"),
                ("Win streak", "win_streak", "{:.0f}"), ("Last 5", "l5_win_rate", None),
                ("Sig. strikes landed / min", "slpm", "{:.2f}"), ("Sig. strikes absorbed / min", "sapm", "{:.2f}"),
                ("Striking defense", "sig_def", "{:.0%}"), ("Takedowns / 15 min", "td_per15", "{:.2f}"),
                ("Takedown defense", "td_def", "{:.0%}"), ("Sub attempts / 15 min", "sub_per15", "{:.2f}"),
                ("Elo", "elo_pre", "{:.0f}"), ("Avg Elo of opponents beaten", "quality_wins", "{:.0f}"),
                ("Opponents' win rate (OWP)", "owp", "{:.0%}")]
        def fmt(s, col, f):
            if col == "l5_win_rate":
                n = pd.to_numeric(s.get("l5_fights"), errors="coerce")
                w = pd.to_numeric(s.get("l5_win_rate"), errors="coerce")
                return "–" if pd.isna(n) or n == 0 or pd.isna(w) else f"{w * n:.0f}-{n - w * n:.0f}"
            v = s.get(col)
            if v is None or (not isinstance(v, str) and pd.isna(v)):
                return "–"
            return f.format(v) if not isinstance(v, str) else v
        return [{"stat": label, "red": fmt(r, col, f), "blue": fmt(b, col, f)} for label, col, f in rows]


NICE = {"age_at_fight": "age", "win_pct": "win rate", "slpm": "strikes landed / min", "sapm": "strikes absorbed / min",
        "sig_acc": "striking accuracy", "sig_def": "striking defense", "td_per15": "takedowns", "td_acc": "takedown accuracy",
        "td_def": "takedown defense", "sub_per15": "submission attempts", "kd_per15": "knockdowns", "ctrl_share": "control time",
        "quality_wins": "quality wins", "quality_losses": "quality losses", "opp_form_avg": "opponents' form",
        "beat_streak_max": "streaks snapped", "last_opp_streak": "last opponent's streak", "last_result_w": "last result",
        "days_since_last": "layoff"}


def location_features(event_country, red_country, blue_country):
    """(home_abroad_diff, us_event_diff) for one fight. Unknown location -> (0, 0)."""
    if not event_country:
        return 0.0, 0.0
    hd = float(red_country == event_country) - float(blue_country == event_country)
    if red_country is None or blue_country is None:
        hd = float(red_country == event_country) if red_country else -float(blue_country == event_country) if blue_country else 0.0
    return (hd, 0.0) if event_country != "USA" else (0.0, hd)


def add_location(t, ff, fights, origin):
    """Notebook 09's location features on a model table (crowd-less 2020-21 fights get no home-abroad credit)."""
    hf = geo.home_features(ff, fights, origin).merge(ff[["fight_id", "fighter", "corner"]], on=["fight_id", "fighter"])
    loc = hf.pivot_table(index="fight_id", columns="corner", values="home", aggfunc="first")
    hd = t.fight_id.map(loc.get("Red")) - t.fight_id.map(loc.get("Blue"))
    ec = t.fight_id.map(fights.set_index("fight_id").location.map(geo.event_country))
    crowd = (~t.date.between(geo.EMPTY_FROM, geo.EMPTY_TO)).astype(float)
    t = t.copy()
    # unknown origin = no home information (0), so the model doesn't learn a one-sided "missing" flag
    t["home_abroad_diff"] = (hd.where(ec != "USA", 0) * crowd).fillna(0)
    t["us_event_diff"] = hd.where(ec == "USA", 0).fillna(0)
    return t


def _num(v):
    v = pd.to_numeric(v, errors="coerce")
    return None if pd.isna(v) else float(v)


def _nice(f):
    return NICE.get(f, f.replace("_", " "))
