"""Fighter lookups: search, profile card, career stats, recent fights, percentiles.

Everything hangs off one FighterDB object that loads the processed tables once,
which is also how a website backend would use it:

    db = FighterDB()
    db.search("adesanya")
    db.card("Israel Adesanya")
    db.last_fights("Israel Adesanya", n=5)
"""
import difflib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ufc import elo
from ufc.cleaning import norm_name

# (column, label, higher_is_better). None = a style mix, not better or worse.
STATS = [
    ("slpm", "Sig. strikes landed / min", True),
    ("sapm", "Sig. strikes absorbed / min", False),
    ("sig_acc", "Striking accuracy", True),
    ("sig_def", "Striking defense", True),
    ("kd_per15", "Knockdowns / 15 min", True),
    ("kd_absorbed_per15", "Knocked down / 15 min", False),
    ("td_per15", "Takedowns / 15 min", True),
    ("td_acc", "Takedown accuracy", True),
    ("td_def", "Takedown defense", True),
    ("sub_per15", "Sub attempts / 15 min", True),
    ("ctrl_share", "Time in control", True),
    ("ctrl_against", "Time controlled by opponent", False),
    ("finish_rate", "Wins by finish", True),
]
MIX = [("head_pct", "Head"), ("body_pct", "Body"), ("leg_pct", "Legs"),
       ("distance_pct", "Distance"), ("clinch_pct", "Clinch"), ("ground_pct", "Ground")]


def _masked_ratio(df, num, den):
    """sum(num)/sum(den) per fighter, using only fights where both were recorded."""
    ok = df[num].notna() & df[den].notna()
    n = df[num].where(ok, 0).groupby(df["fighter"]).sum()
    d = df[den].where(ok, 0).groupby(df["fighter"]).sum()
    return (n / d).where(d > 0)


def _clock(sec):
    return "" if pd.isna(sec) else f"{int(sec) // 60}:{int(sec) % 60:02d}"


def _streak_text(n, kind):
    if kind is None:
        return "–"
    word = {"W": ("win", "wins"), "L": ("loss", "losses"), "D": ("draw", "draws")}[kind]
    return f"{n} {word[0] if n == 1 else word[1]}"


class FighterDB:
    def __init__(self, processed="data/processed"):
        p = Path(processed)
        self.fights = pd.read_parquet(p / "fights.parquet")
        self.ff = pd.read_parquet(p / "fighter_fights.parquet")
        self.rounds = pd.read_parquet(p / "fight_rounds.parquet")
        self.directory = pd.read_parquet(p / "fighters.parquet").set_index("name")
        self.hist = pd.read_parquet(p / "elo_history.parquet")
        self.ff["minutes"] = self.ff["fight_seconds"] / 60
        self.latest = self.fights["date"].max()
        self._keys = pd.Series({n: norm_name(n) for n in self.directory.index})
        self._nick = self.directory["nickname"].fillna("").map(lambda s: norm_name(s) if s else "")

    # ------------------------------------------------------------------ search
    def search(self, query, n=10):
        """Accent/case-insensitive match on names and nicknames, then fuzzy fallback."""
        q = norm_name(query)
        hits = self._keys.index[self._keys.str.contains(q, regex=False)
                                | self._nick.reindex(self._keys.index).str.contains(q, regex=False)]
        if len(hits) == 0:
            close = difflib.get_close_matches(q, self._keys.tolist(), n=n, cutoff=0.6)
            hits = self._keys.index[self._keys.isin(close)]
        d = self.directory.loc[hits]
        out = pd.DataFrame({"nickname": d.nickname, "division": d.weight_class,
                            "ufc_fights": d.ufc_fights, "last_fight": d.last_fight.dt.date})
        return out.sort_values("ufc_fights", ascending=False).head(n)

    def _check(self, name):
        if name not in self.directory.index:
            raise KeyError(f"{name!r} not found. Try db.search({name.split()[-1]!r})")

    def _as_of(self, as_of):
        return pd.Timestamp(as_of) if as_of is not None else self.latest

    # ------------------------------------------------------------------ stats
    @lru_cache(maxsize=32)
    def career_stats(self, as_of=None):
        """Per-fighter career stats from UFC fights on or before `as_of` (one row per fighter)."""
        as_of = self._as_of(as_of)
        df = self.ff[self.ff.date <= as_of].copy()
        g = df.groupby("fighter")
        mins = g.minutes.sum()
        s = pd.DataFrame({
            "ufc_fights": g.size(), "minutes": mins,
            "wins": g.result.apply(lambda r: (r == "W").sum()),
            "slpm": g.sig_str.sum() / mins, "sapm": g.opp_sig_str.sum() / mins,
            "sig_acc": _masked_ratio(df, "sig_str", "sig_str_att"),
            "sig_def": 1 - _masked_ratio(df, "opp_sig_str", "opp_sig_str_att"),
            "kd_per15": 15 * g.kd.sum() / mins, "kd_absorbed_per15": 15 * g.opp_kd.sum() / mins,
            "td_per15": 15 * g.td.sum() / mins,
            "td_acc": _masked_ratio(df, "td", "td_att"),
            "td_def": 1 - _masked_ratio(df, "opp_td", "opp_td_att"),
            "sub_per15": 15 * g.sub_att.sum() / mins,
            "ctrl_share": _masked_ratio(df, "ctrl_sec", "fight_seconds"),
            "ctrl_against": _masked_ratio(df, "opp_ctrl_sec", "fight_seconds"),
        })
        fin = df[(df.result == "W") & df.method_group.isin(["KO/TKO", "SUB"])].groupby("fighter").size()
        s["finish_rate"] = (fin.reindex(s.index, fill_value=0) / s.wins).where(s.wins > 0)
        for col, _ in MIX[:3]:
            s[col] = _masked_ratio(df, col.replace("_pct", ""), "sig_str")
        for col, _ in MIX[3:]:
            s[col] = _masked_ratio(df, col.replace("_pct", ""), "sig_str")
        last = df.sort_values("fight_id").groupby("fighter").tail(1).set_index("fighter")
        s["division"] = last.weight_class
        s["last_fight"] = last.date
        return s.replace([np.inf, -np.inf], np.nan)

    def percentiles(self, name, as_of=None, min_fights=3, active_years=3):
        """The fighter's stats as percentiles vs division peers (100 = best in division)."""
        self._check(name)
        as_of = self._as_of(as_of)
        cs = self.career_stats(as_of)
        me = cs.loc[name]
        peers = cs[(cs.division == me.division) & (cs.ufc_fights >= min_fights)
                   & (cs.last_fight >= as_of - pd.DateOffset(years=active_years))]
        peers = pd.concat([peers, cs.loc[[name]]]).loc[lambda d: ~d.index.duplicated()]
        rows = []
        for col, label, better_high in STATS:
            vals = peers[col].dropna()
            if pd.isna(me[col]) or vals.empty:
                pct = np.nan
            else:
                pct = (vals < me[col]).mean() * 100 + (vals == me[col]).mean() * 50
                pct = pct if better_high else 100 - pct
            rows.append({"stat": col, "label": label, "value": me[col],
                         "division_median": vals.median(), "percentile": pct})
        out = pd.DataFrame(rows).set_index("stat")
        out.attrs.update(division=me.division, peers=len(peers))
        return out

    def style_mix(self, name, as_of=None):
        """Where the fighter's landed sig. strikes go (target) and come from (position), in %."""
        cs = self.career_stats(self._as_of(as_of))
        return pd.Series({label: cs.at[name, col] * 100 for col, label in MIX}).round(1)

    # ------------------------------------------------------------------ card
    def card(self, name, as_of=None):
        """Profile + record + rating summary as a labeled Series."""
        self._check(name)
        as_of = self._as_of(as_of)
        d = self.directory.loc[name]
        mine = self.ff[(self.ff.fighter == name) & (self.ff.date <= as_of)].sort_values("fight_id")
        res = mine.result
        wins, losses = mine[res == "W"], mine[res == "L"]
        by = lambda df: "  ".join(f"{m} {c}" for m, c in df.method_group.value_counts().items()) or "–"

        streak, kind = 0, None
        for r in reversed(res[res != "N"].tolist()):
            if kind is None:
                kind = r
            if r != kind or r == "D":
                break
            streak += 1
        h = self.hist[(self.hist.fighter == name) & (self.hist.date <= as_of)]
        rating = h.elo_post.iloc[-1] if len(h) else np.nan
        peak = h.loc[h.elo_post.idxmax()] if len(h) else None
        age = (as_of - d.dob).days / 365.25 if pd.notna(d.dob) else np.nan
        rank = elo.p4p_rank(self.hist, name, as_of) if len(h) else None

        rec = f"{len(wins)}-{len(losses)}"
        if (res == "D").any():
            rec += f"-{(res == 'D').sum()}"
        if (res == "N").any():
            rec += f" ({(res == 'N').sum()} NC)"
        return pd.Series({
            "Name": name, "Nickname": d.nickname if pd.notna(d.nickname) else "–",
            "Division": mine.weight_class.iloc[-1] if len(mine) else d.weight_class,
            "Age": f"{age:.1f}" if pd.notna(age) else "–",
            "Height": f"{d.height:.0f} cm" if pd.notna(d.height) else "–",
            "Reach": f"{d.reach:.0f} cm" if pd.notna(d.reach) else "–",
            "Stance": d.stance if pd.notna(d.stance) else "–",
            "UFC record": rec,
            "Wins by": by(wins), "Losses by": by(losses),
            "Current streak": _streak_text(streak, kind),
            "Title fights": f"{int(mine.is_title_bout.sum())} ({int((mine.is_title_bout.astype(bool) & (res == 'W')).sum())} won)",
            "UFC debut": mine.date.min().date() if len(mine) else "–",
            "Last fight": mine.date.max().date() if len(mine) else "–",
            "Elo rating": f"{rating:.0f}" if pd.notna(rating) else "–",
            "Peak Elo": f"{peak.elo_post:.0f} ({peak.date.date()})" if peak is not None else "–",
            "P4P rank": f"#{rank}" if rank else "unranked / inactive",
        }, name=name)

    # ------------------------------------------------------------------ fights
    def last_fights(self, name, n=5, as_of=None, raw=False):
        """Most recent n fights with outcome and key stats for both fighters."""
        self._check(name)
        as_of = self._as_of(as_of)
        mine = self.ff[(self.ff.fighter == name) & (self.ff.date <= as_of)]
        mine = mine.sort_values("fight_id", ascending=False).head(n)
        f = self.fights.set_index("fight_id")
        h = self.hist.set_index(["fight_id", "fighter"])
        t = mine.assign(
            method=mine.fight_id.map(f.method), detail=mine.fight_id.map(f.method_detail),
            time_sec=mine.fight_id.map(f.time_sec),
            opp_elo=[h.at[(i, name), "opp_elo_pre"] for i in mine.fight_id],
            elo_change=[h.at[(i, name), "elo_post"] - h.at[(i, name), "elo_pre"] for i in mine.fight_id])
        if raw:
            return t
        frac = lambda a, b: "" if pd.isna(a) else f"{int(a)}/{int(b)}"
        return pd.DataFrame({
            "Date": t.date.dt.date, "Event": t.event, "Opponent": t.opponent, "Result": t.result,
            "Method": t.method + np.where(t.detail.notna(), " (" + t.detail.fillna("") + ")", ""),
            "Rd": t.finish_round, "Time": t.time_sec.map(_clock),
            "Title": np.where(t.is_title_bout == 1, "🏆", ""),
            "Sig str": [frac(a, b) for a, b in zip(t.sig_str, t.sig_str_att)],
            "Opp sig str": [frac(a, b) for a, b in zip(t.opp_sig_str, t.opp_sig_str_att)],
            "TD": [frac(a, b) for a, b in zip(t.td, t.td_att)],
            "Opp TD": [frac(a, b) for a, b in zip(t.opp_td, t.opp_td_att)],
            "KD": t.kd.fillna(0).astype(int).astype(str) + "–" + t.opp_kd.fillna(0).astype(int).astype(str),
            "Ctrl": t.ctrl_sec.map(_clock), "Opp ctrl": t.opp_ctrl_sec.map(_clock),
            "Opp Elo": t.opp_elo.round(0).astype(int), "Elo ±": t.elo_change.round(0).astype(int),
            "fight_id": t.fight_id,
        }).reset_index(drop=True)

    def fight_rounds(self, name, fight_id):
        """Round-by-round stats of one fight from `name`'s point of view."""
        r = self.rounds[self.rounds.fight_id == fight_id]
        me, opp = r[r.fighter == name].set_index("round"), r[r.fighter != name].set_index("round")
        cols = {"sig_str": "Sig str", "head": "Head", "body": "Body", "leg": "Legs",
                "td": "TD", "kd": "KD", "ctrl_sec": "Ctrl s"}
        out = pd.concat({name: me[list(cols)].rename(columns=cols),
                         opp.fighter.iloc[0] if len(opp) else "Opponent": opp[list(cols)].rename(columns=cols)},
                        axis=1)
        out.index.name = "Round"
        return out

    def elo_history(self, name):
        return self.hist[self.hist.fighter == name].sort_values("fight_id")
