"""Ingest the Greco1899/scrape_ufc_stats CSVs (ufcstats.com data, updated regularly).

ufcstats.com itself now sits behind a JavaScript bot challenge, so we don't scrape it;
this public repo republishes the same data with round-by-round stats and fighter DOBs.
Source: https://github.com/Greco1899/scrape_ufc_stats (GPL-3.0)

Outputs the same `fights` / `fighter_fights` schema as cleaning.py, plus:
  fight_rounds.parquet   one row per fighter per round
  judge_scores.parquet   one row per judge per decision (whole-fight scorecards)
"""
import datetime as dt
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ufc.cleaning import (CORNERS_RELIABLE_FROM, EXTRA_STATS, FIGHT_STATS, METHOD_GROUP,
                          build_fighter_fights, clean_weight_class, fix_mojibake)

REPO = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/"
FILES = ["ufc_event_details", "ufc_fight_details", "ufc_fight_results", "ufc_fight_stats",
         "ufc_fighter_details", "ufc_fighter_tott"]

# Upper weight limit (lbs) per division, used to tell apart fighters who share a name.
DIVISION_LBS = {"Strawweight": 115, "Flyweight": 125, "Bantamweight": 135, "Featherweight": 145,
                "Lightweight": 155, "Welterweight": 170, "Middleweight": 185,
                "Light Heavyweight": 205, "Heavyweight": 265}

# Bout-name spelling -> fighter-profile spelling, where normalization alone can't tell.
MANUAL_ALIASES = {"Cam Nelson": "Cameron Nelson", "Shem Rock": "Shaqueme Rock",
                  "Zach Reese": "Zachary Reese", "Rafael Cerquiera": "Rafael Cerqueira",
                  "Ilimbek Akylbek": "Ilimbek Akylbek Uulu"}

METHOD_GROUP_NEW ={**METHOD_GROUP, "Could Not Continue": "Other", "Overturned": "Other"}

# Raw stat column -> our name. "X of Y" columns become X (landed) and Y (attempted).
OF_COLS = {"SIG.STR.": "sig_str", "TOTAL STR.": "str", "TD": "td", "HEAD": "head",
           "BODY": "body", "LEG": "leg", "DISTANCE": "distance", "CLINCH": "clinch",
           "GROUND": "ground"}
COUNT_COLS = {"KD": "kd", "SUB.ATT": "sub_att", "REV.": "rev"}


# ----------------------------------------------------------------------------- download
def download(base_dir="data/raw/greco", date=None):
    """Save a dated snapshot of the repo's CSVs; returns the snapshot folder."""
    out = Path(base_dir) / (date or dt.date.today().isoformat())
    out.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        urllib.request.urlretrieve(f"{REPO}{f}.csv", out / f"{f}.csv")
    return out


def latest_snapshot(base_dir="data/raw/greco"):
    return sorted(p for p in Path(base_dir).iterdir() if p.is_dir())[-1]


# ----------------------------------------------------------------------------- parsers
def parse_of(s):
    """'9 of 37' -> (9, 37)."""
    landed, att = s.str.extract(r"(\d+)\s+of\s+(\d+)").astype(float).T.values
    return landed, att


def parse_clock(s):
    """'1:23' -> 83 seconds; '--' -> NaN."""
    mm = s.astype(str).str.extract(r"(\d+):(\d+)").astype(float)
    return mm[0] * 60 + mm[1]


def parse_height_cm(s):
    """5' 11\" -> 180.34"""
    ftin = s.astype(str).str.extract(r"(\d+)'\s*(\d+)")
    return ((ftin[0].astype(float) * 12 + ftin[1].astype(float)) * 2.54).round(2)


def parse_inches_cm(s):
    return (s.astype(str).str.extract(r"([\d.]+)")[0].astype(float) * 2.54).round(2)


def parse_lbs_kg(s):
    return (s.astype(str).str.extract(r"([\d.]+)")[0].astype(float) * 0.45359237).round(2)


def round_lengths(time_format):
    """'3 Rnd (5-5-5)' -> [5, 5, 5] minutes; 'No Time Limit' -> []."""
    m = re.search(r"\(([\d\-]+)\)", str(time_format))
    return [int(x) for x in m.group(1).split("-")] if m else []


def parse_scorecards(details):
    """'Judge A 28 - 29.Judge B 27 - 30.' -> [(judge, first score, second score), ...]

    ufcstats writes each card as LOSER score - WINNER score, regardless of corner
    (a split decision reads '28 - 29. 29 - 28. 28 - 29.'), so corners are assigned later
    from the fight's winner. For draws the order carries no corner information.
    """
    return [(j.strip(), int(a), int(b))
            for j, a, b in re.findall(r"([^.\d][^.]*?)\s+(\d+)\s*-\s*(\d+)\.", str(details))]


# ----------------------------------------------------------------------------- build
def _load(snapshot):
    snapshot = Path(snapshot)
    read = lambda f: pd.read_csv(snapshot / f"{f}.csv")
    ev, res, st, tott = (read(f) for f in ["ufc_event_details", "ufc_fight_results",
                                            "ufc_fight_stats", "ufc_fighter_tott"])
    for df in (ev, res, st):
        for c in ["EVENT", "BOUT"]:
            if c in df:
                df[c] = df[c].str.strip().map(fix_mojibake)
    for c in ["METHOD", "WEIGHTCLASS", "DETAILS", "TIME FORMAT", "REFEREE"]:
        res[c] = res[c].str.strip()
    return ev, res, st, tott


def _attach_dates(res, ev):
    """Join event dates; a few results use an alternate event prefix ('UFC Fight Night:' vs
    'Noche UFC:'), so fall back to matching the part after the colon."""
    ev = ev.assign(date=pd.to_datetime(ev["DATE"], format="%B %d, %Y"))
    by_name = ev.set_index("EVENT")["date"]
    by_suffix = ev.assign(suf=ev.EVENT.str.split(":").str[-1].str.strip()) \
                  .drop_duplicates("suf", keep=False).set_index("suf")["date"]
    loc = ev.set_index("EVENT")["LOCATION"]
    res["date"] = res["EVENT"].map(by_name)
    miss = res["date"].isna()
    res.loc[miss, "date"] = res.loc[miss, "EVENT"].str.split(":").str[-1].str.strip().map(by_suffix)
    res["location"] = res["EVENT"].map(loc)
    return res


def _name_key(name):
    """'Levi Rodrigues Jr.' / 'Waldo Cortes-Acosta' -> 'levirodrigues' / 'waldocortesacosta'"""
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(name).lower())
    return re.sub(r"[^a-z]", "", n)


def _fighter_table(tott):
    t = tott.copy()
    t["height"] = parse_height_cm(t["HEIGHT"])
    t["reach"] = parse_inches_cm(t["REACH"])
    t["weight"] = parse_lbs_kg(t["WEIGHT"])
    t["weight_lbs"] = t["WEIGHT"].astype(str).str.extract(r"([\d.]+)")[0].astype(float)
    t["dob"] = pd.to_datetime(t["DOB"], format="%b %d, %Y", errors="coerce")
    t["stance"] = t["STANCE"].replace({"Sideways": "Open Stance"})
    t["fighter_url_id"] = t["URL"].str.rsplit("/", n=1).str[-1]
    return t.rename(columns={"FIGHTER": "name"})[
        ["name", "fighter_url_id", "height", "reach", "weight", "weight_lbs", "dob", "stance"]]


def _resolve_fighters(long, fighters):
    """Pick the right physical-profile row for each (fight, fighter name).

    - Names missing from the profile table: match after removing suffixes/spaces/hyphens
      ('YiSak Lee' -> 'Yi Sak Lee', 'Sean King III' -> 'Sean King'), plus a short manual list
      of nicknames. (Plain fuzzy matching is risky: it maps 'Muhammad Saidov' to a different
      fighter, 'Muhammad Said'.)
    - Names shared by different people: choose the row whose listed weight is closest to
      the bout's division limit (and whose DOB gives a plausible age); the fighter's display
      name gets a birth-year suffix so their histories stay separate.
    """
    names = set(fighters["name"])
    by_key = {}
    for n in names:
        by_key.setdefault(_name_key(n), []).append(n)
    alias = {}
    for n in set(long["name"]) - names:
        if n in MANUAL_ALIASES:
            alias[n] = MANUAL_ALIASES[n]
        elif len(by_key.get(_name_key(n), [])) == 1:
            alias[n] = by_key[_name_key(n)][0]
    long["lookup"] = long["name"].map(lambda n: alias.get(n, n))

    counts = fighters["name"].value_counts()
    dup_names = set(counts[counts > 1].index)
    single = fighters[~fighters.name.isin(dup_names)].set_index("name")
    long["profile_row"] = long["lookup"].map(pd.Series(single.index.map(
        lambda n: fighters.index[fighters.name == n][0]), index=single.index))

    for i in long.index[long["lookup"].isin(dup_names)]:
        cands = fighters[fighters.name == long.at[i, "lookup"]]
        limit = DIVISION_LBS.get(long.at[i, "weight_class"].replace("Women's ", ""))
        age = (long.at[i, "date"] - cands["dob"]).dt.days / 365.25
        plausible = cands[age.between(18, 48) | age.isna()]
        cands = plausible if len(plausible) else cands
        if limit is not None and cands["weight_lbs"].notna().any():
            pick = (cands["weight_lbs"] - limit).abs().idxmin()
        else:
            pick = cands["weight_lbs"].notna().idxmax()
        long.at[i, "profile_row"] = pick

    # Separate display names only where one name really maps to 2+ people in the fight data.
    used = long.dropna(subset=["profile_row"]).groupby("name")["profile_row"].nunique()
    for n in used[used > 1].index:
        m = long["name"] == n
        yrs = fighters.loc[long.loc[m, "profile_row"].astype(int), "dob"].dt.year
        long.loc[m, "name"] = [f"{n} ({int(y)})" if pd.notna(y) else f"{n} (#{int(r)})"
                               for y, r in zip(yrs, long.loc[m, "profile_row"])]
    return long, alias


def _directory(long, snapshot):
    """One row per fighter (as named in the fight data): profile + UFC career span."""
    long = long.sort_values("date")
    g = long.groupby("name")
    d = g.agg(fighter_url_id=("fighter_url_id", "last"), dob=("dob", "last"),
              height=("height", "last"), reach=("reach", "last"), stance=("stance", "last"),
              weight_class=("weight_class", "last"), first_fight=("date", "min"),
              last_fight=("date", "max"), ufc_fights=("date", "size")).reset_index()
    det = pd.read_csv(snapshot / "ufc_fighter_details.csv")
    det["fighter_url_id"] = det["URL"].str.rsplit("/", n=1).str[-1]
    nick = det.drop_duplicates("fighter_url_id").set_index("fighter_url_id")["NICKNAME"]
    d.insert(1, "nickname", d["fighter_url_id"].map(nick))
    return d


def build(snapshot):
    ev, res, st, tott = _load(snapshot)
    fighters = _fighter_table(tott)

    # --- fights: results + dates
    res = _attach_dates(res, ev)
    res["fight_url_id"] = res["URL"].str.rsplit("/", n=1).str[-1]
    # Some events appear twice under alternate names ('Noche UFC:' / 'UFC Fight Night:');
    # the fight URL is the true ID, so keep the copy whose event name matches the event list.
    res["_exact"] = res["EVENT"].isin(set(ev["EVENT"]))
    res = res.sort_values("_exact", ascending=False, kind="stable").drop_duplicates("fight_url_id")
    undated = res[res["date"].isna()]
    res = res[res["date"].notna()].copy()
    # one event lists the same bout twice (a restarted fight); keep the decided entry
    res = res.sort_values("OUTCOME").drop_duplicates(["EVENT", "BOUT"], keep="last")
    ab = res["BOUT"].str.split(" vs. ", n=1, expand=True)
    res["r_name"], res["b_name"] = ab[0].str.strip(), ab[1].str.strip()

    res["weight_class_raw"] = res["WEIGHTCLASS"]
    res["weight_class"] = res["WEIGHTCLASS"].map(clean_weight_class)
    res["is_title_bout"] = res["WEIGHTCLASS"].str.contains("Title").astype(int)
    res["winner"] = res["OUTCOME"].map({"W/L": "Red", "L/W": "Blue", "D/D": "Draw", "NC/NC": "NC"})
    res["method"] = res["METHOD"]
    res["method_group"] = res["METHOD"].map(METHOD_GROUP_NEW)
    res["method_detail"] = res["DETAILS"].where(~res["METHOD"].str.startswith("Decision"))
    res["finish_round"] = res["ROUND"].astype(int)
    res["time_sec"] = parse_clock(res["TIME"])
    res["time_format"] = res["TIME FORMAT"]
    lengths = res["TIME FORMAT"].map(round_lengths)
    res["total_rounds"] = lengths.map(len).replace(0, 1)
    res["fight_seconds"] = [sum(L[:r - 1]) * 60 + t if L else t
                            for L, r, t in zip(lengths, res["finish_round"], res["time_sec"])]
    res["referee"] = res["REFEREE"]
    res["bout_order"] = res.groupby("EVENT").cumcount(ascending=False)
    res = res.rename(columns={"EVENT": "event"})

    # --- gender: a fighter is "Women" if any of their bouts is a women's division
    women = set(res.loc[res.weight_class.str.startswith("Women"), ["r_name", "b_name"]].values.ravel())
    res["gender"] = np.where(res.weight_class.str.startswith("Women") | res.r_name.isin(women)
                             | res.b_name.isin(women), "Women", "Men")

    # --- resolve fighter identities / physical profiles
    long = pd.concat([res[["fight_url_id", "date", "weight_class"]].assign(side=s, name=res[f"{s}_name"])
                      for s in "rb"], ignore_index=True)
    long, alias = _resolve_fighters(long, fighters)
    prof = fighters.loc[long["profile_row"].dropna().astype(int)].set_index(long["profile_row"].dropna().index)
    for c in ["height", "reach", "weight", "stance", "dob", "fighter_url_id"]:
        long[c] = prof[c]
    long["age_at_fight"] = ((long["date"] - long["dob"]).dt.days / 365.25).round(1)
    directory = _directory(long, Path(snapshot))
    for s in "rb":
        side = long[long.side == s].set_index("fight_url_id")
        res[f"{s}_fighter"] = res["fight_url_id"].map(side["name"])
        for c in ["height", "reach", "weight", "stance", "age_at_fight", "fighter_url_id"]:
            res[f"{s}_{c}"] = res["fight_url_id"].map(side[c])

    # --- per-round stats
    st = st.dropna(subset=["FIGHTER"]).copy()
    st["FIGHTER"] = st["FIGHTER"].str.strip()
    st["round"] = st["ROUND"].str.extract(r"(\d+)")[0].astype(int)
    for raw, name in OF_COLS.items():
        st[name], st[f"{name}_att"] = parse_of(st[raw])
    for raw, name in COUNT_COLS.items():
        st[name] = st[raw].astype(float)
    st["ctrl_sec"] = parse_clock(st["CTRL"])
    key = res.set_index(["event", "BOUT"])
    idx = pd.MultiIndex.from_arrays([st["EVENT"], st["BOUT"]])
    st["fight_url_id"] = key["fight_url_id"].reindex(idx).values
    st = st.dropna(subset=["fight_url_id"])
    r_name = res.set_index("fight_url_id")["r_name"]
    st["corner"] = np.where(st["FIGHTER"] == st["fight_url_id"].map(r_name), "Red", "Blue")
    stat_cols = FIGHT_STATS + EXTRA_STATS
    rounds = st[["fight_url_id", "corner", "round"] + stat_cols].copy()

    # --- fight totals per corner
    tot = rounds.groupby(["fight_url_id", "corner"])[stat_cols].sum(min_count=1)
    for s, color in [("r", "Red"), ("b", "Blue")]:
        part = tot.xs(color, level="corner").add_prefix(f"{s}_")
        res = res.join(part, on="fight_url_id")

    # --- order, ids, flags
    res = res.sort_values(["date", "bout_order"]).reset_index(drop=True)
    res.insert(0, "fight_id", np.arange(len(res)))
    res["red_win"] = res["winner"].map({"Red": 1, "Blue": 0})
    res["corners_reliable"] = res["date"] >= CORNERS_RELIABLE_FROM
    fid = res.set_index("fight_url_id")["fight_id"]

    rounds["fight_id"] = rounds["fight_url_id"].map(fid)
    names = res.set_index("fight_id")[["r_fighter", "b_fighter"]]
    rounds["fighter"] = np.where(rounds.corner == "Red", rounds.fight_id.map(names.r_fighter),
                                 rounds.fight_id.map(names.b_fighter))
    rounds = rounds.sort_values(["fight_id", "round", "corner"]).reset_index(drop=True)

    judges = [(f, w, j, lo, hi) for f, d, m, w in zip(res.fight_id, res.DETAILS, res.METHOD, res.winner)
              if str(m).startswith("Decision") for j, lo, hi in parse_scorecards(d)]
    judges = pd.DataFrame(judges, columns=["fight_id", "winner", "judge", "loser_score", "winner_score"])
    red_won = judges["winner"] == "Red"
    decided = judges["winner"].isin(["Red", "Blue"])
    judges["r_score"] = np.where(red_won, judges.winner_score, judges.loser_score)
    judges["b_score"] = np.where(red_won, judges.loser_score, judges.winner_score)
    judges.loc[~decided, ["r_score", "b_score"]] = np.nan
    judges = judges.drop(columns="winner")

    keep = (["fight_id", "fight_url_id", "date", "event", "location", "bout_order",
             "r_fighter", "b_fighter", "winner", "red_win", "corners_reliable", "weight_class",
             "weight_class_raw", "gender", "is_title_bout", "method", "method_group", "method_detail",
             "finish_round", "total_rounds", "time_sec", "time_format", "fight_seconds", "referee"]
            + [f"{s}_{c}" for s in "rb" for c in stat_cols]
            + [f"{s}_{c}" for s in "rb" for c in ["height", "reach", "weight", "stance",
                                                    "age_at_fight", "fighter_url_id"]])
    report = {"undated_events": sorted(undated["EVENT"].unique()), "undated_fights": len(undated),
              "name_aliases": alias,
              "split_names": sorted({n for n in res.r_fighter.tolist() + res.b_fighter.tolist()
                                     if re.search(r"\(\d{4}\)$|\(#\d+\)$", str(n))})}
    return res[keep], rounds, judges, directory, report


def run(snapshot=None, out_dir="data/processed"):
    snapshot = snapshot or latest_snapshot()
    fights, rounds, judges, directory, report = build(snapshot)
    ff = build_fighter_fights(fights)

    assert fights["fight_id"].is_unique and fights["date"].is_monotonic_increasing
    assert len(ff) == 2 * len(fights)
    assert (ff.groupby("fight_id").result.apply(lambda r: "".join(sorted(r)))
              .isin(["LW", "DD", "NN"])).all()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fights.to_parquet(out / "fights.parquet", index=False)
    ff.to_parquet(out / "fighter_fights.parquet", index=False)
    rounds.to_parquet(out / "fight_rounds.parquet", index=False)
    judges.to_parquet(out / "judge_scores.parquet", index=False)
    directory.to_parquet(out / "fighters.parquet", index=False)
    from ufc import elo  # local import: elo depends on the tables written above
    elo.run(fights, out_dir)
    return fights, ff, rounds, judges, report


if __name__ == "__main__":
    fights, ff, rounds, judges, report = run()
    print(fights.shape, ff.shape, rounds.shape, judges.shape)
    for k, v in report.items():
        print(k, v)
