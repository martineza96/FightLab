"""Cleaning: turn the raw ufcstats scrapes into one tidy fights table + a long fighter-fight table.

Sources (data/raw/):
  large_dataset.csv   one row per decided fight, rich stats, NO date
  medium_dataset.csv  one row per fight incl. draws, HAS date, winner listed first
  fighter_stats.csv   one row per fighter, career snapshot at scrape time
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# The data was scraped right after this event (last completed event in the files).
# Any "career" or "age" value in the raw files is as of this date, NOT as of the fight.
SNAPSHOT_DATE = pd.Timestamp("2024-03-23")

# Before this date ufcstats lists the WINNER in the red slot for every fight (red wins 100%),
# so "corner" encodes the result. Those fights are fine for fighter history but not as targets.
CORNERS_RELIABLE_FROM = pd.Timestamp("2010-03-21")

# Per-fighter in-fight stats (known only AFTER the fight -> leakage if used directly).
FIGHT_STATS = ["kd", "sig_str", "sig_str_att", "str", "str_att", "td", "td_att",
               "sub_att", "rev", "ctrl_sec"]

# Strike breakdowns (by target and position), available in the newer data source only.
EXTRA_STATS = [f"{k}{x}" for k in ["head", "body", "leg", "distance", "clinch", "ground"]
               for x in ["", "_att"]]

# Order matters: check "Light Heavyweight" / "Super Heavyweight" before "Heavyweight".
DIVISIONS = ["Strawweight", "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
             "Welterweight", "Middleweight", "Light Heavyweight", "Super Heavyweight",
             "Heavyweight", "Catch Weight", "Open Weight"]

METHOD_GROUP = {
    "Decision - Unanimous": "DEC", "Decision - Split": "DEC", "Decision - Majority": "DEC",
    "KO/TKO": "KO/TKO", "TKO - Doctor's Stoppage": "KO/TKO",
    "Submission": "SUB", "DQ": "Other",
    # medium_dataset spellings
    "U-DEC": "DEC", "S-DEC": "DEC", "M-DEC": "DEC", "SUB": "SUB", "Other": "Other",
}


def fix_mojibake(s):
    """Repair UTF-8 text that was mis-decoded as cp1251 (e.g. 'ClГЎudia' -> 'Cláudia')."""
    if not isinstance(s, str) or not re.search("[Ѐ-ӿ]", s):
        return s
    try:
        return s.encode("cp1251").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def norm_name(s):
    """Join key for names/events: repair encoding, strip accents, collapse whitespace, lowercase."""
    s = fix_mojibake(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().lower()


def clean_weight_class(wc):
    """'UFC Women's Flyweight Title' -> "Women's Flyweight"; old tournaments -> 'Open Weight'."""
    prefix = "Women's " if "Women" in wc else ""
    for d in DIVISIONS:
        if d in wc:
            return prefix + d
    return "Open Weight"  # e.g. 'UFC 2 Tournament Title', 'UFC Superfight Championship'


def load_raw(raw_dir):
    raw_dir = Path(raw_dir)
    large = pd.read_csv(raw_dir / "large_dataset.csv")
    medium = pd.read_csv(raw_dir / "medium_dataset.csv")
    fighters = pd.read_csv(raw_dir / "fighter_stats.csv")
    return large, medium, fighters


def _fight_key(event, a, b):
    return norm_name(event) + "|" + "|".join(sorted([norm_name(a), norm_name(b)]))


def build_fights(large, medium, fighters):
    """One row per fight (decided fights + draws), dated, chronologically ordered, with fight_id."""
    # 85 medium rows are fully empty scrape failures ("Fight was not properly finished").
    medium = medium.dropna(subset=["event"]).copy()
    for c in ["event", "r_fighter", "b_fighter", "location"]:
        medium[c] = medium[c].map(fix_mojibake)
    medium["date"] = pd.to_datetime(medium["date"], format="%m/%d/%Y")
    medium["key"] = [_fight_key(e, a, b) for e, a, b in zip(medium.event, medium.r_fighter, medium.b_fighter)]
    # Row position inside an event: raw files list the main event first.
    medium["bout_order"] = medium.groupby("event").cumcount(ascending=False)

    # --- decided fights: large_dataset + date/location from medium
    f = large.rename(columns={"event_name": "event"}).copy()
    f["key"] = [_fight_key(e, a, b) for e, a, b in zip(f.event, f.r_fighter, f.b_fighter)]
    f = f.merge(medium[["key", "date", "location", "bout_order"]], on="key", how="left", validate="1:1")
    assert f["date"].notna().all(), "some large_dataset fights did not get a date"

    # --- draws: only in medium, with a reduced stat set
    d = medium[medium.status == "draw"].copy()
    d = d.rename(columns={"r_str": "r_sig_str", "b_str": "b_sig_str", "r_sub": "r_sub_att",
                          "b_sub": "b_sub_att", "round": "finish_round"})
    d["winner"] = "Draw"
    d["gender"] = np.where(d.weight_class.str.contains("Women"), "Women", "Men")
    d["finish_round"] = d["finish_round"].astype(int)
    mins, secs = d["time"].str.split(":", expand=True).astype(int).T.values
    d["time_sec"] = mins * 60 + secs
    d["total_rounds"] = d["finish_round"]          # decisions always go the full distance
    d["is_title_bout"] = 0                          # not recorded for draws in this scrape
    # A few names appear twice in fighter_stats (old duplicate scrapes); keep the first.
    snap = fighters.drop_duplicates("name").set_index("name")
    for side in "rb":
        for col in ["height", "weight", "reach", "stance", "age"]:
            d[f"{side}_{col}"] = d[f"{side}_fighter"].map(snap[col])

    keep = (["event", "date", "location", "bout_order", "r_fighter", "b_fighter", "winner",
             "weight_class", "gender", "is_title_bout", "method", "finish_round", "total_rounds",
             "time_sec", "referee"]
            + [f"{s}_{c}" for s in "rb" for c in FIGHT_STATS]
            + [f"{s}_{c}" for s in "rb" for c in ["height", "weight", "reach", "stance", "age"]])
    fights = pd.concat([f, d.reindex(columns=keep)], ignore_index=True)[keep]

    # Name is our only fighter ID, so two different people with one name would share a history.
    # Known case: two "Bruno Silva"s (a middleweight and a flyweight); split them by weight.
    for s in "rb":
        bs = fights[f"{s}_fighter"] == "Bruno Silva"
        fights.loc[bs, f"{s}_fighter"] = np.where(fights.loc[bs, f"{s}_weight"] > 70,
                                                  "Bruno Silva (MW)", "Bruno Silva (FLW)")

    # --- tidy columns
    fights["weight_class_raw"] = fights["weight_class"]
    fights["weight_class"] = fights["weight_class"].map(clean_weight_class)
    fights["method_group"] = fights["method"].map(METHOD_GROUP)
    assert fights["method_group"].notna().all(), fights.loc[fights.method_group.isna(), "method"].unique()
    # Earliest UFCs had no rounds (one unlimited period): treat as a single-round fight.
    fights["total_rounds"] = fights["total_rounds"].fillna(1).astype(int)
    fights["fight_seconds"] = (fights["finish_round"] - 1) * 300 + fights["time_sec"]
    fights.loc[fights.total_rounds == 1, "fight_seconds"] = fights["time_sec"]
    for s in "rb":
        fights[f"{s}_stance"] = fights[f"{s}_stance"].replace({"Sideways": "Open Stance"})
        # Age in the raw data is age at SNAPSHOT_DATE -> shift back to the fight date (±1 yr).
        fights[f"{s}_age_at_fight"] = (fights[f"{s}_age"]
                                       - (SNAPSHOT_DATE - fights["date"]).dt.days / 365.25).round(1)
        fights = fights.drop(columns=f"{s}_age")

    # Chronological order. bout_order breaks same-night ties (early tournaments had
    # fighters compete several times in one night).
    fights = fights.sort_values(["date", "bout_order"]).reset_index(drop=True)
    fights.insert(0, "fight_id", np.arange(len(fights)))
    fights["red_win"] = fights["winner"].map({"Red": 1, "Blue": 0})  # NaN for draws
    fights["corners_reliable"] = fights["date"] >= CORNERS_RELIABLE_FROM
    return fights


def build_fighter_fights(fights):
    """Long table: 2 rows per fight (one per fighter) with own + opponent stats.

    This is the backbone for historical features: group by fighter, sort by fight_id,
    and aggregate only rows strictly before the current fight.
    """
    shared = ["fight_id", "date", "event", "weight_class", "gender", "is_title_bout",
              "method_group", "finish_round", "total_rounds", "fight_seconds"]
    rows = []
    for me, opp, my_color in [("r", "b", "Red"), ("b", "r", "Blue")]:
        part = fights[shared].copy()
        part["corner"] = my_color
        part["fighter"] = fights[f"{me}_fighter"]
        part["opponent"] = fights[f"{opp}_fighter"]
        part["result"] = np.select([fights.winner == my_color, fights.winner == "Draw",
                                    fights.winner == "NC"], ["W", "D", "N"], "L")
        for c in [c for c in FIGHT_STATS + EXTRA_STATS if f"r_{c}" in fights.columns]:
            part[c] = fights[f"{me}_{c}"]
            part[f"opp_{c}"] = fights[f"{opp}_{c}"]
        for c in ["height", "reach", "stance", "age_at_fight"]:
            part[c] = fights[f"{me}_{c}"]
        rows.append(part)
    return pd.concat(rows, ignore_index=True).sort_values(["fight_id", "corner"]).reset_index(drop=True)


def run(raw_dir="data/raw", out_dir="data/processed"):
    large, medium, fighters = load_raw(raw_dir)
    fights = build_fights(large, medium, fighters)
    ff = build_fighter_fights(fights)

    assert fights["fight_id"].is_unique
    assert len(ff) == 2 * len(fights)
    assert (ff.groupby("fight_id").result.apply(lambda r: "".join(sorted(r))).isin(["LW", "DD", "NN"])).all()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fights.to_parquet(out / "fights.parquet", index=False)
    ff.to_parquet(out / "fighter_fights.parquet", index=False)
    return fights, ff


if __name__ == "__main__":
    fights, ff = run()
    print(fights.shape, ff.shape)
