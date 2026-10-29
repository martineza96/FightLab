"""Upcoming UFC cards from ESPN's public scoreboard JSON, matched to our fighter directory.

ESPN lists each event's bouts in card order (main event last), the number of rounds,
each fighter's overall pro record and an athlete id. ESPN does not say who is in the red
corner, so predictions for these bouts are corner-neutral.

Responses are cached on disk (default 6 hours) so the app keeps working offline after one
fetch, and so tests can run without the network.
"""
import difflib
import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

from ufc.ingest import MANUAL_ALIASES
from ufc.ingest import _name_key as _ascii_key

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
ATHLETE = "https://site.web.api.espn.com/apis/common/v3/sports/mma/athletes/{id}"
HEADERS = {"User-Agent": "ufc-dataset-local-app/1.0 (personal project)"}


def _cached_get(url, path, max_age_hours, params=None):
    path = Path(path)
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600:
        return json.loads(path.read_text())
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return data
    except Exception:
        if path.exists():                     # stale cache beats nothing
            return json.loads(path.read_text())
        raise


def _name_key(name):
    """Accent-insensitive key: 'Jiří Procházka' -> 'jiriprochazka'."""
    plain = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return _ascii_key(plain)


def _weight_class(abbrev):
    if not abbrev:
        return None
    w = abbrev.replace("W ", "Women's ") if abbrev.startswith("W ") else abbrev
    return {"Catchweight": "Catch Weight"}.get(w, w)


def _inches(text):
    """'5\\' 11"' -> cm, '72"' -> cm."""
    if not text:
        return None
    m = re.match(r"(\d+)'\s*(\d+)", text)
    if m:
        return (int(m.group(1)) * 12 + int(m.group(2))) * 2.54
    m = re.match(r"([\d.]+)", text)
    return float(m.group(1)) * 2.54 if m else None


class Matcher:
    """Map an ESPN display name to our directory name (or None = not in our data)."""

    def __init__(self, directory):
        d = directory.reset_index() if "name" not in directory.columns else directory
        self.names = d["name"].tolist()
        self.division = dict(zip(d["name"], d["weight_class"]))
        self.by_key = {}
        for n in self.names:
            base = re.sub(r"\s*\(\d{4}\)$", "", n)          # "Bruno Silva (1989)" -> "Bruno Silva"
            self.by_key.setdefault(_name_key(base), []).append(n)
        reverse = {v: k for k, v in MANUAL_ALIASES.items()}
        self.alias = {**MANUAL_ALIASES, **reverse}

    def __call__(self, name, weight_class=None):
        cands = self.by_key.get(_name_key(name)) or self.by_key.get(_name_key(self.alias.get(name, "")))
        if not cands:
            close = difflib.get_close_matches(_name_key(name), list(self.by_key), n=3, cutoff=0.9)
            cands = [n for k in close for n in self.by_key[k]]
        if not cands:
            return None
        if len(cands) > 1 and weight_class:
            same = [c for c in cands if self.division.get(c) == weight_class]
            cands = same or cands
        return cands[0]


def athlete_profile(athlete_id, cache_dir, max_age_hours=24 * 7):
    """dob / height / reach / stance / headshot / age from ESPN's athlete endpoint (cached)."""
    try:
        d = _cached_get(ATHLETE.format(id=athlete_id), Path(cache_dir) / f"athlete_{athlete_id}.json", max_age_hours)
    except Exception:
        return {}
    a = d.get("athlete", d)
    dob = pd.to_datetime(a.get("displayDOB"), errors="coerce", format="%d/%m/%Y")
    return {"dob": None if pd.isna(dob) else dob.date().isoformat(),
            "height": _inches(a.get("displayHeight")), "reach": _inches(a.get("displayReach")),
            "stance": (a.get("stance") or {}).get("text"), "age": a.get("age"),
            "headshot": (a.get("headshot") or {}).get("href")}


def _parse_events(data):
    """ESPN scoreboard JSON -> plain event dicts (bouts in card order, main event first)."""
    events = []
    for e in data.get("events", []):
        comps = e.get("competitions", [])
        venue = (e.get("venues") or [{}])[0]
        bouts = []
        for i, c in enumerate(reversed(comps)):          # main event first
            fighters = []
            for x in sorted(c.get("competitors", []), key=lambda x: x.get("order", 0)):
                a = x.get("athlete", {})
                rec = next((r.get("summary") for r in x.get("records", []) if r.get("type") == "total"), None)
                fighters.append({"espn_id": x.get("id"), "name": a.get("displayName"),
                                 "country": (a.get("flag") or {}).get("alt"), "record": rec,
                                 "winner": x.get("winner")})
            rounds = int(((c.get("format") or {}).get("regulation") or {}).get("periods") or 3)
            st = c.get("status") or {}
            status = (st.get("type") or {}).get("description")
            result = None
            if status == "Final" and st.get("period"):
                # ESPN gives no method, but the clock tells a decision (full distance) from a finish
                result = {"round": int(st["period"]), "clock": st.get("displayClock"),
                          "decision": int(st["period"]) == rounds and st.get("displayClock") == "5:00"}
            bouts.append({"id": c.get("id"), "order": i + 1, "main_event": i == 0, "rounds": rounds,
                          "weight_class": _weight_class((c.get("type") or {}).get("abbreviation")),
                          "status": status, "result": result, "fighters": fighters})
        events.append({"id": e.get("id"), "name": e.get("name"), "short_name": e.get("shortName"),
                       "date": e.get("date"), "venue": venue.get("fullName"),
                       "city": (venue.get("address") or {}).get("city"),
                       "country": (venue.get("address") or {}).get("country"),
                       "status": ((e.get("status") or {}).get("type") or {}).get("description"),
                       "bouts": bouts})
    return sorted(events, key=lambda ev: ev["date"] or "")


def fetch_events(cache_dir="data/processed/cache", days=90, max_age_hours=6, today=None):
    """Upcoming events in the next `days` days, parsed into plain dicts (bouts in card order)."""
    today = pd.Timestamp(today or pd.Timestamp.now(tz="UTC").date())
    end = today + pd.Timedelta(days=days)
    data = _cached_get(SCOREBOARD, Path(cache_dir) / "espn_scoreboard.json", max_age_hours,
                       params={"dates": f"{today:%Y%m%d}-{end:%Y%m%d}"})
    return _parse_events(data)


def fetch_recent_results(cache_dir="data/processed/cache", days=21, max_age_hours=6, today=None):
    """Completed events from the last `days` days, newest first (bouts carry winners + result)."""
    today = pd.Timestamp(today or pd.Timestamp.now(tz="UTC").date())
    start = today - pd.Timedelta(days=days)
    data = _cached_get(SCOREBOARD, Path(cache_dir) / "espn_results.json", max_age_hours,
                       params={"dates": f"{start:%Y%m%d}-{today:%Y%m%d}"})
    return [e for e in reversed(_parse_events(data))
            if e["status"] == "Final" and "contender series" not in (e["name"] or "").lower()]


def predict_card(event, predictor, matcher, cache_dir="data/processed/cache", profiles=True):
    """Attach our directory match + a prediction to every bout of one event."""
    date = pd.Timestamp(event["date"]).tz_localize(None).normalize() if event.get("date") else None
    out = dict(event, bouts=[])
    for b in event["bouts"]:
        if len(b["fighters"]) != 2:
            continue
        sides = []
        for f in b["fighters"]:
            ours = matcher(f["name"], b["weight_class"])
            prof = athlete_profile(f["espn_id"], cache_dir) if profiles and f.get("espn_id") else {}
            prof = {**prof, "country": f.get("country")}           # ESPN flag: used when we have no origin
            sides.append({**f, "ours": ours, "in_data": ours is not None, "headshot": prof.get("headshot"),
                          "profile": prof})
        a, c = sides
        pred = predictor.predict(a["ours"] or a["name"], c["ours"] or c["name"], date=date, rounds=b["rounds"],
                                 red_profile=a["profile"], blue_profile=c["profile"],
                                 weight_class=b["weight_class"], country=event.get("country"))
        out["bouts"].append({**b, "fighters": [{k: v for k, v in s.items() if k != "profile"} for s in sides],
                             "prediction": pred})
    return out
